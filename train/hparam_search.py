"""
双模型超参数对比实验 v2
基于 train_dual_or_fusion（恢复版）进行系统性超参数搜索

搜索空间（按用户指定）：
  IOC 图模型:
    ioc_hidden_dim  : [64, 128, 256]
    ioc_num_layers  : [2, 3, 4]
    ioc_num_bases   : [8, 16, 32]
    ioc_dropout     : [0.2, 0.3, 0.4]
    ioc_lr          : [1e-4, 3e-4, 5e-4]

  TTP Transformer:
    ttp_d_model     : [64, 128, 256]
    ttp_num_layers  : [2, 3, 4]
    ttp_nhead       : [4, 8, 16]  (自动跳过 d_model % nhead != 0 的组合)
    ttp_dropout     : [0.1, 0.2, 0.3]
    ttp_lr          : [1e-4, 3e-4, 5e-4]

训练设置：完整 5-fold，与 train_dual_or_fusion.py 完全一致
  IOC: 最多 150 epoch，patience=30，ReduceLROnPlateau(factor=0.5, patience=15)
  TTP: 最多 200 epoch，patience=30，ReduceLROnPlateau(factor=0.5, patience=10)

搜索策略（三阶段）：
  Phase-1A  IOC 单因素扫描：固定 TTP 基线，逐一遍历 IOC 每个超参的所有候选值
  Phase-1B  TTP 单因素扫描：固定 IOC 基线，逐一遍历 TTP 每个超参的所有候选值
  Phase-2   Grid Search：对核心维度
            (ioc_hidden_dim × ioc_num_layers × ttp_d_model × ttp_num_layers) 做全组合，
            其余超参取 Phase-1 各因素的最优值

输出（每次运行生成带时间戳的目录）：
  results/hparam_<timestamp>/
    sweep_ioc.json       IOC 单因素扫描全部结果
    sweep_ttp.json       TTP 单因素扫描全部结果
    grid_search.json     Grid Search 全部结果
    summary.json         所有阶段汇总 + 最优配置
    progress.log         实时写入的进度日志（可 tail -f 观察）

运行示例：
  python hparam_search.py --ioc-data ./apt_kg_ioc.pt --ttp-data ./apt_kg_ttp.pt
  python hparam_search.py --search ioc                  # 只跑 IOC 扫描
  python hparam_search.py --search ttp                  # 只跑 TTP 扫描
  python hparam_search.py --search grid                 # 只跑 Grid Search
  python hparam_search.py --device cuda:0               # 指定 GPU
"""

import os, sys, json, itertools, argparse, time, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.path.dirname(__file__))
from train_rgcn_embedding import (
    EmbeddingGraphConfig,
    EmbeddingGraphDataProcessor,
    FocalLoss,
    filter_top_k_classes,
)

# ══════════════════════════════════════════════════════════════════════
# 全局常量（与 train_dual_or_fusion.py 保持一致）
# ══════════════════════════════════════════════════════════════════════
N_FOLDS       = 5
VAL_RATIO     = 0.15
SEED          = 42
TOP_K_CLASSES = 15
IOC_MAX_EPOCH = 150
TTP_MAX_EPOCH = 200
PATIENCE      = 30
IOC_BATCH     = 128
TTP_BATCH     = 64
NUM_NEIGHBORS = [30, 20]

# ══════════════════════════════════════════════════════════════════════
# 搜索空间
# ══════════════════════════════════════════════════════════════════════
BASELINE_IOC = dict(
    hidden_dim = 256,
    num_layers = 3,
    num_bases  = 8,
    dropout    = 0.2,
    lr         = 5e-4,
)
BASELINE_TTP = dict(
    d_model    = 256,
    nhead      = 8,
    num_layers = 3,
    dropout    = 0.2,
    lr         = 3e-4,
)

IOC_SWEEP = dict(
    hidden_dim = [64, 128, 256],
    num_layers = [2, 3, 4],
    num_bases  = [8, 16, 32],
    dropout    = [0.2, 0.3, 0.4],
    lr         = [1e-4, 3e-4, 5e-4],
)
TTP_SWEEP = dict(
    d_model    = [64, 128, 256],
    nhead      = [4, 8, 16],
    num_layers = [2, 3, 4],
    dropout    = [0.1, 0.2, 0.3],
    lr         = [1e-4, 3e-4, 5e-4],
)

GRID_IOC_HIDDEN = [64, 128, 256]
GRID_IOC_LAYERS = [2, 3, 4]
GRID_TTP_DMODEL = [64, 128, 256]
GRID_TTP_LAYERS = [2, 3, 4]


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _select_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0')
    return torch.device('cpu')


def _build_edge_attr_dict(batch) -> dict:
    result = {}
    for et in batch.edge_types:
        if not hasattr(batch[et], 'edge_attr'):
            continue
        ea = batch[et].edge_attr
        if ea is None or ea.numel() == 0:
            continue
        result[et] = ea
    return result


def _build_num_nodes_dict(batch) -> dict:
    return {nt: batch[nt].num_nodes for nt in batch.node_types}


def _pack_ttp(idx_list: list, ttp_seqs, phase_seqs,
              subseq_feats, global_feats, device) -> Tuple:
    seqs   = [ttp_seqs[i] for i in idx_list]
    phases = [phase_seqs[i] for i in idx_list] if phase_seqs else None

    padded = pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in seqs],
        batch_first=True, padding_value=0
    ).to(device)
    mask = torch.zeros_like(padded)
    for i, s in enumerate(seqs):
        mask[i, :len(s)] = 1

    phase_t = None
    if phases:
        phase_t = pad_sequence(
            [torch.tensor(p, dtype=torch.long) for p in phases],
            batch_first=True, padding_value=0
        ).to(device)

    it   = torch.tensor(idx_list)
    subq = subseq_feats[it].to(device) if subseq_feats is not None else None
    glob = global_feats[it].to(device) if global_feats is not None else None

    return padded, phase_t, subq, glob, mask


def _ioc_loader(ioc_data, idx, shuffle: bool):
    return NeighborLoader(
        ioc_data,
        num_neighbors = NUM_NEIGHBORS,
        batch_size    = IOC_BATCH,
        input_nodes   = ('EVENT', torch.tensor(idx, dtype=torch.long)),
        shuffle       = shuffle,
    )


# ══════════════════════════════════════════════════════════════════════
# 模型定义（与 train_dual_or_fusion.py 完全一致）
# ══════════════════════════════════════════════════════════════════════

class IOCClassifier(nn.Module):
    def __init__(self, metadata, input_dims, hidden_dim=256, num_layers=4,
                 num_bases=16, num_classes=15, dropout=0.3,
                 edge_type_embed_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.dropout    = dropout

        self.input_projs = nn.ModuleDict()
        for nt, dim in input_dims.items():
            if dim is not None and dim > 0:
                self.input_projs[nt] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                )

        self.edge_type_embedding = nn.Embedding(len(metadata[1]), edge_type_embed_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_type_embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_type_map = {et: i for i, et in enumerate(self.edge_types)}

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, len(metadata[1]), num_bases=num_bases)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict):
        dev = x_dict[list(x_dict.keys())[0]].device if x_dict else 'cpu'

        h_dict = {}
        for nt in self.node_types:
            if nt in x_dict and x_dict[nt] is not None:
                h_dict[nt] = (
                    self.input_projs[nt](x_dict[nt]) if nt in self.input_projs
                    else torch.zeros(max(num_nodes_dict.get(nt, 1), 1),
                                     self.hidden_dim, device=dev)
                )
            else:
                h_dict[nt] = torch.zeros(max(num_nodes_dict.get(nt, 1), 1),
                                          self.hidden_dim, device=dev)

        parts, offsets, cur = [], {}, 0
        for nt in self.node_types:
            parts.append(h_dict[nt]); offsets[nt] = cur; cur += h_dict[nt].shape[0]
        x_all = torch.cat(parts, 0)

        ei_l, et_l, ew_l = [], [], []
        for ek, ei in edge_index_dict.items():
            if ei is None or ei.numel() == 0:
                continue
            s, _, d = ek
            ni = ei.clone(); ni[0] += offsets[s]; ni[1] += offsets[d]
            ei_l.append(ni)
            et_l.append(torch.full((ei.shape[1],), self.edge_type_map[ek],
                                    dtype=torch.long, device=dev))
            ea = edge_attr_dict.get(ek)
            ew_l.append(
                ea[:, 1].to(dev)
                if ea is not None and ea.numel() > 0 and ea.shape[1] >= 2
                else torch.ones(ei.shape[1], device=dev)
            )

        if not ei_l:
            s = offsets['EVENT']
            return x_all[s: s + h_dict['EVENT'].shape[0]]

        ei_all = torch.cat(ei_l, 1); et_all = torch.cat(et_l, 0); ew_all = torch.cat(ew_l, 0)
        ee  = self.edge_type_embedding(et_all)
        emh = self.edge_mlp(torch.cat([ee, ew_all.unsqueeze(1)], -1))

        h = x_all
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, ei_all, et_all)
            _, dst = ei_all
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]), emh)
            deg = torch.zeros_like(h)
            deg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]), torch.ones_like(emh))
            h_new = h_new + 0.1 * agg / deg.clamp(min=1)
            h_new = norm(h_new); h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h

        s = offsets['EVENT']
        return h[s: s + h_dict['EVENT'].shape[0]]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TTPTransformer(nn.Module):
    def __init__(self, num_techniques=369, d_model=256, nhead=8, num_layers=4,
                 num_classes=15, dropout=0.2, pretrained_embeddings=None,
                 num_phases=14, num_subseq_features=0, global_feature_dim=6):
        super().__init__()
        self.d_model = d_model

        if pretrained_embeddings is not None:
            self.tech_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.float(), padding_idx=0, freeze=False)
            sem_dim = pretrained_embeddings.shape[1]
            self.semantic_proj = (
                nn.Sequential(nn.Linear(sem_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
                if sem_dim != d_model else nn.Identity()
            )
        else:
            self.tech_embedding = nn.Embedding(num_techniques, d_model, padding_idx=0)
            self.semantic_proj  = nn.Identity()

        self.phase_embedding = nn.Embedding(num_phases, 16, padding_idx=0)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + 16, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_encoding = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.use_subseq = num_subseq_features > 0
        self.use_global = global_feature_dim > 0
        if self.use_subseq:
            self.subseq_mlp = nn.Sequential(
                nn.Linear(num_subseq_features, d_model // 2),
                nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout))
        if self.use_global:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_feature_dim, d_model // 4),
                nn.LayerNorm(d_model // 4), nn.GELU(), nn.Dropout(dropout))

        cls_in = d_model
        if self.use_subseq: cls_in += d_model // 2
        if self.use_global:  cls_in += d_model // 4
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, cls_in // 2), nn.LayerNorm(cls_in // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(cls_in // 2, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                if not hasattr(m, 'weight') or m.weight.shape[1] != 384:
                    nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, seqs, phases=None, subseq=None, glob=None, mask=None):
        x = self.semantic_proj(self.tech_embedding(seqs))
        if phases is not None:
            x = self.fusion_proj(torch.cat([x, self.phase_embedding(phases)], -1))
        x = self.pos_encoding(x)
        pad_mask = (mask == 0) if mask is not None else None
        enc = self.transformer(x, src_key_padding_mask=pad_mask)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (enc * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = enc.mean(1)
        feats = [pooled]
        if self.use_subseq and subseq is not None: feats.append(self.subseq_mlp(subseq))
        if self.use_global  and glob  is not None: feats.append(self.global_mlp(glob))
        return self.classifier(torch.cat(feats, -1))


# ══════════════════════════════════════════════════════════════════════
# 单折训练函数（完整训练，与原始脚本一致）
# ══════════════════════════════════════════════════════════════════════

def _train_ioc_fold(model, ioc_data, g_train, g_val,
                    class_weights, lr, device) -> float:
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=15)
    crit = FocalLoss(alpha=class_weights, gamma=3, label_smoothing=0.1)

    best_bacc, best_state, pat = 0.0, None, 0

    for epoch in range(IOC_MAX_EPOCH):
        model.train()
        for batch in _ioc_loader(ioc_data, g_train, shuffle=True):
            batch = batch.to(device); bs = batch['EVENT'].batch_size
            x_d = {nt: batch[nt].x for nt in batch.node_types
                   if nt in ['IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT']}
            eid = {et: batch[et].edge_index for et in batch.edge_types}
            opt.zero_grad()
            emb  = model(x_d, eid, _build_edge_attr_dict(batch), _build_num_nodes_dict(batch))
            loss = crit(model.classifier(emb[:bs]), batch['EVENT'].y[:bs])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for batch in _ioc_loader(ioc_data, g_val, shuffle=False):
                batch = batch.to(device); bs = batch['EVENT'].batch_size
                x_d = {nt: batch[nt].x for nt in batch.node_types
                       if nt in ['IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT']}
                eid = {et: batch[et].edge_index for et in batch.edge_types}
                emb = model(x_d, eid, _build_edge_attr_dict(batch), _build_num_nodes_dict(batch))
                vp.append(model.classifier(emb[:bs]).argmax(1).cpu())
                vl.append(batch['EVENT'].y[:bs].cpu())
        val_bacc = balanced_accuracy_score(torch.cat(vl).numpy(), torch.cat(vp).numpy())
        sch.step(val_bacc)

        if val_bacc > best_bacc:
            best_bacc  = val_bacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    model.load_state_dict(best_state)
    return best_bacc


def _eval_ioc(model, ioc_data, g_test, device):
    model.eval(); ap, al = [], []
    with torch.no_grad():
        for batch in _ioc_loader(ioc_data, g_test, shuffle=False):
            batch = batch.to(device); bs = batch['EVENT'].batch_size
            x_d = {nt: batch[nt].x for nt in batch.node_types
                   if nt in ['IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT']}
            eid = {et: batch[et].edge_index for et in batch.edge_types}
            emb = model(x_d, eid, _build_edge_attr_dict(batch), _build_num_nodes_dict(batch))
            ap.append(model.classifier(emb[:bs]).argmax(1).cpu())
            al.append(batch['EVENT'].y[:bs].cpu())
    return torch.cat(ap).numpy(), torch.cat(al).numpy()


def _train_ttp_fold(model, ttp_seqs, phase_seqs, subseq_feats, global_feats,
                    l_train, l_val, valid_y, class_weights, lr, device,
                    fold_seed: int = 0) -> float:
    emb_params   = list(model.tech_embedding.parameters())
    emb_ids      = {id(p) for p in emb_params}
    other_params = [p for p in model.parameters() if id(p) not in emb_ids]
    opt = torch.optim.AdamW([
        {'params': emb_params,   'lr': lr / 3},
        {'params': other_params, 'lr': lr},
    ], weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=10, min_lr=1e-6)
    crit = FocalLoss(alpha=class_weights, gamma=3, label_smoothing=0.1)

    train_ds     = list(zip(l_train.tolist(), valid_y[l_train].tolist()))
    train_loader = DataLoader(train_ds, batch_size=TTP_BATCH, shuffle=True,
                              collate_fn=lambda b: list(zip(*b)))
    val_packed   = _pack_ttp(l_val.tolist(), ttp_seqs, phase_seqs, subseq_feats, global_feats, device)
    val_labels   = torch.tensor(valid_y[l_val], dtype=torch.long).to(device)

    best_bacc, best_state, pat = 0.0, None, 0
    torch.manual_seed(fold_seed)

    for epoch in range(TTP_MAX_EPOCH):
        model.train()
        for bidx, blabels in train_loader:
            bidx = list(bidx)
            padded, phase_t, subq, glob, mask = _pack_ttp(
                bidx, ttp_seqs, phase_seqs, subseq_feats, global_feats, device)
            by = torch.tensor(list(blabels), dtype=torch.long).to(device)
            opt.zero_grad()
            loss = crit(model(padded, phase_t, subq, glob, mask), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_padded, val_phase, val_subq, val_glob, val_mask = val_packed
        with torch.no_grad():
            vl = model(val_padded, val_phase, val_subq, val_glob, val_mask)
            val_bacc = balanced_accuracy_score(val_labels.cpu().numpy(), vl.argmax(1).cpu().numpy())
        sch.step(val_bacc)

        if val_bacc > best_bacc:
            best_bacc  = val_bacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    model.load_state_dict(best_state)
    return best_bacc


def _eval_ttp(model, ttp_seqs, phase_seqs, subseq_feats, global_feats,
              l_test, valid_y, device):
    model.eval()
    test_padded, test_phase, test_subq, test_glob, test_mask = _pack_ttp(
        l_test.tolist(), ttp_seqs, phase_seqs, subseq_feats, global_feats, device)
    with torch.no_grad():
        preds = model(test_padded, test_phase, test_subq, test_glob,
                      test_mask).argmax(1).cpu().numpy()
    return preds, valid_y[l_test]


# ══════════════════════════════════════════════════════════════════════
# 完整 5-fold 实验
# ══════════════════════════════════════════════════════════════════════

def run_experiment(ioc_data, ttp_data, valid_idx, valid_y,
                   ioc_input_dims, num_classes, class_weights,
                   ioc_cfg, ttp_cfg, device,
                   logger: logging.Logger,
                   exp_tag: str = '') -> Optional[dict]:
    if ttp_cfg['d_model'] % ttp_cfg['nhead'] != 0:
        return None

    ttp_seqs   = ttp_data.get('causal_sequences', ttp_data.get('technique_sequences'))
    phase_seqs = ttp_data.get('phase_sequences', None)
    subseq_f   = ttp_data.get('subseq_features', None)
    global_f   = ttp_data.get('global_features', None)
    pretrained = ttp_data.get('technique_embeddings', None)
    num_phases = ttp_data.get('num_phases', 14)
    num_subseq = ttp_data.get('num_feature_subseq', 0)
    gf_dim     = ttp_data.get('global_feature_dim', 5)
    all_ids    = [i for s in ttp_seqs for i in s]
    num_tech   = max(max(all_ids) + 1 if all_ids else 369,
                     ttp_data.get('num_techniques', 369))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_ioc_accs, fold_ioc_baccs = [], []
    fold_ttp_accs, fold_ttp_baccs = [], []
    fold_or_accs,  fold_or_baccs  = [], []
    fold_or_f1s                   = []
    fold_details                  = []

    for fold, (tmp_train, l_test) in enumerate(skf.split(valid_idx, valid_y)):
        l_train, l_val = train_test_split(
            tmp_train, test_size=VAL_RATIO,
            stratify=valid_y[tmp_train], random_state=SEED)
        g_train = valid_idx[l_train]
        g_val   = valid_idx[l_val]
        g_test  = valid_idx[l_test]

        # IOC
        ioc_model = IOCClassifier(
            metadata=ioc_data.metadata(), input_dims=ioc_input_dims,
            hidden_dim=ioc_cfg['hidden_dim'], num_layers=ioc_cfg['num_layers'],
            num_bases=ioc_cfg['num_bases'],   num_classes=num_classes,
            dropout=ioc_cfg['dropout'],
        ).to(device)
        _train_ioc_fold(ioc_model, ioc_data, g_train, g_val,
                        class_weights, ioc_cfg['lr'], device)
        ioc_preds, test_labels = _eval_ioc(ioc_model, ioc_data, g_test, device)
        ioc_acc  = accuracy_score(test_labels, ioc_preds)
        ioc_bacc = balanced_accuracy_score(test_labels, ioc_preds)

        # TTP
        ttp_model = TTPTransformer(
            num_techniques=num_tech,   d_model=ttp_cfg['d_model'],
            nhead=ttp_cfg['nhead'],    num_layers=ttp_cfg['num_layers'],
            num_classes=num_classes,   dropout=ttp_cfg['dropout'],
            pretrained_embeddings=pretrained,
            num_phases=num_phases, num_subseq_features=num_subseq,
            global_feature_dim=gf_dim,
        ).to(device)
        _train_ttp_fold(ttp_model, ttp_seqs, phase_seqs, subseq_f, global_f,
                        l_train, l_val, valid_y,
                        class_weights, ttp_cfg['lr'], device,
                        fold_seed=SEED + fold)
        ttp_preds, _ = _eval_ttp(ttp_model, ttp_seqs, phase_seqs, subseq_f, global_f,
                                  l_test, valid_y, device)
        ttp_acc  = accuracy_score(test_labels, ttp_preds)
        ttp_bacc = balanced_accuracy_score(test_labels, ttp_preds)

        # OR 融合
        or_preds = np.where(ioc_preds == test_labels, ioc_preds, ttp_preds)
        or_acc   = accuracy_score(test_labels, or_preds)
        or_bacc  = balanced_accuracy_score(test_labels, or_preds)
        or_f1    = f1_score(test_labels, or_preds, average='macro')

        fold_ioc_accs.append(ioc_acc);  fold_ioc_baccs.append(ioc_bacc)
        fold_ttp_accs.append(ttp_acc);  fold_ttp_baccs.append(ttp_bacc)
        fold_or_accs.append(or_acc);    fold_or_baccs.append(or_bacc)
        fold_or_f1s.append(or_f1)

        detail = dict(fold=fold,
                      ioc_acc=round(ioc_acc,4),  ioc_bacc=round(ioc_bacc,4),
                      ttp_acc=round(ttp_acc,4),  ttp_bacc=round(ttp_bacc,4),
                      or_acc=round(or_acc,4),    or_bacc=round(or_bacc,4),
                      or_f1=round(or_f1,4))
        fold_details.append(detail)

        logger.info(
            f'  {exp_tag}  Fold {fold}: '
            f'IOC={ioc_acc:.4f}({ioc_bacc:.4f})  '
            f'TTP={ttp_acc:.4f}({ttp_bacc:.4f})  '
            f'OR={or_acc:.4f}({or_bacc:.4f})  F1={or_f1:.4f}'
        )

    def ms(a): return float(np.mean(a)), float(np.std(a))
    ioc_am, ioc_as  = ms(fold_ioc_accs);  ioc_bm, ioc_bs  = ms(fold_ioc_baccs)
    ttp_am, ttp_as  = ms(fold_ttp_accs);  ttp_bm, ttp_bs  = ms(fold_ttp_baccs)
    or_am,  or_as   = ms(fold_or_accs);   or_bm,  or_bs   = ms(fold_or_baccs)
    or_fm,  or_fs   = ms(fold_or_f1s)

    return dict(
        ioc_acc=round(ioc_am,4),  ioc_bacc=round(ioc_bm,4),
        ioc_acc_std=round(ioc_as,4), ioc_bacc_std=round(ioc_bs,4),
        ttp_acc=round(ttp_am,4),  ttp_bacc=round(ttp_bm,4),
        ttp_acc_std=round(ttp_as,4), ttp_bacc_std=round(ttp_bs,4),
        or_acc=round(or_am,4),    or_bacc=round(or_bm,4),
        or_acc_std=round(or_as,4),   or_bacc_std=round(or_bs,4),
        or_f1=round(or_fm,4),     or_f1_std=round(or_fs,4),
        fold_details=fold_details,
        ioc_cfg=ioc_cfg, ttp_cfg=ttp_cfg, tag=exp_tag,
    )


# ══════════════════════════════════════════════════════════════════════
# 搜索器
# ══════════════════════════════════════════════════════════════════════

class HparamSearcher:

    def __init__(self, ioc_data, ttp_data, valid_idx, valid_y,
                 ioc_input_dims, num_classes, class_weights,
                 device: torch.device, output_dir: Path):
        self.ioc_data       = ioc_data
        self.ttp_data       = ttp_data
        self.valid_idx      = valid_idx
        self.valid_y        = valid_y
        self.ioc_input_dims = ioc_input_dims
        self.num_classes    = num_classes
        self.class_weights  = class_weights
        self.device         = device
        self.output_dir     = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self.results_ioc:  List[dict] = []
        self.results_ttp:  List[dict] = []
        self.results_grid: List[dict] = []

        # 日志（同时写文件和终端）
        log_path = output_dir / 'progress.log'
        fmt      = '%(asctime)s  %(message)s'
        logging.basicConfig(
            level=logging.INFO, format=fmt, datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger('hparam')
        self.logger.info(f'结果目录: {output_dir}')

    def _run(self, ioc_cfg: dict, ttp_cfg: dict, tag: str = '') -> Optional[dict]:
        t0  = time.time()
        res = run_experiment(
            self.ioc_data, self.ttp_data,
            self.valid_idx, self.valid_y,
            self.ioc_input_dims, self.num_classes, self.class_weights,
            ioc_cfg, ttp_cfg, self.device, self.logger, exp_tag=tag,
        )
        elapsed = time.time() - t0
        if res is None:
            self.logger.info(f'  [跳过] {tag}  ← d_model % nhead != 0')
            return None
        res['elapsed_s'] = round(elapsed, 1)
        self.logger.info(
            f'  ▶ {tag:<52} '
            f'OR={res["or_acc"]:.4f}±{res["or_acc_std"]:.4f}  '
            f'OR-B={res["or_bacc"]:.4f}±{res["or_bacc_std"]:.4f}  '
            f'IOC={res["ioc_acc"]:.4f}  TTP={res["ttp_acc"]:.4f}  '
            f'({elapsed:.0f}s)'
        )
        return res

    # ── Phase-1A ─────────────────────────────────────────────────────
    def sweep_ioc(self) -> List[dict]:
        n_total = sum(len(v) for v in IOC_SWEEP.values())
        self.logger.info('\n' + '═'*70)
        self.logger.info('  Phase-1A: IOC 超参单因素扫描'.center(70))
        self.logger.info(f'  固定 TTP 基线: {BASELINE_TTP}'.center(70))
        self.logger.info(f'  共 {n_total} 个配置 × {N_FOLDS}-fold'.center(70))
        self.logger.info('═'*70)

        done = 0
        for factor, values in IOC_SWEEP.items():
            self.logger.info(f'\n  ── {factor} ──')
            for v in values:
                done += 1
                ioc_cfg = {**BASELINE_IOC, factor: v}
                tag     = f'[{done:>2}/{n_total}] IOC {factor}={v}'
                res = self._run(ioc_cfg, BASELINE_TTP, tag=tag)
                if res:
                    res['factor'] = factor; res['value'] = v
                    self.results_ioc.append(res)

        self._save('sweep_ioc.json', self.results_ioc)
        self._print_sweep_table('IOC 单因素扫描', self.results_ioc)
        return self.results_ioc

    # ── Phase-1B ─────────────────────────────────────────────────────
    def sweep_ttp(self) -> List[dict]:
        n_total = sum(len(v) for v in TTP_SWEEP.values())
        self.logger.info('\n' + '═'*70)
        self.logger.info('  Phase-1B: TTP 超参单因素扫描'.center(70))
        self.logger.info(f'  固定 IOC 基线: {BASELINE_IOC}'.center(70))
        self.logger.info(f'  共 {n_total} 个配置 × {N_FOLDS}-fold'.center(70))
        self.logger.info('═'*70)

        done = 0
        for factor, values in TTP_SWEEP.items():
            self.logger.info(f'\n  ── {factor} ──')
            for v in values:
                done += 1
                ttp_cfg = {**BASELINE_TTP, factor: v}
                tag     = f'[{done:>2}/{n_total}] TTP {factor}={v}'
                res = self._run(BASELINE_IOC, ttp_cfg, tag=tag)
                if res:
                    res['factor'] = factor; res['value'] = v
                    self.results_ttp.append(res)

        self._save('sweep_ttp.json', self.results_ttp)
        self._print_sweep_table('TTP 单因素扫描', self.results_ttp)
        return self.results_ttp

    # ── Phase-2 Grid Search ──────────────────────────────────────────
    def grid_search(self,
                    best_ioc_cfg: Optional[dict] = None,
                    best_ttp_cfg: Optional[dict] = None) -> List[dict]:
        base_ioc = {**BASELINE_IOC, **(best_ioc_cfg or {})}
        base_ttp = {**BASELINE_TTP, **(best_ttp_cfg or {})}

        combos  = list(itertools.product(
            GRID_IOC_HIDDEN, GRID_IOC_LAYERS, GRID_TTP_DMODEL, GRID_TTP_LAYERS))
        n_valid = sum(1 for _, _, td, _ in combos if td % base_ttp['nhead'] == 0)

        self.logger.info('\n' + '═'*70)
        self.logger.info('  Phase-2: Grid Search'.center(70))
        self.logger.info(f'  IOC hidden×layers : {GRID_IOC_HIDDEN} × {GRID_IOC_LAYERS}'.center(70))
        self.logger.info(f'  TTP d_model×layers: {GRID_TTP_DMODEL} × {GRID_TTP_LAYERS}'.center(70))
        self.logger.info(f'  nhead={base_ttp["nhead"]}，有效组合: {n_valid} × {N_FOLDS}-fold'.center(70))
        self.logger.info('═'*70)

        done = 0
        for ih, il, td, tl in combos:
            ioc_cfg = {**base_ioc, 'hidden_dim': ih, 'num_layers': il}
            ttp_cfg = {**base_ttp, 'd_model':    td, 'num_layers': tl}
            if td % ttp_cfg['nhead'] != 0:
                self.logger.info(
                    f'  [跳过] IOC(h={ih},L={il}) TTP(d={td},L={tl})'
                    f' ← d_model={td} % nhead={ttp_cfg["nhead"]} != 0')
                continue
            done += 1
            tag = (f'[{done:>3}/{n_valid}] '
                   f'IOC(h={ih},L={il}) TTP(d={td},L={tl})')
            res = self._run(ioc_cfg, ttp_cfg, tag=tag)
            if res:
                self.results_grid.append(res)

        self._save('grid_search.json', self.results_grid)
        self._print_grid_table()
        return self.results_grid

    # ── 输出 ─────────────────────────────────────────────────────────
    def _save(self, fname: str, data: list):
        path = self.output_dir / fname
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=float)
        self.logger.info(f'  → 已保存: {path}')

    def _print_sweep_table(self, title: str, results: List[dict]):
        if not results: return
        self.logger.info(f'\n  {"─"*70}')
        self.logger.info(f'  {title} 汇总'.center(70))
        self.logger.info(f'  {"─"*70}')
        by_factor: Dict[str, List[dict]] = {}
        for r in results:
            by_factor.setdefault(r['factor'], []).append(r)
        for factor, rows in by_factor.items():
            rows_s = sorted(rows, key=lambda x: x['or_bacc'], reverse=True)
            self.logger.info(f'\n  【{factor}】')
            self.logger.info(
                f"    {'值':<8} {'OR-Acc':>7} {'±':>6} {'OR-BAcc':>8} {'±':>6} "
                f"{'OR-F1':>6} {'IOC-Acc':>8} {'TTP-Acc':>8}")
            self.logger.info(f"    {'─'*65}")
            for j, r in enumerate(rows_s):
                marker = '  ◀ 最优' if j == 0 else ''
                self.logger.info(
                    f"    {str(r['value']):<8} "
                    f"{r['or_acc']:>7.4f} ±{r['or_acc_std']:.4f} "
                    f"{r['or_bacc']:>8.4f} ±{r['or_bacc_std']:.4f} "
                    f"{r['or_f1']:>6.4f} "
                    f"{r['ioc_acc']:>8.4f} {r['ttp_acc']:>8.4f}"
                    f"{marker}")

    def _print_grid_table(self):
        if not self.results_grid: return
        rs = sorted(self.results_grid, key=lambda x: x['or_bacc'], reverse=True)
        self.logger.info(f'\n  {"─"*70}')
        self.logger.info('  Grid Search Top-15 (按 OR-BAcc 降序)'.center(70))
        self.logger.info(f'  {"─"*70}')
        self.logger.info(
            f"  {'IOC 配置':<22} {'TTP 配置':<22} "
            f"{'OR-Acc':>7} {'OR-BAcc':>8} {'OR-F1':>6} "
            f"{'IOC':>7} {'TTP':>7}")
        self.logger.info(f"  {'─'*75}")
        for r in rs[:15]:
            ic = r['ioc_cfg']; tc = r['ttp_cfg']
            ioc_s = f"h={ic['hidden_dim']},L={ic['num_layers']}"
            ttp_s = f"d={tc['d_model']},L={tc['num_layers']}"
            self.logger.info(
                f"  {ioc_s:<22} {ttp_s:<22} "
                f"{r['or_acc']:>7.4f} {r['or_bacc']:>8.4f} {r['or_f1']:>6.4f} "
                f"{r['ioc_acc']:>7.4f} {r['ttp_acc']:>7.4f}")

    def print_recommendation(self):
        all_r = self.results_ioc + self.results_ttp + self.results_grid
        if not all_r:
            self.logger.info('  尚无结果。'); return

        best = max(all_r, key=lambda x: x['or_bacc'])
        self.logger.info('\n' + '═'*70)
        self.logger.info('  最优配置推荐'.center(70))
        self.logger.info('═'*70)
        self.logger.info(
            f'\n  全局最优 OR-BAcc : {best["or_bacc"]:.4f} ± {best["or_bacc_std"]:.4f}')
        self.logger.info(
            f'  全局最优 OR-Acc  : {best["or_acc"]:.4f} ± {best["or_acc_std"]:.4f}')
        self.logger.info(f'  来源             : {best["tag"]}')
        self.logger.info(f'  IOC 配置         : {best["ioc_cfg"]}')
        self.logger.info(f'  TTP 配置         : {best["ttp_cfg"]}')

        self.logger.info('\n  ── 各因素独立最优值 ──')
        for label, sweep_res, baseline in [
            ('IOC', self.results_ioc, BASELINE_IOC),
            ('TTP', self.results_ttp, BASELINE_TTP),
        ]:
            if not sweep_res: continue
            fb: Dict[str, dict] = {}
            for r in sweep_res:
                f = r['factor']
                if f not in fb or r['or_bacc'] > fb[f]['or_bacc']: fb[f] = r
            self.logger.info(f'\n  {label}:')
            for f, r in fb.items():
                bv = baseline.get(f, '?')
                changed = '  ← 与基线不同' if r['value'] != bv else ''
                self.logger.info(
                    f'    {f:<15} = {str(r["value"]):<8} '
                    f'(OR-BAcc={r["or_bacc"]:.4f}){changed}')

        self.logger.info('\n  ── 建议 ──')
        self.logger.info('  将上述最优值代入 train_dual_or_fusion.py，跑完整 5-fold 作为最终确认。')
        self.logger.info('═'*70)


# ══════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='双模型超参数对比实验')
    parser.add_argument('--ioc-data',   type=str, default='./apt_kg_ioc.pt')
    parser.add_argument('--ttp-data',   type=str, default='./apt_kg_ttp.pt')
    parser.add_argument('--device',     type=str, default=None)
    parser.add_argument('--search',     type=str, default='all',
                        choices=['all', 'ioc', 'ttp', 'grid'])
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    device = _select_device(args.device)
    torch.manual_seed(SEED); np.random.seed(SEED)
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) if args.output_dir else Path('../v2/results') / f'hparam_{ts}'

    print('=' * 70)
    print('  双模型超参数对比实验'.center(70))
    print(f'  设备: {device}   搜索范围: {args.search}'.center(70))
    print(f'  训练: {N_FOLDS}-fold, IOC≤{IOC_MAX_EPOCH}ep, TTP≤{TTP_MAX_EPOCH}ep, patience={PATIENCE}'.center(70))
    print('=' * 70)

    print(f'\n[数据] 加载 IOC: {args.ioc_data}')
    ioc_data = torch.load(args.ioc_data, weights_only=False)
    print(f'[数据] 加载 TTP: {args.ttp_data}')
    ttp_data = torch.load(args.ttp_data, weights_only=False)

    orig_cls = len(np.unique(ioc_data['EVENT'].y[ioc_data['EVENT'].y != -1].numpy()))
    if TOP_K_CLASSES < orig_cls:
        ioc_data = filter_top_k_classes(ioc_data, TOP_K_CLASSES)
        valid_idx = torch.where(ioc_data['EVENT'].y != -1)[0].numpy()
        raw_seqs  = ttp_data.get('causal_sequences', ttp_data.get('technique_sequences'))
        seq_key   = 'causal_sequences' if 'causal_sequences' in ttp_data else 'technique_sequences'
        ttp_f = {seq_key: [raw_seqs[i] for i in valid_idx],
                 'labels': ttp_data['labels'][valid_idx],
                 'num_techniques': ttp_data.get('num_techniques', 369)}
        for key in ['phase_sequences', 'global_features', 'technique_embeddings',
                    'num_phases', 'global_feature_dim', 'num_feature_subseq',
                    'num_events', 'num_classes', 'apt_classes', 'padding_value',
                    'seq_stats', 'tactic_mapping', 'tactic_phase_order', 'sequence_type']:
            if key not in ttp_data: continue
            val = ttp_data[key]
            if key in ['phase_sequences', 'global_features']:
                ttp_f[key] = (val[valid_idx] if isinstance(val, torch.Tensor)
                              else [val[i] for i in valid_idx])
            else:
                ttp_f[key] = val
        ttp_data = ttp_f
    else:
        valid_idx = torch.where(ioc_data['EVENT'].y != -1)[0].numpy()

    valid_y  = ioc_data['EVENT'].y[valid_idx].numpy()
    num_cls  = len(np.unique(valid_y))
    weights  = compute_class_weight('balanced', classes=np.unique(valid_y), y=valid_y)
    cw       = torch.tensor(weights, dtype=torch.float32).to(device)
    ioc_dims = EmbeddingGraphDataProcessor.build_input_dims(ioc_data)

    print(f'[数据] 类别数={num_cls}, 样本数={len(valid_y)}')
    print(f'[输出] 结果目录: {out_dir}\n')

    # 估算规模
    n_ioc  = sum(len(v) for v in IOC_SWEEP.values())
    n_ttp  = sum(len(v) for v in TTP_SWEEP.values())
    n_grid = len(list(itertools.product(GRID_IOC_HIDDEN, GRID_IOC_LAYERS,
                                         GRID_TTP_DMODEL, GRID_TTP_LAYERS)))
    runs   = {'all': n_ioc + n_ttp + n_grid, 'ioc': n_ioc, 'ttp': n_ttp, 'grid': n_grid}
    n      = runs[args.search]
    print(f'[规模] 本次约 {n} 组 × {N_FOLDS}-fold（nhead 不整除时自动跳过）')
    print(f'[规模] 估算时间: 每组 ≈ 20~40 分钟 → 全程约 {n*20//60}~{n*40//60} 小时\n')

    searcher = HparamSearcher(
        ioc_data, ttp_data, valid_idx, valid_y,
        ioc_dims, num_cls, cw, device, out_dir)

    best_ioc_cfg, best_ttp_cfg = None, None

    if args.search in ('all', 'ioc'):
        res_ioc = searcher.sweep_ioc()
        if res_ioc:
            fb = {}
            for r in res_ioc:
                f = r['factor']
                if f not in fb or r['or_bacc'] > fb[f]['or_bacc']: fb[f] = r
            best_ioc_cfg = {**BASELINE_IOC, **{f: r['value'] for f, r in fb.items()}}

    if args.search in ('all', 'ttp'):
        res_ttp = searcher.sweep_ttp()
        if res_ttp:
            fb = {}
            for r in res_ttp:
                f = r['factor']
                if f not in fb or r['or_bacc'] > fb[f]['or_bacc']: fb[f] = r
            best_ttp_cfg = {**BASELINE_TTP, **{f: r['value'] for f, r in fb.items()}}

    if args.search in ('all', 'grid'):
        searcher.grid_search(best_ioc_cfg=best_ioc_cfg, best_ttp_cfg=best_ttp_cfg)

    searcher.print_recommendation()

    all_results = searcher.results_ioc + searcher.results_ttp + searcher.results_grid
    summary = dict(
        timestamp=ts, device=str(device), n_folds=N_FOLDS,
        baseline_ioc=BASELINE_IOC, baseline_ttp=BASELINE_TTP,
        ioc_sweep=IOC_SWEEP, ttp_sweep=TTP_SWEEP,
        best_ioc_cfg=best_ioc_cfg, best_ttp_cfg=best_ttp_cfg,
        best_overall=max(all_results, key=lambda x: x['or_bacc']) if all_results else None,
        all_results=all_results,
    )
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f'\n[完成] 所有结果已保存至: {out_dir}')
    print(f'  summary.json      — 全部实验汇总 + 最优配置')
    print(f'  sweep_ioc.json    — IOC 单因素扫描')
    print(f'  sweep_ttp.json    — TTP 单因素扫描')
    print(f'  grid_search.json  — Grid Search')
    print(f'  progress.log      — 完整运行日志（可 tail -f 实时跟踪）')


if __name__ == '__main__':
    main()