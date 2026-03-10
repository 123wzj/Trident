"""
APT 双模型增量学习 —— Experience Replay 方法
=============================================

实验设置
--------
方法共 4 种，用于消融对比：
  1. Base (Old Only)       —— 仅旧数据从头训练，不做增量（下界）
  2. Incremental (Replay)  —— 本方法：旧数据回放 + 新数据微调
  3. Full Finetune         —— 旧+新数据合并从头训练（性能上界）
  4. Joint (New Only)      —— 仅新数据从头训练（迁移能力参考）

评估指标（学术标准）
--------------------
  - Accuracy (Acc) / Balanced Accuracy (B-Acc)
  - OR Acc：IOC 预测正确则取 IOC，否则取 TTP（双模型融合上界）
  - FM  (Forgetting Measure)   ：旧任务峰值 − 增量后旧任务，越小越好
  - BT  (Backward Transfer)    ：-FM，越大越好，负=遗忘，正=反向迁移
  - FT  (Forward Transfer)     ：旧模型对新任务零样本能力
  - Int (Intransigence)        ：Full_New − Inc_New，因保旧损失的新任务性能
  - PSR (Plasticity-Stability Ratio)：新旧性能比，越接近 1 越好
  - 耗时节省比、参数量

两点设计目标
------------
  1. 增量训练耗时 < 全量训练
  2. Test_All OR 与 Full Finetune 差距 ≤ 1%

超参（Replay 相关）
-------------------
  replay_ratio    = 0.2   每 step 旧数据 batch 占比
  old_loss_weight = 1.0   旧数据损失权重
  finetune_epochs = 100   增量微调轮数（相比 baseline 150/200 更少）
"""

import json
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              classification_report)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.nn import RGCNConv
from torch_geometric.loader import NeighborLoader
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore', category=UserWarning)

OUTPUT_DIR = Path("../v2/incremental_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_VOCAB_SIZE  = 1024
SAFE_PHASES_SIZE = 64


# ══════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════

@dataclass
class Config:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed:   int = 42

    # IOC 图模型
    ioc_hidden_dim:          int   = 256
    ioc_num_layers:          int   = 3
    ioc_num_bases:           int   = 8
    ioc_dropout:             float = 0.2
    ioc_edge_type_embed_dim: int   = 16

    # TTP 序列模型
    ttp_d_model:    int   = 256
    ttp_num_heads:  int   = 8
    ttp_num_layers: int   = 3
    ttp_dropout:    float = 0.2

    # Baseline 从头训练
    ioc_max_epochs: int   = 150
    ioc_patience:   int   = 50
    ioc_lr:         float = 5e-4
    ttp_max_epochs: int   = 200
    ttp_patience:   int   = 40
    ttp_lr:         float = 3e-4

    # Replay 增量微调
    finetune_epochs:   int   = 100     # 比 baseline 少，保证耗时更短
    finetune_patience: int   = 25
    ioc_finetune_lr:   float = 5e-4
    ttp_finetune_lr:   float = 2e-4
    replay_ratio:      float = 0.2    # 每 step 旧数据 batch 占比
    old_loss_weight:   float = 1.0    # 旧数据损失权重

    # 公共
    batch_size:    int        = 128
    weight_decay:  float      = 1e-4
    num_neighbors: List[int]  = field(default_factory=lambda: [30, 20])

    # 交叉验证
    n_folds:       int   = 5
    val_size:      float = 0.2
    top_k_classes: int   = 15

    def show(self):
        W = 58
        print('\n' + '='*W)
        print('  Replay 增量学习配置'.center(W))
        print('='*W)
        for k, v in [
            ('device',           self.device),
            ('seed',             self.seed),
            ('--- Baseline ---', ''),
            ('ioc_max_epochs',   self.ioc_max_epochs),
            ('ioc_lr',           self.ioc_lr),
            ('ttp_max_epochs',   self.ttp_max_epochs),
            ('ttp_lr',           self.ttp_lr),
            ('--- Replay 微调 ---', ''),
            ('finetune_epochs',  self.finetune_epochs),
            ('ioc_finetune_lr',  self.ioc_finetune_lr),
            ('ttp_finetune_lr',  self.ttp_finetune_lr),
            ('replay_ratio',     self.replay_ratio),
            ('old_loss_weight',  self.old_loss_weight),
            ('--- 公共 ---',     ''),
            ('batch_size',       self.batch_size),
            ('n_folds',          self.n_folds),
        ]:
            if v == '':
                print(f'\n  {k}')
            else:
                print(f'  {k:<26} = {v}')
        print('='*W + '\n')


# ══════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none',
                             weight=self.alpha,
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ══════════════════════════════════════════════════
# IOC 图分类模型（RGCN）
# ══════════════════════════════════════════════════

class IOCClassifier(nn.Module):
    def __init__(self, metadata, input_dims, hidden_dim=256, num_layers=3,
                 num_bases=8, num_classes=15, dropout=0.2,
                 edge_type_embed_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.dropout = dropout

        self.input_projs = nn.ModuleDict()
        for nt, dim in input_dims.items():
            if dim is not None and dim > 0:
                self.input_projs[nt] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim), nn.ReLU())

        self.edge_type_embedding = nn.Embedding(
            len(metadata[1]), edge_type_embed_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_type_embed_dim + 1, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.edge_type_map = {et: i for i, et in enumerate(self.edge_types)}

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim,
                                       num_relations=len(metadata[1]),
                                       num_bases=num_bases))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict):
        ref_dev = (next(iter(x_dict.values())).device
                   if x_dict else torch.device('cpu'))
        h_dict = {}
        for nt in self.node_types:
            x = x_dict.get(nt)
            n = num_nodes_dict.get(nt, 1)
            h_dict[nt] = (self.input_projs[nt](x)
                          if x is not None and nt in self.input_projs
                          else torch.zeros(max(n, 1), self.hidden_dim,
                                           device=ref_dev))

        x_all, offsets, cur = [], {}, 0
        for nt in self.node_types:
            x_all.append(h_dict[nt])
            offsets[nt] = cur
            cur += h_dict[nt].shape[0]
        x_all = torch.cat(x_all, dim=0)

        ei_list, et_list, ew_list = [], [], []
        for ek, ei in edge_index_dict.items():
            if ei is None or ei.numel() == 0:
                continue
            s, _, d = ek
            ni = ei.clone()
            ni[0] += offsets[s]; ni[1] += offsets[d]
            ei_list.append(ni)
            et_list.append(torch.full((ei.shape[1],),
                                      self.edge_type_map[ek],
                                      dtype=torch.long, device=x_all.device))
            ea = edge_attr_dict.get(ek)
            ew_list.append(
                ea[:, 1].to(x_all.device)
                if ea is not None and ea.numel() > 0 and ea.shape[1] >= 2
                else torch.ones(ei.shape[1], device=x_all.device))

        if not ei_list:
            return h_dict['EVENT']

        ei_all = torch.cat(ei_list, dim=1)
        et_all = torch.cat(et_list, dim=0)
        ew_all = torch.cat(ew_list, dim=0)
        ee = self.edge_type_embedding(et_all)
        em = self.edge_mlp(torch.cat([ee, ew_all.unsqueeze(1)], dim=-1))

        h = x_all
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, ei_all, et_all)
            _, dst = ei_all
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]), em)
            deg = torch.zeros_like(h)
            deg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]),
                              torch.ones_like(em))
            h_new = h_new + 0.1 * agg / deg.clamp(min=1)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h

        ev_s = offsets['EVENT']
        return h[ev_s: ev_s + h_dict['EVENT'].shape[0]]


# ══════════════════════════════════════════════════
# TTP 序列模型（Transformer）
# ══════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TTPTransformer(nn.Module):
    def __init__(self, num_classes=15, d_model=256, nhead=8, num_layers=3,
                 dropout=0.2, pretrained_embeddings=None,
                 global_feature_dim=5):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.global_feature_dim = global_feature_dim

        if pretrained_embeddings is not None:
            sem_dim  = pretrained_embeddings.shape[1]
            vocab_sz = pretrained_embeddings.shape[0]
            if vocab_sz < SAFE_VOCAB_SIZE:
                ext = torch.cat([pretrained_embeddings.float(),
                                  torch.zeros(SAFE_VOCAB_SIZE - vocab_sz,
                                              sem_dim)], dim=0)
            else:
                ext = pretrained_embeddings.float()[:SAFE_VOCAB_SIZE]
            self.tech_embedding = nn.Embedding.from_pretrained(ext, freeze=False)
            self.semantic_proj  = (nn.Linear(sem_dim, d_model)
                                   if sem_dim != d_model else nn.Identity())
        else:
            self.tech_embedding = nn.Embedding(SAFE_VOCAB_SIZE, d_model,
                                               padding_idx=0)
            self.semantic_proj  = nn.Identity()

        self.phase_embedding = nn.Embedding(SAFE_PHASES_SIZE, 16, padding_idx=0)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + 16, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_encoding = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.use_global = global_feature_dim > 0
        if self.use_global:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_feature_dim, d_model // 2),
                nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout))

        cls_in = d_model + (d_model // 2 if self.use_global else 0)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, cls_in // 2), nn.LayerNorm(cls_in // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(cls_in // 2, num_classes))

    def forward(self, tech_seqs, phase_seqs, global_feats, mask):
        ts = tech_seqs.clamp(0, SAFE_VOCAB_SIZE - 1)
        x  = self.semantic_proj(self.tech_embedding(ts))
        if phase_seqs is not None:
            ps = phase_seqs.clamp(0, SAFE_PHASES_SIZE - 1)
            x  = self.fusion_proj(
                torch.cat([x, self.phase_embedding(ps)], dim=-1))
        x   = self.pos_encoding(x)
        kpm = (mask == 0) if mask is not None else None
        enc = self.transformer(x, src_key_padding_mask=kpm)
        if mask is not None:
            m      = mask.unsqueeze(-1).float()
            pooled = (enc * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = enc.mean(1)
        feats = [pooled]
        if self.use_global and global_feats is not None:
            feats.append(self.global_mlp(global_feats))
        return self.classifier(torch.cat(feats, dim=-1))


# ══════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════

def safe_pad(sequences, device):
    """CPU 上 pad 后再移到 device，空序列替换为 [0]"""
    if sequences is None:
        return None, None
    safe    = [list(s) if len(s) > 0 else [0] for s in sequences]
    tensors = [torch.tensor(s, dtype=torch.long) for s in safe]
    padded  = pad_sequence(tensors, batch_first=True, padding_value=0)
    mask    = torch.zeros_like(padded)
    for i, s in enumerate(sequences):
        if len(s) > 0:
            mask[i, :min(len(s), padded.shape[1])] = 1
    return padded.to(device), mask.to(device)


def ioc_inputs(batch):
    valid   = {'IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT'}
    x_dict  = {nt: batch[nt].x for nt in batch.node_types if nt in valid}
    ei_dict = {et: batch[et].edge_index for et in batch.edge_types}
    ea_dict = {et: batch[et].edge_attr for et in batch.edge_types
               if hasattr(batch[et], 'edge_attr')
               and batch[et].edge_attr is not None
               and batch[et].edge_attr.numel() > 0}
    nn_dict = {nt: batch[nt].num_nodes for nt in batch.node_types}
    return x_dict, ei_dict, ea_dict, nn_dict


def make_crit(labels, device):
    w  = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    cw = torch.tensor(w, dtype=torch.float).to(device)
    return FocalLoss(alpha=cw, gamma=2, label_smoothing=0.1), cw


def ioc_dims(ioc_data):
    dims = {}
    for nt in ioc_data.node_types:
        x = ioc_data[nt].x
        if x is None or x.shape[1] == 0:
            dims[nt] = None
        elif nt == 'EVENT' and (x != 0).sum() == 0:
            dims[nt] = None
        else:
            dims[nt] = x.shape[1]
    return dims


def build_ioc(ioc_data, num_classes, cfg):
    return IOCClassifier(
        metadata            = ioc_data.metadata(),
        input_dims          = ioc_dims(ioc_data),
        hidden_dim          = cfg.ioc_hidden_dim,
        num_layers          = cfg.ioc_num_layers,
        num_bases           = cfg.ioc_num_bases,
        num_classes         = num_classes,
        dropout             = cfg.ioc_dropout,
        edge_type_embed_dim = cfg.ioc_edge_type_embed_dim)


def build_ttp(ttp_data, num_classes, cfg):
    return TTPTransformer(
        num_classes           = num_classes,
        d_model               = cfg.ttp_d_model,
        nhead                 = cfg.ttp_num_heads,
        num_layers            = cfg.ttp_num_layers,
        dropout               = cfg.ttp_dropout,
        pretrained_embeddings = ttp_data.get('technique_embeddings'),
        global_feature_dim    = ttp_data.get('global_feature_dim', 5))


def ttp_seqs(ttp_data):
    return ttp_data.get('causal_sequences') or ttp_data.get('technique_sequences')


def ioc_loader(ioc_data, node_idx, cfg, shuffle):
    return NeighborLoader(ioc_data, num_neighbors=cfg.num_neighbors,
                          batch_size=cfg.batch_size,
                          input_nodes=('EVENT',
                                       torch.tensor(node_idx, dtype=torch.long)),
                          shuffle=shuffle)


# ══════════════════════════════════════════════════
# 训练：IOC 从头训练
# ══════════════════════════════════════════════════

def train_ioc(model, ioc_data, g_tr, g_va, cw, crit,
              max_ep, lr, patience, cfg, device, tag=''):
    n_cls = len(cw)
    bad   = (ioc_data['EVENT'].y[g_tr].cpu().numpy() < 0) | \
            (ioc_data['EVENT'].y[g_tr].cpu().numpy() >= n_cls)
    if bad.any():
        g_tr = g_tr[~bad]

    tr_ld = ioc_loader(ioc_data, g_tr, cfg, shuffle=True)
    va_ld = ioc_loader(ioc_data, g_va, cfg, shuffle=False)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=cfg.weight_decay)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=15)

    best_bacc, best_sd, pat = 0.0, None, 0
    for ep in range(max_ep):
        model.train()
        for b in tr_ld:
            b   = b.to(device); bs = b['EVENT'].batch_size
            lbl = b['EVENT'].y[:bs]
            if (lbl < 0).any() or (lbl >= n_cls).any():
                continue
            xi, ei, ea, nn_ = ioc_inputs(b)
            loss = crit(model.classifier(model(xi, ei, ea, nn_)[:bs]), lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        bacc = _val_ioc(model, va_ld, device)
        sch.step(bacc)
        if bacc > best_bacc:
            best_bacc = bacc
            best_sd   = deepcopy(model.state_dict())
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f'    [{tag}] Early stop ep{ep+1} B-Acc={best_bacc:.4f}')
                break
        if (ep + 1) % 30 == 0:
            print(f'      [{tag}] Ep{ep+1:03d} B-Acc={bacc:.4f}')

    model.load_state_dict(best_sd)
    return model


def _val_ioc(model, va_ld, device):
    model.eval(); preds, lbls = [], []
    with torch.no_grad():
        for b in va_ld:
            b  = b.to(device); bs = b['EVENT'].batch_size
            xi, ei, ea, nn_ = ioc_inputs(b)
            preds.append(model.classifier(model(xi, ei, ea, nn_)[:bs])
                         .argmax(1).cpu())
            lbls.append(b['EVENT'].y[:bs].cpu())
    return balanced_accuracy_score(torch.cat(lbls).numpy(),
                                    torch.cat(preds).numpy())


# ══════════════════════════════════════════════════
# 训练：TTP 从头训练
# ══════════════════════════════════════════════════

def train_ttp(model, ttp_data, tr_pos, va_pos, labels,
              crit, max_ep, lr, patience, cfg, device, tag=''):
    seqs   = ttp_seqs(ttp_data)
    phases = ttp_data.get('phase_sequences')
    gfeats = ttp_data.get('global_features')
    nc     = model.num_classes

    tr_pos = [i for i in (list(tr_pos) if hasattr(tr_pos, 'tolist')
                           else list(tr_pos))
              if 0 <= int(labels[i]) < nc]
    va_pos = list(va_pos) if hasattr(va_pos, 'tolist') else list(va_pos)

    tr_ld  = DataLoader(list(zip(tr_pos, labels[tr_pos].tolist())),
                        batch_size=32, shuffle=True,
                        collate_fn=lambda b: list(zip(*b)))
    vp, vm = safe_pad([seqs[i] for i in va_pos], device)
    vph, _ = safe_pad([phases[i] for i in va_pos] if phases else None, device)
    vg     = gfeats[torch.tensor(va_pos)].to(device) if gfeats is not None else None
    vlbl   = torch.tensor(labels[va_pos], dtype=torch.long).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=20, min_lr=1e-6)

    best_bacc, best_sd, pat = 0.0, None, 0
    for ep in range(max_ep):
        model.train()
        for bidx, blbl in tr_ld:
            bidx = list(bidx)
            bp, bm = safe_pad([seqs[i] for i in bidx], device)
            bph, _ = safe_pad([phases[i] for i in bidx] if phases else None, device)
            bg     = gfeats[torch.tensor(bidx)].to(device) if gfeats is not None else None
            blab   = torch.tensor(list(blbl), dtype=torch.long).to(device)
            v      = (blab >= 0) & (blab < nc)
            if not v.any(): continue
            if not v.all():
                bp, bm, blab = bp[v], bm[v], blab[v]
                if bph is not None: bph = bph[v]
                if bg  is not None: bg  = bg[v]
            loss = crit(model(bp, bph, bg, bm), blab)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            bacc = balanced_accuracy_score(
                vlbl.cpu().numpy(),
                model(vp, vph, vg, vm).argmax(1).cpu().numpy())
        sch.step(bacc)
        if bacc > best_bacc:
            best_bacc = bacc
            best_sd   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f'    [{tag}] Early stop ep{ep+1} B-Acc={best_bacc:.4f}')
                break
        if (ep + 1) % 20 == 0:
            print(f'      [{tag}] Ep{ep+1:03d} B-Acc={bacc:.4f} '
                  f'LR={opt.param_groups[0]["lr"]:.6f}')

    model.load_state_dict(best_sd)
    return model


# ══════════════════════════════════════════════════
# Replay 增量微调：IOC
# ══════════════════════════════════════════════════

def replay_ioc(base_model, old_ioc, g_tr_old,
               new_ioc, g_tr_new, g_va_new,
               cw, crit, cfg, device, tag=''):
    model = deepcopy(base_model).to(device)
    nc    = len(cw)

    for arr, dat, which in [(g_tr_new, new_ioc, 'new'),
                              (g_tr_old, old_ioc, 'old')]:
        bad = ((dat['EVENT'].y[arr].cpu().numpy() < 0) |
               (dat['EVENT'].y[arr].cpu().numpy() >= nc))
        if bad.any():
            if which == 'new': g_tr_new = g_tr_new[~bad]
            else:              g_tr_old = g_tr_old[~bad]

    old_bs = max(4, int(cfg.batch_size * cfg.replay_ratio))
    new_bs = max(4, cfg.batch_size - old_bs)

    new_ld = ioc_loader(new_ioc, g_tr_new, cfg, shuffle=True)
    old_ld = ioc_loader(old_ioc, g_tr_old, cfg, shuffle=True)
    va_ld  = ioc_loader(new_ioc, g_va_new, cfg, shuffle=False)

    # 临时修改 batch_size（NeighborLoader 不支持动态调整，用独立 loader）
    old_ld = NeighborLoader(old_ioc, num_neighbors=cfg.num_neighbors,
                             batch_size=old_bs,
                             input_nodes=('EVENT',
                                          torch.tensor(g_tr_old, dtype=torch.long)),
                             shuffle=True)
    new_ld = NeighborLoader(new_ioc, num_neighbors=cfg.num_neighbors,
                             batch_size=new_bs,
                             input_nodes=('EVENT',
                                          torch.tensor(g_tr_new, dtype=torch.long)),
                             shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ioc_finetune_lr,
                             weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.finetune_epochs, eta_min=1e-5)

    best_bacc, best_sd, pat = 0.0, None, 0
    for ep in range(cfg.finetune_epochs):
        model.train()
        old_it = iter(old_ld)
        for nb in new_ld:
            nb  = nb.to(device); bs = nb['EVENT'].batch_size
            lbl = nb['EVENT'].y[:bs]
            if (lbl < 0).any() or (lbl >= nc).any(): continue
            xi, ei, ea, nn_ = ioc_inputs(nb)
            loss = crit(model.classifier(model(xi, ei, ea, nn_)[:bs]), lbl)

            try:
                ob = next(old_it)
            except StopIteration:
                old_it = iter(old_ld); ob = next(old_it)
            ob  = ob.to(device); obs = ob['EVENT'].batch_size
            ol  = ob['EVENT'].y[:obs]
            if not ((ol < 0).any() or (ol >= nc).any()):
                oxi, oei, oea, onn = ioc_inputs(ob)
                loss = loss + cfg.old_loss_weight * crit(
                    model.classifier(model(oxi, oei, oea, onn)[:obs]), ol)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        bacc = _val_ioc(model, va_ld, device)
        if bacc > best_bacc:
            best_bacc = bacc; best_sd = deepcopy(model.state_dict()); pat = 0
        else:
            pat += 1
            if pat >= cfg.finetune_patience:
                print(f'    [{tag}] Early stop ep{ep+1} B-Acc={best_bacc:.4f}')
                break
        if (ep + 1) % 20 == 0:
            print(f'      [{tag}] Ep{ep+1:03d} B-Acc={bacc:.4f}')

    model.load_state_dict(best_sd)
    return model


# ══════════════════════════════════════════════════
# Replay 增量微调：TTP
# ══════════════════════════════════════════════════

def replay_ttp(base_model, old_ttp, tr_old, lbl_old,
               new_ttp, tr_new, va_new, lbl_new,
               crit, cfg, device, tag=''):
    model  = deepcopy(base_model).to(device)
    nc     = model.num_classes
    os_    = ttp_seqs(old_ttp); oph = old_ttp.get('phase_sequences')
    ogf    = old_ttp.get('global_features')
    ns_    = ttp_seqs(new_ttp); nph = new_ttp.get('phase_sequences')
    ngf    = new_ttp.get('global_features')

    tr_old = [i for i in list(tr_old) if 0 <= int(lbl_old[i]) < nc]
    tr_new = [i for i in list(tr_new) if 0 <= int(lbl_new[i]) < nc]
    va_new = list(va_new) if hasattr(va_new, 'tolist') else list(va_new)

    nbs = 24; obs = max(4, int(nbs * cfg.replay_ratio))
    new_ld = DataLoader(list(zip(tr_new, lbl_new[tr_new].tolist())),
                        batch_size=nbs, shuffle=True,
                        collate_fn=lambda b: list(zip(*b)))
    old_ld = DataLoader(list(zip(tr_old, lbl_old[tr_old].tolist())),
                        batch_size=obs, shuffle=True,
                        collate_fn=lambda b: list(zip(*b)))

    vp, vm = safe_pad([ns_[i] for i in va_new], device)
    vph, _ = safe_pad([nph[i] for i in va_new] if nph else None, device)
    vg     = ngf[torch.tensor(va_new)].to(device) if ngf is not None else None
    vlbl   = torch.tensor(lbl_new[va_new], dtype=torch.long).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ttp_finetune_lr,
                             weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.finetune_epochs, eta_min=1e-6)

    best_bacc, best_sd, pat = 0.0, None, 0
    for ep in range(cfg.finetune_epochs):
        model.train(); old_it = iter(old_ld)
        for bidx, blbl in new_ld:
            bidx = list(bidx)
            bp, bm = safe_pad([ns_[i] for i in bidx], device)
            bph, _ = safe_pad([nph[i] for i in bidx] if nph else None, device)
            bg     = ngf[torch.tensor(bidx)].to(device) if ngf is not None else None
            blab   = torch.tensor(list(blbl), dtype=torch.long).to(device)
            v      = (blab >= 0) & (blab < nc)
            if not v.any(): continue
            if not v.all():
                bp, bm, blab = bp[v], bm[v], blab[v]
                if bph is not None: bph = bph[v]
                if bg  is not None: bg  = bg[v]
            loss = crit(model(bp, bph, bg, bm), blab)

            try:    oidx, olbl = next(old_it)
            except: old_it = iter(old_ld); oidx, olbl = next(old_it)
            oidx = list(oidx)
            op, om = safe_pad([os_[i] for i in oidx], device)
            oph_, _ = safe_pad([oph[i] for i in oidx] if oph else None, device)
            og      = ogf[torch.tensor(oidx)].to(device) if ogf is not None else None
            olab    = torch.tensor(list(olbl), dtype=torch.long).to(device)
            vo      = (olab >= 0) & (olab < nc)
            if vo.any():
                if not vo.all():
                    op, om, olab = op[vo], om[vo], olab[vo]
                    if oph_ is not None: oph_ = oph_[vo]
                    if og   is not None: og   = og[vo]
                loss = loss + cfg.old_loss_weight * crit(
                    model(op, oph_, og, om), olab)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            bacc = balanced_accuracy_score(
                vlbl.cpu().numpy(),
                model(vp, vph, vg, vm).argmax(1).cpu().numpy())
        if bacc > best_bacc:
            best_bacc = bacc
            best_sd   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= cfg.finetune_patience:
                print(f'    [{tag}] Early stop ep{ep+1} B-Acc={best_bacc:.4f}')
                break
        if (ep + 1) % 20 == 0:
            print(f'      [{tag}] Ep{ep+1:03d} B-Acc={bacc:.4f} '
                  f'LR={opt.param_groups[0]["lr"]:.6f}')

    model.load_state_dict(best_sd)
    return model


# ══════════════════════════════════════════════════
# Full Finetune（性能上界）
# ══════════════════════════════════════════════════

def full_finetune(old_ioc, old_ttp, new_ioc, new_ttp,
                   g_tr_old, g_va_old, ti_old, vi_old,
                   g_tr_new, g_va_new, ti_new, vi_new,
                   lbl_old, lbl_new, cw_old, cw_new,
                   nc, cfg, device):
    cw   = (cw_old[:nc] + cw_new[:nc]) / 2.0
    crit = FocalLoss(alpha=cw, gamma=2, label_smoothing=0.1)

    # IOC
    m_ioc = build_ioc(new_ioc, nc, cfg).to(device)
    opt   = torch.optim.AdamW(m_ioc.parameters(), lr=cfg.ioc_lr,
                               weight_decay=cfg.weight_decay)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=15)
    bad = ((old_ioc['EVENT'].y[g_tr_old].cpu().numpy() < 0) |
           (old_ioc['EVENT'].y[g_tr_old].cpu().numpy() >= nc))
    if bad.any(): g_tr_old = g_tr_old[~bad]

    ld_o = ioc_loader(old_ioc, g_tr_old, cfg, shuffle=True)
    ld_n = ioc_loader(new_ioc, g_tr_new, cfg, shuffle=True)
    ld_v = ioc_loader(new_ioc, g_va_new, cfg, shuffle=False)

    def _run(loader):
        for b in loader:
            b = b.to(device); bs = b['EVENT'].batch_size
            lbl = b['EVENT'].y[:bs]
            if (lbl < 0).any() or (lbl >= nc).any(): continue
            xi, ei, ea, nn_ = ioc_inputs(b)
            loss = crit(m_ioc.classifier(m_ioc(xi, ei, ea, nn_)[:bs]), lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m_ioc.parameters(), 1.0)
            opt.step()

    best, sd, pat = 0.0, None, 0
    for ep in range(cfg.ioc_max_epochs):
        m_ioc.train(); _run(ld_o); _run(ld_n)
        bacc = _val_ioc(m_ioc, ld_v, device); sch.step(bacc)
        if bacc > best: best = bacc; sd = deepcopy(m_ioc.state_dict()); pat = 0
        else:
            pat += 1
            if pat >= cfg.ioc_patience:
                print(f'    [Full-IOC] Early stop ep{ep+1} B-Acc={best:.4f}'); break
        if (ep + 1) % 30 == 0:
            print(f'      [Full-IOC] Ep{ep+1:03d} B-Acc={bacc:.4f}')
    m_ioc.load_state_dict(sd)

    # TTP 合并序列
    os_ = ttp_seqs(old_ttp); oph = old_ttp.get('phase_sequences')
    ogf = old_ttp.get('global_features')
    ns_ = ttp_seqs(new_ttp); nph = new_ttp.get('phase_sequences')
    ngf = new_ttp.get('global_features')

    off   = len(os_)
    c_seq = os_ + ns_
    c_ph  = (oph + nph if oph is not None and nph is not None else None)
    og_t  = (ogf if isinstance(ogf, torch.Tensor) else
              torch.tensor(ogf)) if ogf is not None else None
    ng_t  = (ngf if isinstance(ngf, torch.Tensor) else
              torch.tensor(ngf)) if ngf is not None else None
    c_g   = (torch.cat([og_t, ng_t], 0)
              if og_t is not None and ng_t is not None else None)

    ti_n  = list(ti_new.tolist() if hasattr(ti_new, 'tolist') else ti_new)
    vi_n  = list(vi_new.tolist() if hasattr(vi_new, 'tolist') else vi_new)
    tr_c  = list(ti_old) + [i + off for i in ti_n]
    va_c  = [i + off for i in vi_n]
    lbl_c = np.concatenate([lbl_old, lbl_new])

    sk = ('causal_sequences' if 'causal_sequences' in new_ttp
          else 'technique_sequences')
    c_ttp = {sk: c_seq, 'phase_sequences': c_ph, 'global_features': c_g,
             'num_techniques': max(old_ttp.get('num_techniques', 370),
                                   new_ttp.get('num_techniques', 370)),
             'global_feature_dim': new_ttp.get('global_feature_dim', 5),
             'technique_embeddings': new_ttp.get('technique_embeddings')}
    m_ttp = build_ttp(c_ttp, nc, cfg).to(device)
    m_ttp = train_ttp(m_ttp, c_ttp, tr_c, va_c, lbl_c, crit,
                       cfg.ttp_max_epochs, cfg.ttp_lr, cfg.ttp_patience,
                       cfg, device, tag='Full-TTP')
    return m_ioc, m_ttp


# ══════════════════════════════════════════════════
# 评估函数
# ══════════════════════════════════════════════════

def eval_ioc(model, ioc_data, g_te, cfg, device):
    model.eval(); logits_all, lbl_all = [], []
    for b in ioc_loader(ioc_data, g_te, cfg, shuffle=False):
        b  = b.to(device); bs = b['EVENT'].batch_size
        xi, ei, ea, nn_ = ioc_inputs(b)
        with torch.no_grad():
            logits_all.append(model.classifier(model(xi, ei, ea, nn_)[:bs]).cpu())
        lbl_all.append(b['EVENT'].y[:bs].cpu())
    lg  = torch.cat(logits_all); lb = torch.cat(lbl_all).numpy()
    pr  = lg.argmax(1).numpy()
    return accuracy_score(lb, pr), balanced_accuracy_score(lb, pr), pr, lb


def eval_ttp(model, ttp_data, te_pos, labels, cfg, device):
    model.eval()
    seqs = ttp_seqs(ttp_data); phases = ttp_data.get('phase_sequences')
    gf   = ttp_data.get('global_features')
    te_pos = list(te_pos) if hasattr(te_pos, 'tolist') else list(te_pos)
    tp, tm = safe_pad([seqs[i] for i in te_pos], device)
    tph, _ = safe_pad([phases[i] for i in te_pos] if phases else None, device)
    tg     = gf[torch.tensor(te_pos)].to(device) if gf is not None else None
    te_y   = labels[te_pos]
    with torch.no_grad():
        pr = model(tp, tph, tg, tm).argmax(1).cpu().numpy()
    return accuracy_score(te_y, pr), balanced_accuracy_score(te_y, pr), pr, te_y


def get_metrics(im, tm, ioc_d, ttp_d, g_te, pos_te, labels, cfg, device):
    if len(g_te) == 0:
        return {k: 0.0 for k in ['ioc_acc','ioc_bacc','ttp_acc','ttp_bacc',
                                   'or_acc','or_bacc']}
    ia, ib, ip, il = eval_ioc(im, ioc_d, g_te, cfg, device)
    ta, tb, tp_, _  = eval_ttp(tm, ttp_d, pos_te, labels, cfg, device)
    n  = min(len(ip), len(tp_))
    op = np.where(ip[:n] == il[:n], ip[:n], tp_[:n])
    return {
        'ioc_acc':  float(ia), 'ioc_bacc': float(ib),
        'ttp_acc':  float(ta), 'ttp_bacc': float(tb),
        'or_acc':   float(accuracy_score(il[:n], op)),
        'or_bacc':  float(balanced_accuracy_score(il[:n], op)),
    }


# ══════════════════════════════════════════════════
# 持续学习指标
# ══════════════════════════════════════════════════

def cl_metrics(summary, base_k, inc_k, full_k):
    """
    FM  = Base_Old_peak - Inc_Old    （遗忘量，越小越好）
    BT  = -FM                         （后向迁移，越大越好）
    FT  = Base_New - Full_New         （前向迁移，增量前零样本能力）
    Int = Full_New - Inc_New          （顽固性，越小越好）
    PSR = Inc_New / Inc_Old           （可塑稳定比，越接近1越好）
    """
    def m(k, sp): return summary[k][sp]['or_acc_mean']

    fm   = m(base_k, 'old') - m(inc_k, 'old')
    bt   = -fm
    ft   = m(base_k, 'new') - m(full_k, 'new')
    intr = m(full_k, 'new') - m(inc_k,  'new')
    psr  = (m(inc_k, 'new') / m(inc_k, 'old')
            if m(inc_k, 'old') > 0 else 0.0)
    return {'FM': round(fm,4), 'BT': round(bt,4), 'FT': round(ft,4),
            'Intransigence': round(intr,4), 'PSR': round(psr,4)}


# ══════════════════════════════════════════════════
# 数据预处理
# ══════════════════════════════════════════════════

def filter_to_classes(data, target_classes):
    cur = data._apt_classes if hasattr(data, '_apt_classes') else []
    c2l = {name: idx for idx, name in enumerate(cur)}
    found, tl = [], []
    for cls in target_classes:
        if cls in c2l:
            found.append(cls); tl.append(c2l[cls])
    if not found: return data
    tset = set(tl); o2n = {old: new for new, old in enumerate(tl)}
    ny = torch.full((len(data['EVENT'].y),), -1, dtype=torch.long)
    kept = 0
    for i, lbl in enumerate(data['EVENT'].y):
        lv = lbl.item()
        if lv != -1 and lv in tset:
            ny[i] = o2n[lv]; kept += 1
    data['EVENT'].y   = ny
    data._apt_classes = np.array([c for c in target_classes if c in found])
    print(f'    保留: {kept} 样本, {len(data._apt_classes)} 类')
    return data


def filter_top_k(data, k=15):
    valid  = data['EVENT'].y != -1
    labels = data['EVENT'].y[valid].numpy()
    u, cnt = np.unique(labels, return_counts=True)
    top_k  = u[np.argsort(-cnt)][:k]
    cur    = data._apt_classes if hasattr(data, '_apt_classes') else []
    tgt    = ([cur[l] for l in top_k if l < len(cur)] if len(cur) > 0
              else [f'Class_{l}' for l in top_k])
    return filter_to_classes(data, tgt)


def filter_ttp(ttp_data, ioc_data):
    vi  = torch.where(ioc_data['EVENT'].y != -1)[0].numpy()
    sk  = ('causal_sequences' if 'causal_sequences' in ttp_data
           else 'technique_sequences')
    nc  = ioc_data._apt_classes if hasattr(ioc_data, '_apt_classes') else []
    gf  = ttp_data.get('global_features')
    d   = {
        sk:                    [ttp_data[sk][i] for i in vi],
        'labels':              ioc_data['EVENT'].y[vi].clone(),
        'num_techniques':      ttp_data.get('num_techniques', 369),
        'num_events':          len(vi),
        'num_classes':         len(nc),
        'apt_classes':         nc,
        'num_phases':          ttp_data.get('num_phases'),
        'global_feature_dim':  ttp_data.get('global_feature_dim'),
        'technique_embeddings':ttp_data.get('technique_embeddings'),
        'valid_idx':           vi,
        'phase_sequences':     ([ttp_data['phase_sequences'][i] for i in vi]
                                 if ttp_data.get('phase_sequences') else None),
        'global_features':     (gf[vi] if gf is not None else None),
    }
    print(f'    TTP 过滤: {len(vi)} 条')
    return d


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════

def main():
    print('\n' + '='*65)
    print('  APT 增量学习 —— Experience Replay'.center(65))
    print('='*65)

    # ── 加载数据 ────────────────────────────────────────────────
    paths = {k: OUTPUT_DIR / v for k, v in {
        'old_ioc': 'apt_kg_ioc_old.pt',
        'old_ttp': 'apt_kg_ttp_old.pt',
        'new_ioc': 'apt_kg_ioc_updated.pt',
        'new_ttp': 'apt_kg_ttp_updated.pt',
    }.items()}
    for name, p in paths.items():
        if not p.exists():
            print(f'  [ERROR] 缺少文件: {p}'); return

    print('\n  [1/5] 加载数据...')
    old_ioc = torch.load(paths['old_ioc'], weights_only=False)
    new_ioc = torch.load(paths['new_ioc'], weights_only=False)
    old_ttp = torch.load(paths['old_ttp'], weights_only=False)
    new_ttp = torch.load(paths['new_ttp'], weights_only=False)
    print(f'    旧IOC: {old_ioc["EVENT"].num_nodes:,} 节点  '
          f'新IOC: {new_ioc["EVENT"].num_nodes:,} 节点')

    # ── 类别对齐 ─────────────────────────────────────────────────
    print('\n  [2/5] 类别过滤与对齐...')
    TOP_K   = 15
    old_ioc = filter_top_k(old_ioc, TOP_K)
    sel_cls = (old_ioc._apt_classes.tolist()
               if hasattr(old_ioc, '_apt_classes') else [])
    new_ioc = filter_to_classes(new_ioc, sel_cls)

    old_vi = torch.where(old_ioc['EVENT'].y != -1)[0]
    new_vi = torch.where(new_ioc['EVENT'].y != -1)[0]
    inc_idx = (new_vi[new_vi > old_vi.max().item()]
               if len(old_vi) > 0 and len(new_vi) > 0 else new_vi)
    print(f'    选定类别: {len(sel_cls)} 个')
    print(f'    旧数据有效样本: {len(old_vi):,}  新增样本: {len(inc_idx):,}')

    old_ttp_f = filter_ttp(old_ttp, old_ioc)
    new_ttp_f = filter_ttp(new_ttp, new_ioc)

    sk    = ('causal_sequences' if 'causal_sequences' in old_ttp_f
             else 'technique_sequences')
    old_n = int((old_ioc['EVENT'].y != -1).sum())
    new_n = int((new_ioc['EVENT'].y != -1).sum())
    if old_n != len(old_ttp_f[sk]) or new_n != len(new_ttp_f[sk]):
        print('  [ERROR] IOC 与 TTP 样本数不对齐'); return

    # ── 配置 ─────────────────────────────────────────────────────
    nc     = len(sel_cls)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = Config(device=str(device))
    cfg.show()

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    old_mask  = old_ioc['EVENT'].y != -1
    old_vidx  = torch.where(old_mask)[0].cpu().numpy()
    old_lbl   = old_ioc['EVENT'].y[old_mask].cpu().numpy()
    new_mask  = new_ioc['EVENT'].y != -1
    new_vidx  = torch.where(new_mask)[0].cpu().numpy()
    new_lbl   = new_ioc['EVENT'].y[new_mask].cpu().numpy()

    old_crit, old_cw = make_crit(old_lbl, device)
    new_crit, new_cw = make_crit(new_lbl, device)

    skf      = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                                random_state=cfg.seed)
    new_splt = list(StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                                     random_state=cfg.seed)
                    .split(range(len(new_lbl)), new_lbl))

    # ── 交叉验证 ─────────────────────────────────────────────────
    print('\n  [3/5] 5 折交叉验证...')
    fold_results = []
    timing       = {k: [] for k in ['1.Base','2.Replay','3.Full','4.Joint']}
    out_dir = OUTPUT_DIR / 'replay_final'; out_dir.mkdir(exist_ok=True)

    for fold, (tmp_tr, te_idx) in enumerate(
            skf.split(range(len(old_lbl)), old_lbl)):

        print(f'\n  {"="*60}')
        print(f'  Fold {fold+1}/{cfg.n_folds}')
        print(f'  {"="*60}')

        tr_idx, va_idx = train_test_split(
            tmp_tr, test_size=cfg.val_size,
            stratify=old_lbl[tmp_tr], random_state=42)
        g_tr = old_vidx[tr_idx]
        g_va = old_vidx[va_idx]
        g_te = old_vidx[te_idx]

        nt, nti = new_splt[fold]
        ntr, nva = train_test_split(nt, test_size=cfg.val_size,
                                     stratify=new_lbl[nt], random_state=42)
        ng_tr = new_vidx[ntr]; ng_va = new_vidx[nva]; ng_te = new_vidx[nti]

        fold_models = {}

        # 1. Base
        print(f'\n  ▶ [1/4] Base（旧数据从头训练）')
        t0 = time.time()
        b_ioc = train_ioc(build_ioc(new_ioc, nc, cfg).to(device),
                           old_ioc, g_tr, g_va, old_cw, old_crit,
                           cfg.ioc_max_epochs, cfg.ioc_lr, cfg.ioc_patience,
                           cfg, device, tag='Base-IOC')
        b_ttp = train_ttp(build_ttp(old_ttp_f, nc, cfg).to(device),
                           old_ttp_f, tr_idx, va_idx, old_lbl, old_crit,
                           cfg.ttp_max_epochs, cfg.ttp_lr, cfg.ttp_patience,
                           cfg, device, tag='Base-TTP')
        timing['1.Base'].append(time.time() - t0)
        fold_models['1. Base (Old Only)'] = (b_ioc, b_ttp)

        # 2. Replay Incremental
        print(f'\n  ▶ [2/4] Incremental (Replay)')
        t0 = time.time()
        r_ioc = replay_ioc(b_ioc, old_ioc, g_tr, new_ioc, ng_tr, ng_va,
                            new_cw, new_crit, cfg, device, tag='Replay-IOC')
        r_ttp = replay_ttp(b_ttp, old_ttp_f, tr_idx, old_lbl,
                            new_ttp_f, ntr, nva, new_lbl,
                            new_crit, cfg, device, tag='Replay-TTP')
        timing['2.Replay'].append(time.time() - t0)
        fold_models['2. Incremental (Replay)'] = (r_ioc, r_ttp)

        # 3. Full Finetune
        print(f'\n  ▶ [3/4] Full Finetune（旧+新合并，上界）')
        t0 = time.time()
        f_ioc, f_ttp = full_finetune(
            old_ioc, old_ttp_f, new_ioc, new_ttp_f,
            g_tr, g_va, tr_idx, va_idx,
            ng_tr, ng_va, ntr, nva,
            old_lbl, new_lbl, old_cw, new_cw, nc, cfg, device)
        timing['3.Full'].append(time.time() - t0)
        fold_models['3. Full Finetune'] = (f_ioc, f_ttp)

        # 4. Joint
        print(f'\n  ▶ [4/4] Joint（仅新数据，迁移参考）')
        t0 = time.time()
        j_ioc = train_ioc(build_ioc(new_ioc, nc, cfg).to(device),
                           new_ioc, ng_tr, ng_va, new_cw, new_crit,
                           cfg.ioc_max_epochs, cfg.ioc_lr, cfg.ioc_patience,
                           cfg, device, tag='Joint-IOC')
        j_ttp = train_ttp(build_ttp(new_ttp_f, nc, cfg).to(device),
                           new_ttp_f, ntr, nva, new_lbl, new_crit,
                           cfg.ttp_max_epochs, cfg.ttp_lr, cfg.ttp_patience,
                           cfg, device, tag='Joint-TTP')
        timing['4.Joint'].append(time.time() - t0)
        fold_models['4. Joint (New Only)'] = (j_ioc, j_ttp)

        # ── 纯新样本测试集 ──────────────────────────────────────
        nti_arr = nti if isinstance(nti, np.ndarray) else np.array(nti)
        if inc_idx is not None and len(inc_idx) > 0:
            _inc_list = (inc_idx.tolist() if hasattr(inc_idx, 'tolist')
                         else list(inc_idx))
            inc_set   = set(_inc_list)
            # is_new 按 ng_te（IOC全局节点ID）判断是否属于新增样本
            is_new    = np.array([int(g) in inc_set for g in ng_te])
            g_te_pure = ng_te[is_new]
            lt_pure   = nti_arr[is_new]   # TTP本地索引，与ng_te一一对应
        else:
            g_te_pure = ng_te
            lt_pure   = nti_arr

        # ── Fold 评估 ───────────────────────────────────────────
        fold_res = {}
        W = 96
        print(f'\n  [评估] Fold {fold+1}')
        print(f'  {"-"*W}')
        print(f'  {"Method":<26} | {"Old_OR":>8} | {"New_OR":>8} | '
              f'{"All_OR":>8} | {"IOC_Acc":>8} | {"TTP_Acc":>8} | '
              f'{"OR_BAcc":>8} | {"Time(s)":>7}')
        print(f'  {"-"*W}')

        for name, (im, tm) in fold_models.items():
            r_old = get_metrics(im, tm, old_ioc, old_ttp_f,
                                g_te, te_idx, old_lbl, cfg, device)
            r_new = get_metrics(im, tm, new_ioc, new_ttp_f,
                                g_te_pure, lt_pure, new_lbl, cfg, device)
            r_all = get_metrics(im, tm, new_ioc, new_ttp_f,
                                ng_te, nti_arr, new_lbl, cfg, device)
            tk    = {'1. Base (Old Only)':   '1.Base',
                     '2. Incremental (Replay)':'2.Replay',
                     '3. Full Finetune':     '3.Full',
                     '4. Joint (New Only)':  '4.Joint'}.get(name, '')
            t_s   = timing[tk][-1] if tk else 0.0
            fold_res[name] = {'old': r_old, 'new': r_new,
                               'all': r_all, 'time': t_s}
            print(f'  {name:<26} | '
                  f'{r_old["or_acc"]:>8.4f} | '
                  f'{r_new["or_acc"]:>8.4f} | '
                  f'{r_all["or_acc"]:>8.4f} | '
                  f'{r_all["ioc_acc"]:>8.4f} | '
                  f'{r_all["ttp_acc"]:>8.4f} | '
                  f'{r_all["or_bacc"]:>8.4f} | '
                  f'{t_s:>7.0f}')
        print(f'  {"-"*W}')
        fold_results.append(fold_res)

    # ── 汇总统计 ─────────────────────────────────────────────────
    print('\n  [4/5] 汇总统计...')
    model_names = list(fold_results[0].keys())
    splits      = ['old', 'new', 'all']
    metrics_k   = ['ioc_acc', 'ioc_bacc', 'ttp_acc', 'ttp_bacc',
                    'or_acc', 'or_bacc']
    summary = {}
    for mn in model_names:
        summary[mn] = {}
        for sp in splits:
            summary[mn][sp] = {}
            for mt in metrics_k:
                vals = [f[mn][sp].get(mt, 0.) for f in fold_results]
                summary[mn][sp][f'{mt}_mean'] = float(np.mean(vals))
                summary[mn][sp][f'{mt}_std']  = float(np.std(vals))
        tk = {'1. Base (Old Only)':   '1.Base',
              '2. Incremental (Replay)':'2.Replay',
              '3. Full Finetune':     '3.Full',
              '4. Joint (New Only)':  '4.Joint'}.get(mn, '')
        ts = timing.get(tk, [0.])
        summary[mn]['time_mean'] = float(np.mean(ts))
        summary[mn]['time_std']  = float(np.std(ts))

    # ── 最终报告 ─────────────────────────────────────────────────
    print('\n  [5/5] 输出报告...')
    W = 108
    print(f'\n  {"="*W}')
    print('  🏆 最终报告：OR Acc（5折均值 ± 标准差）'.center(W))
    print(f'  {"="*W}')
    print(f'  {"Method":<26} | {"Test_Old OR":^18} | {"Test_New OR":^18} | '
          f'{"Test_All OR":^18} | {"B-Acc(All)":^12} | {"Time(s)":^10}')
    print(f'  {"-"*W}')
    for mn, res in summary.items():
        def f(sp, mt='or_acc'):
            return f'{res[sp][mt+"_mean"]:.4f}±{res[sp][mt+"_std"]:.4f}'
        t_str = f'{res["time_mean"]:.0f}±{res["time_std"]:.0f}'
        print(f'  {mn:<26} | {f("old"):^18} | {f("new"):^18} | '
              f'{f("all"):^18} | {f("all","or_bacc"):^12} | {t_str:^10}')
    print(f'  {"="*W}')

    # ── 持续学习指标 ─────────────────────────────────────────────
    base_k = '1. Base (Old Only)'
    inc_k  = '2. Incremental (Replay)'
    full_k = '3. Full Finetune'

    cl = cl_metrics(summary, base_k, inc_k, full_k)
    print(f'\n  📐 持续学习标准指标（Replay vs Full Finetune）')
    print(f'  {"─"*60}')
    desc = {
        'FM':            ('遗忘度量',          '越小越好，负数=无遗忘/反向迁移'),
        'BT':            ('后向迁移',          '越大越好，正=提升旧任务'),
        'FT':            ('前向迁移',          '旧模型对新任务零样本能力'),
        'Intransigence': ('顽固性',            '因保旧而损失的新任务性能，越小越好'),
        'PSR':           ('可塑-稳定比',       '新/旧性能比，越接近1越好'),
    }
    for k, v in cl.items():
        zh, tip = desc[k]
        flag    = ''
        if k == 'FM':  flag = '✅' if v <= 0 else ('⚠️' if v < 0.05 else '❌')
        if k == 'BT':  flag = '✅' if v >= 0 else '❌'
        if k == 'Int': flag = '✅' if v < 0.05 else '⚠️'
        if k == 'PSR': flag = '✅' if 0.85 <= v <= 1.15 else '⚠️'
        print(f'    {k:<16} {zh:<10} = {v:>8.4f}  {flag}  ({tip})')

    # ── 两点目标验证 ─────────────────────────────────────────────
    fv = summary[full_k]['all']['or_acc_mean']
    rv = summary[inc_k]['all']['or_acc_mean']
    ft = summary[full_k]['time_mean']
    rt = summary[inc_k]['time_mean']
    gap = rv - fv
    tsave = (ft - rt) / ft if ft > 0 else 0.

    print(f'\n  ✅ 两点设计目标验证')
    print(f'  {"─"*60}')
    g1 = '✅ 达标' if abs(gap) < 0.01 else f'❌ 差距 {abs(gap):.2%}'
    g2 = '✅ 达标' if tsave > 0      else '❌ 未达标'
    print(f'  目标1 准确率差距 <1%：')
    print(f'    Full={fv:.4f}  Replay={rv:.4f}  gap={gap:+.4f}  → {g1}')
    print(f'  目标2 耗时更短：')
    print(f'    Full={ft:.0f}s  Replay={rt:.0f}s  节省={tsave:.1%}  → {g2}')

    # ── 参数量 ───────────────────────────────────────────────────
    print(f'\n  📊 模型参数量')
    sample_ioc = build_ioc(new_ioc, nc, cfg)
    sample_ttp = build_ttp(new_ttp_f, nc, cfg)
    n_ioc = sum(p.numel() for p in sample_ioc.parameters())
    n_ttp = sum(p.numel() for p in sample_ttp.parameters())
    print(f'    IOC (RGCN):       {n_ioc:>10,} 参数')
    print(f'    TTP (Transformer):{n_ttp:>10,} 参数')
    print(f'    合计:             {n_ioc+n_ttp:>10,} 参数')

    # ── 保存结果 ─────────────────────────────────────────────────
    results = {
        'config': {
            'replay_ratio':    cfg.replay_ratio,
            'old_loss_weight': cfg.old_loss_weight,
            'finetune_epochs': cfg.finetune_epochs,
            'n_folds':         cfg.n_folds,
            'num_classes':     nc,
        },
        'summary':      summary,
        'fold_results': fold_results,
        'cl_metrics':   cl,
        'goal1_gap':    float(gap),
        'goal2_tsave':  float(tsave),
        'timing':       {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                         for k, v in timing.items()},
    }
    sp = out_dir / 'replay_final_results.json'
    with open(sp, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f'\n  [保存] {sp}')
    print(f'  [完成]\n')


if __name__ == '__main__':
    main()