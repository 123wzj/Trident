"""
双模型OR融合 - 消融实验
共6组对比实验：
  Full  : OR融合完整模型（基准）
  C0    : 仅IOC模型
  C1    : 仅TTP模型
  C3    : 软投票融合（替代OR策略）
  B1    : TTP去除预训练嵌入（随机初始化）
  B2    : TTP去除阶段嵌入
  A1    : IOC去除边类型嵌入

用法:
  python ablation_study.py \
      --ioc-data ./apt_kg_ioc.pt \
      --ttp-data ./apt_kg_ttp.pt \
      --top-k-classes 15 \
      --experiments Full C0 C1 C3 B1 B2 A1
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List
import argparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append(os.path.dirname(__file__))
from train_rgcn_embedding import (
    EmbeddingGraphConfig,
    EmbeddingGraphDataProcessor,
    FocalLoss,
    filter_top_k_classes
)
# 从原训练脚本复用模型和工具
from train.train_dual_or_fusion import (
    IOCModelConfig,
    TTPModelConfig,
    IOCClassifier,
    TTPTransformer,
    _build_edge_attr_dict,
    _build_num_nodes_dict,
)


# ==================== 消融变体：IOC去除边类型嵌入 (A1) ====================

class IOCClassifierNoEdgeEmbed(nn.Module):
    """A1：去除边类型嵌入 + edge_mlp，退化为标准RGCN"""

    def __init__(self, metadata, input_dims, num_classes: int, cfg: IOCModelConfig):
        super().__init__()
        hidden_dim       = cfg.hidden_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = cfg.num_layers
        self.num_classes = num_classes
        self.node_types  = metadata[0]
        self.edge_types  = metadata[1]
        self.dropout     = cfg.dropout

        self.input_projs = nn.ModuleDict()
        for node_type, dim in input_dims.items():
            if dim is not None and dim > 0:
                self.input_projs[node_type] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                )

        # [A1] 无边类型嵌入，直接使用 RGCN
        self.edge_type_map = {etype: i for i, etype in enumerate(self.edge_types)}

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(RGCNConv(
                in_channels=hidden_dim, out_channels=hidden_dim,
                num_relations=len(metadata[1]), num_bases=cfg.num_bases
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict):
        h_dict = {}
        ref_device = next(
            (x_dict[k].device for k in x_dict if x_dict[k] is not None), torch.device('cpu')
        )
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type] is not None:
                h_dict[node_type] = (
                    self.input_projs[node_type](x_dict[node_type])
                    if node_type in self.input_projs
                    else torch.zeros(num_nodes_dict.get(node_type, 1), self.hidden_dim,
                                     device=ref_device)
                )
            else:
                h_dict[node_type] = torch.zeros(
                    max(num_nodes_dict.get(node_type, 1), 1),
                    self.hidden_dim, device=ref_device
                )

        x_all, node_offsets, curr = [], {}, 0
        for nt in self.node_types:
            x_all.append(h_dict[nt])
            node_offsets[nt] = curr
            curr += h_dict[nt].shape[0]
        x_all = torch.cat(x_all, dim=0)

        edge_indices, edge_types_list = [], []
        for edge_key, edge_index in edge_index_dict.items():
            if edge_index is None or edge_index.numel() == 0:
                continue
            src_t, _, dst_t = edge_key
            new_idx = edge_index.clone()
            new_idx[0] += node_offsets[src_t]
            new_idx[1] += node_offsets[dst_t]
            edge_indices.append(new_idx)
            edge_types_list.append(torch.full(
                (edge_index.shape[1],),
                self.edge_type_map[edge_key],
                dtype=torch.long, device=ref_device
            ))

        if not edge_indices:
            return h_dict['EVENT']

        edge_index_all = torch.cat(edge_indices, dim=1)
        edge_type_all  = torch.cat(edge_types_list, dim=0)

        h = x_all
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index_all, edge_type_all)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h  # 保留残差

        event_start = node_offsets['EVENT']
        event_end   = event_start + h_dict['EVENT'].shape[0]
        return h[event_start:event_end]


# ==================== 消融变体：TTP去除预训练嵌入 (B1) ====================

def build_ttp_model_b1(num_techniques, num_classes, cfg, ttp_data):
    """B1：强制使用随机初始化嵌入，忽略 technique_embeddings"""
    num_phases          = ttp_data.get('num_phases', 14)
    num_subseq_features = ttp_data.get('num_feature_subseq', 0)
    global_feature_dim  = ttp_data.get('global_feature_dim', 5)
    return TTPTransformer(
        num_techniques=num_techniques,
        num_classes=num_classes,
        cfg=cfg,
        pretrained_embeddings=None,       # [B1] 不传入预训练嵌入
        num_phases=num_phases,
        num_subseq_features=num_subseq_features,
        global_feature_dim=global_feature_dim
    )


# ==================== 消融变体：TTP去除阶段嵌入 (B2) ====================

class TTPTransformerNoPhase(nn.Module):
    """B2：去除 phase_embedding 和 fusion_proj，只用技术语义序列"""

    def __init__(self, num_techniques: int, num_classes: int,
                 cfg: TTPModelConfig,
                 pretrained_embeddings=None,
                 num_subseq_features: int = 0,
                 global_feature_dim: int = 6):
        super().__init__()
        d_model          = cfg.d_model
        self.num_classes = num_classes

        if pretrained_embeddings is not None:
            self.tech_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.float(), padding_idx=0, freeze=False
            )
            semantic_dim = pretrained_embeddings.shape[1]
            self.semantic_proj = (
                nn.Sequential(nn.Linear(semantic_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
                if semantic_dim != d_model else nn.Identity()
            )
        else:
            self.tech_embedding = nn.Embedding(num_techniques, d_model, padding_idx=0)
            self.semantic_proj  = nn.Identity()

        # [B2] 无阶段嵌入，无 fusion_proj
        from train.train_dual_or_fusion import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cfg.nhead,
            dim_feedforward=d_model * 4,
            dropout=cfg.dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.use_subseq = num_subseq_features > 0
        self.subseq_mlp = (
            nn.Sequential(
                nn.Linear(num_subseq_features, d_model // 2),
                nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(cfg.dropout)
            ) if self.use_subseq else None
        )
        self.use_global = global_feature_dim > 0
        self.global_mlp = (
            nn.Sequential(
                nn.Linear(global_feature_dim, d_model // 4),
                nn.LayerNorm(d_model // 4), nn.GELU(), nn.Dropout(cfg.dropout)
            ) if self.use_global else None
        )

        cls_in = d_model
        if self.use_subseq: cls_in += d_model // 2
        if self.use_global: cls_in += d_model // 4

        self.classifier = nn.Sequential(
            nn.Linear(cls_in, cls_in // 2),
            nn.LayerNorm(cls_in // 2), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cls_in // 2, num_classes)
        )
        print(f"    [B2-TTP] 无阶段嵌入，分类器输入: {cls_in}维")

    def forward(self, technique_sequences, phase_sequences=None,
                subseq_features=None, global_features=None, attention_mask=None):
        # [B2] 忽略 phase_sequences
        x = self.semantic_proj(self.tech_embedding(technique_sequences))
        x = self.pos_encoding(x)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            pooled = encoded.mean(1)

        features = [pooled]
        if self.use_subseq and subseq_features is not None:
            features.append(self.subseq_mlp(subseq_features))
        if self.use_global and global_features is not None:
            features.append(self.global_mlp(global_features))

        return self.classifier(torch.cat(features, dim=-1))


# ==================== 消融实验主类 ====================

class AblationStudy:

    EXPERIMENT_NAMES = {
        'Full': 'OR融合完整模型',
        'C0':   '仅IOC模型',
        'C1':   '仅TTP模型',
        'C3':   '软投票融合',
        'B1':   'TTP去除预训练嵌入',
        'B2':   'TTP去除阶段嵌入',
        'A1':   'IOC去除边类型嵌入',
    }

    def __init__(self, config: EmbeddingGraphConfig,
                 ioc_cfg: IOCModelConfig,
                 ttp_cfg: TTPModelConfig,
                 experiments: List[str]):
        self.config      = config
        self.ioc_cfg     = ioc_cfg
        self.ttp_cfg     = ttp_cfg
        self.experiments = experiments

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path("../v2/results") / f"ablation_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(self.results_dir / "config.json", "w") as f:
            json.dump({
                "experiments": experiments,
                "base": config.to_dict(),
                "ioc":  ioc_cfg.to_dict(),
                "ttp":  ttp_cfg.to_dict(),
            }, f, indent=2)

        print(f"\n[消融实验] 结果目录: {self.results_dir}")
        print(f"[消融实验] 实验组: {experiments}")

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def run(self, ioc_data, ttp_data, class_weights, valid_idx):
        valid_y      = ioc_data['EVENT'].y[valid_idx].numpy()
        num_classes  = len(np.unique(valid_y))
        ioc_input_dims = EmbeddingGraphDataProcessor.build_input_dims(ioc_data)

        ttp_sequences = ttp_data.get('causal_sequences',
                                     ttp_data.get('technique_sequences'))
        all_ids       = [id_ for seq in ttp_sequences for id_ in seq]
        num_techniques = max(
            (max(all_ids) + 1) if all_ids else 369,
            ttp_data.get('num_techniques', 369)
        )

        skf = StratifiedKFold(
            n_splits=self.config.n_folds, shuffle=True, random_state=42
        )
        # results[exp_name] = list of per-fold dicts
        all_results = {exp: [] for exp in self.experiments}

        for fold, (temp_local_train_idx, local_test_idx) in enumerate(
                skf.split(valid_idx, valid_y)):

            print(f"\n{'='*70}")
            print(f"Fold {fold+1}/{self.config.n_folds}".center(70))
            print(f"{'='*70}")

            local_train_idx, local_val_idx = train_test_split(
                temp_local_train_idx,
                test_size=self.config.val_ratio,
                stratify=valid_y[temp_local_train_idx],
                random_state=42
            )
            global_train_idx = valid_idx[local_train_idx]
            global_val_idx   = valid_idx[local_val_idx]
            global_test_idx  = valid_idx[local_test_idx]

            # ---- 按需训练各模型 ----
            need_ioc     = any(e in self.experiments for e in ['Full', 'C0', 'C3', 'A1'])
            need_ttp     = any(e in self.experiments for e in ['Full', 'C1', 'C3', 'B1', 'B2'])
            need_ioc_a1  = 'A1' in self.experiments
            need_ttp_b1  = 'B1' in self.experiments
            need_ttp_b2  = 'B2' in self.experiments

            ioc_model = ttp_model = ioc_a1_model = ttp_b1_model = ttp_b2_model = None

            # 标准IOC模型（Full / C0 / C3）
            if need_ioc and not (len(self.experiments) == 1 and 'A1' in self.experiments):
                print(f"\n  训练标准IOC模型")
                ioc_model = IOCClassifier(
                    metadata=ioc_data.metadata(),
                    input_dims=ioc_input_dims,
                    num_classes=num_classes,
                    cfg=self.ioc_cfg
                ).to(self.config.device)
                ioc_model = self._train_ioc(
                    ioc_model, ioc_data, global_train_idx, global_val_idx, class_weights
                )

            # A1：无边类型嵌入的IOC模型
            if need_ioc_a1:
                print(f"\n  训练A1-IOC模型（无边类型嵌入）")
                ioc_a1_model = IOCClassifierNoEdgeEmbed(
                    metadata=ioc_data.metadata(),
                    input_dims=ioc_input_dims,
                    num_classes=num_classes,
                    cfg=self.ioc_cfg
                ).to(self.config.device)
                ioc_a1_model = self._train_ioc(
                    ioc_a1_model, ioc_data, global_train_idx, global_val_idx, class_weights
                )

            # 标准TTP模型（Full / C1 / C3）
            if need_ttp:
                print(f"\n  训练标准TTP模型")
                ttp_model = self._build_standard_ttp(
                    num_techniques, num_classes, ttp_data
                ).to(self.config.device)
                ttp_model = self._train_ttp(
                    ttp_model, ttp_sequences,
                    ttp_data.get('phase_sequences'),
                    ttp_data.get('subseq_features'),
                    ttp_data.get('global_features'),
                    local_train_idx, local_val_idx, valid_y, class_weights
                )

            # B1：随机嵌入TTP
            if need_ttp_b1:
                print(f"\n  训练B1-TTP模型（随机初始化嵌入）")
                ttp_b1_model = build_ttp_model_b1(
                    num_techniques, num_classes, self.ttp_cfg, ttp_data
                ).to(self.config.device)
                ttp_b1_model = self._train_ttp(
                    ttp_b1_model, ttp_sequences,
                    ttp_data.get('phase_sequences'),
                    ttp_data.get('subseq_features'),
                    ttp_data.get('global_features'),
                    local_train_idx, local_val_idx, valid_y, class_weights
                )

            # B2：无阶段嵌入TTP
            if need_ttp_b2:
                print(f"\n  训练B2-TTP模型（无阶段嵌入）")
                ttp_b2_model = TTPTransformerNoPhase(
                    num_techniques=num_techniques,
                    num_classes=num_classes,
                    cfg=self.ttp_cfg,
                    pretrained_embeddings=ttp_data.get('technique_embeddings'),
                    num_subseq_features=ttp_data.get('num_feature_subseq', 0),
                    global_feature_dim=ttp_data.get('global_feature_dim', 5)
                ).to(self.config.device)
                ttp_b2_model = self._train_ttp(
                    ttp_b2_model, ttp_sequences,
                    None,                              # [B2] 不传phase_sequences
                    ttp_data.get('subseq_features'),
                    ttp_data.get('global_features'),
                    local_train_idx, local_val_idx, valid_y, class_weights
                )

            # ---- 评估各实验组 ----
            ioc_preds = ioc_labels = None
            if ioc_model is not None:
                _, ioc_preds, ioc_labels = self._eval_ioc(
                    ioc_model, ioc_data, global_test_idx
                )

            ttp_preds = None
            if ttp_model is not None:
                _, ttp_preds, _ = self._eval_ttp(
                    ttp_model, ttp_sequences,
                    ttp_data.get('phase_sequences'),
                    ttp_data.get('subseq_features'),
                    ttp_data.get('global_features'),
                    local_test_idx, valid_y
                )

            test_labels = valid_y[local_test_idx]

            for exp in self.experiments:
                metrics = self._compute_metrics_for_exp(
                    exp, ioc_preds, ttp_preds, ioc_labels, test_labels,
                    ioc_data, global_test_idx,
                    ttp_sequences, ttp_data, local_test_idx, valid_y,
                    ioc_a1_model, ttp_b1_model, ttp_b2_model
                )
                all_results[exp].append({'fold': fold, **metrics})
                print(f"    [{exp}] {self.EXPERIMENT_NAMES[exp]}: "
                      f"Acc={metrics['acc']:.4f}, B-Acc={metrics['bacc']:.4f}, "
                      f"F1={metrics['f1']:.4f}")

        self._save_and_print(all_results)

    # ------------------------------------------------------------------
    # 每组实验的评估分发
    # ------------------------------------------------------------------
    def _compute_metrics_for_exp(self, exp,
                                  ioc_preds, ttp_preds, ioc_labels, test_labels,
                                  ioc_data, global_test_idx,
                                  ttp_sequences, ttp_data, local_test_idx, valid_y,
                                  ioc_a1_model, ttp_b1_model, ttp_b2_model):
        if exp == 'Full':
            # OR融合：IOC正确则取IOC，否则取TTP
            preds = np.where(ioc_preds == ioc_labels, ioc_preds, ttp_preds)

        elif exp == 'C0':
            preds = ioc_preds

        elif exp == 'C1':
            preds = ttp_preds

        elif exp == 'C3':
            # 软投票：两模型 logit 均值后 argmax
            preds = self._soft_vote(
                ioc_data, global_test_idx,
                ttp_sequences, ttp_data, local_test_idx, valid_y
            )

        elif exp == 'A1':
            _, preds, _ = self._eval_ioc(ioc_a1_model, ioc_data, global_test_idx)

        elif exp == 'B1':
            _, preds, _ = self._eval_ttp(
                ttp_b1_model, ttp_sequences,
                ttp_data.get('phase_sequences'),
                ttp_data.get('subseq_features'),
                ttp_data.get('global_features'),
                local_test_idx, valid_y
            )

        elif exp == 'B2':
            _, preds, _ = self._eval_ttp(
                ttp_b2_model, ttp_sequences,
                None,                    # [B2] 不传phase
                ttp_data.get('subseq_features'),
                ttp_data.get('global_features'),
                local_test_idx, valid_y
            )
        else:
            raise ValueError(f"未知实验组: {exp}")

        labels = test_labels if exp not in ('C0', 'A1') else ioc_labels
        # C0/A1 走IOC loader，标签来自ioc_labels；其余走local_test_idx
        if exp in ('C0', 'A1'):
            labels = ioc_labels
        else:
            labels = test_labels

        return {
            'acc':  accuracy_score(labels, preds),
            'bacc': balanced_accuracy_score(labels, preds),
            'f1':   f1_score(labels, preds, average='macro', zero_division=0),
        }

    # ------------------------------------------------------------------
    # 软投票（C3）
    # ------------------------------------------------------------------
    def _soft_vote(self, ioc_data, global_test_idx,
                   ttp_sequences, ttp_data, local_test_idx, valid_y):
        """收集两模型 logit，均值后 argmax"""
        # 这里需要缓存上一轮训练好的模型 logit
        # 设计上在 run() 里已训练好 ioc_model / ttp_model，
        # 此处通过额外一次前向传播取 logit
        raise NotImplementedError(
            "软投票需在 run() 中直接访问模型对象，请使用 _soft_vote_with_models()"
        )

    def _soft_vote_with_models(self, ioc_model, ttp_model,
                                ioc_data, global_test_idx,
                                ttp_sequences, ttp_data, local_test_idx, valid_y):
        """C3：两模型 logit 均值后 argmax"""
        device = self.config.device

        # IOC logit
        ioc_model.eval()
        ioc_logits_list = []
        test_loader = NeighborLoader(
            ioc_data, num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            input_nodes=('EVENT', torch.tensor(global_test_idx, dtype=torch.long)),
            shuffle=False
        )
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                bs    = batch['EVENT'].batch_size
                x_dict          = {nt: batch[nt].x for nt in batch.node_types
                                   if nt in ['IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT']}
                edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
                edge_attr_dict  = _build_edge_attr_dict(batch)
                num_nodes_dict  = _build_num_nodes_dict(batch)
                emb = ioc_model(x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict)
                ioc_logits_list.append(ioc_model.classifier(emb[:bs]).cpu())
        ioc_logits = torch.cat(ioc_logits_list, dim=0)  # [N, C]

        # TTP logit
        ttp_model.eval()
        phase_sequences = ttp_data.get('phase_sequences')
        subseq_features = ttp_data.get('subseq_features')
        global_features = ttp_data.get('global_features')

        test_seqs   = [ttp_sequences[i] for i in local_test_idx]
        test_phases = [phase_sequences[i] for i in local_test_idx] if phase_sequences else None

        test_padded = pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in test_seqs],
            batch_first=True, padding_value=0
        ).to(device)
        test_mask = torch.zeros_like(test_padded)
        for i, s in enumerate(test_seqs):
            test_mask[i, :len(s)] = 1

        test_phase_padded = None
        if test_phases:
            test_phase_padded = pad_sequence(
                [torch.tensor(p, dtype=torch.long) for p in test_phases],
                batch_first=True, padding_value=0
            ).to(device)

        idx_t       = torch.tensor(local_test_idx)
        test_subseq = subseq_features[idx_t].to(device) if subseq_features is not None else None
        test_global = global_features[idx_t].to(device) if global_features is not None else None

        with torch.no_grad():
            ttp_logits = ttp_model(
                test_padded, test_phase_padded, test_subseq, test_global, test_mask
            ).cpu()   # [N, C]

        # 对齐维度（两模型输出类别数应相同）
        assert ioc_logits.shape == ttp_logits.shape, \
            f"logit维度不匹配: IOC {ioc_logits.shape} vs TTP {ttp_logits.shape}"

        soft_preds = (ioc_logits + ttp_logits).argmax(dim=1).numpy()
        return soft_preds

    # ------------------------------------------------------------------
    # IOC训练（复用原脚本逻辑）
    # ------------------------------------------------------------------
    def _train_ioc(self, model, ioc_data, train_idx, val_idx, class_weights):
        cfg = self.ioc_cfg
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=cfg.scheduler_patience
        )
        criterion = FocalLoss(alpha=class_weights, gamma=3, label_smoothing=0.1)

        train_loader = NeighborLoader(
            ioc_data, num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            input_nodes=('EVENT', torch.tensor(train_idx, dtype=torch.long)),
            shuffle=True
        )
        val_loader = NeighborLoader(
            ioc_data, num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            input_nodes=('EVENT', torch.tensor(val_idx, dtype=torch.long)),
            shuffle=False
        )

        best_bacc, best_state, patience_counter = 0, None, 0

        for epoch in range(cfg.max_epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(self.config.device)
                bs    = batch['EVENT'].batch_size
                x_dict          = {nt: batch[nt].x for nt in batch.node_types
                                   if nt in ['IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT']}
                edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
                edge_attr_dict  = _build_edge_attr_dict(batch)
                num_nodes_dict  = _build_num_nodes_dict(batch)

                optimizer.zero_grad()
                emb  = model(x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict)
                loss = criterion(model.classifier(emb[:bs]), batch['EVENT'].y[:bs])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds, val_labels = [], []
                for batch in val_loader:
                    batch = batch.to(self.config.device)
                    bs    = batch['EVENT'].batch_size
                    x_dict          = {nt: batch[nt].x for nt in batch.node_types
                                       if nt in ['IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT']}
                    edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
                    edge_attr_dict  = _build_edge_attr_dict(batch)
                    num_nodes_dict  = _build_num_nodes_dict(batch)
                    emb = model(x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict)
                    val_preds.append(model.classifier(emb[:bs]).argmax(1).cpu())
                    val_labels.append(batch['EVENT'].y[:bs].cpu())

            val_bacc = balanced_accuracy_score(
                torch.cat(val_labels).numpy(), torch.cat(val_preds).numpy()
            )
            scheduler.step(val_bacc)

            if val_bacc > best_bacc:
                best_bacc, best_state, patience_counter = val_bacc, \
                    {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

            if (epoch + 1) % cfg.log_interval == 0:
                print(f"        Epoch {epoch+1:03d} | Val B-Acc: {val_bacc:.4f}")

        model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    # TTP训练（复用原脚本逻辑）
    # ------------------------------------------------------------------
    def _train_ttp(self, model, ttp_sequences, phase_sequences,
                   subseq_features, global_features,
                   local_train_idx, local_val_idx, valid_y, class_weights):
        cfg    = self.ttp_cfg
        device = self.config.device

        def collate_fn(batch):
            indices, labels = zip(*batch)
            return indices, labels

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5,
            patience=cfg.scheduler_patience, min_lr=cfg.min_lr
        )
        criterion = FocalLoss(alpha=class_weights, gamma=3, label_smoothing=0.1)

        train_dataset = list(zip(
            local_train_idx.tolist(), valid_y[local_train_idx].tolist()
        ))
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
        )

        # 验证集准备
        val_seqs   = [ttp_sequences[i] for i in local_val_idx]
        val_phases = ([phase_sequences[i] for i in local_val_idx]
                      if phase_sequences is not None else None)
        val_labels = torch.tensor(valid_y[local_val_idx], dtype=torch.long).to(device)
        val_padded = pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in val_seqs],
            batch_first=True, padding_value=0
        ).to(device)
        val_mask = torch.zeros_like(val_padded)
        for i, s in enumerate(val_seqs):
            val_mask[i, :len(s)] = 1

        val_phase_padded = None
        if val_phases:
            val_phase_padded = pad_sequence(
                [torch.tensor(p, dtype=torch.long) for p in val_phases],
                batch_first=True, padding_value=0
            ).to(device)

        idx_val     = torch.tensor(local_val_idx)
        val_subseq  = subseq_features[idx_val].to(device) if subseq_features is not None else None
        val_global  = global_features[idx_val].to(device) if global_features is not None else None

        best_bacc, best_state, patience_counter = 0, None, 0

        for epoch in range(cfg.max_epochs):
            model.train()
            total_loss = 0
            for batch_indices, batch_labels_raw in train_loader:
                idx_list     = list(batch_indices)
                batch_seqs   = [ttp_sequences[i] for i in idx_list]
                batch_phases = ([phase_sequences[i] for i in idx_list]
                                if phase_sequences is not None else None)

                padded = pad_sequence(
                    [torch.tensor(s, dtype=torch.long) for s in batch_seqs],
                    batch_first=True, padding_value=0
                ).to(device)
                mask = torch.zeros_like(padded)
                for i, s in enumerate(batch_seqs):
                    mask[i, :len(s)] = 1

                padded_phases = None
                if batch_phases:
                    padded_phases = pad_sequence(
                        [torch.tensor(p, dtype=torch.long) for p in batch_phases],
                        batch_first=True, padding_value=0
                    ).to(device)

                idx_t        = torch.tensor(idx_list)
                batch_subseq = subseq_features[idx_t].to(device) if subseq_features is not None else None
                batch_global = global_features[idx_t].to(device) if global_features is not None else None
                batch_labels_t = torch.tensor(list(batch_labels_raw), dtype=torch.long).to(device)

                logits = model(padded, padded_phases, batch_subseq, batch_global, mask)
                loss   = criterion(logits, batch_labels_t)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_logits = model(val_padded, val_phase_padded, val_subseq, val_global, val_mask)
                val_bacc   = balanced_accuracy_score(
                    val_labels.cpu().numpy(), val_logits.argmax(1).cpu().numpy()
                )

            scheduler.step(val_bacc)
            if val_bacc > best_bacc:
                best_bacc, best_state, patience_counter = val_bacc, \
                    {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

            if (epoch + 1) % cfg.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"        Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f}"
                      f" | Val B-Acc: {val_bacc:.4f} | LR: {lr:.6f}")

        model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    # 评估（IOC）
    # ------------------------------------------------------------------
    def _eval_ioc(self, model, ioc_data, global_test_idx):
        model.eval()
        all_preds, all_labels = [], []
        test_loader = NeighborLoader(
            ioc_data, num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            input_nodes=('EVENT', torch.tensor(global_test_idx, dtype=torch.long)),
            shuffle=False
        )
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.config.device)
                bs    = batch['EVENT'].batch_size
                x_dict          = {nt: batch[nt].x for nt in batch.node_types
                                   if nt in ['IP', 'domain', 'URL', 'File', 'CVE', 'ASN', 'EVENT']}
                edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
                edge_attr_dict  = _build_edge_attr_dict(batch)
                num_nodes_dict  = _build_num_nodes_dict(batch)
                emb = model(x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict)
                all_preds.append(model.classifier(emb[:bs]).argmax(1).cpu())
                all_labels.append(batch['EVENT'].y[:bs].cpu())

        preds  = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        return accuracy_score(labels, preds), preds, labels

    # ------------------------------------------------------------------
    # 评估（TTP）
    # ------------------------------------------------------------------
    def _eval_ttp(self, model, ttp_sequences, phase_sequences,
                  subseq_features, global_features, local_test_idx, valid_y):
        model.eval()
        device = self.config.device

        test_seqs   = [ttp_sequences[i] for i in local_test_idx]
        test_phases = ([phase_sequences[i] for i in local_test_idx]
                       if phase_sequences is not None else None)
        test_y      = valid_y[local_test_idx]

        test_padded = pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in test_seqs],
            batch_first=True, padding_value=0
        ).to(device)
        test_mask = torch.zeros_like(test_padded)
        for i, s in enumerate(test_seqs):
            test_mask[i, :len(s)] = 1

        test_phase_padded = None
        if test_phases:
            test_phase_padded = pad_sequence(
                [torch.tensor(p, dtype=torch.long) for p in test_phases],
                batch_first=True, padding_value=0
            ).to(device)

        idx_t       = torch.tensor(local_test_idx)
        test_subseq = subseq_features[idx_t].to(device) if subseq_features is not None else None
        test_global = global_features[idx_t].to(device) if global_features is not None else None

        with torch.no_grad():
            logits = model(
                test_padded, test_phase_padded, test_subseq, test_global, test_mask
            )
            preds = logits.argmax(1).cpu().numpy()

        return accuracy_score(test_y, preds), preds, test_y

    # ------------------------------------------------------------------
    # 辅助：构建标准TTP模型
    # ------------------------------------------------------------------
    def _build_standard_ttp(self, num_techniques, num_classes, ttp_data):
        return TTPTransformer(
            num_techniques=num_techniques,
            num_classes=num_classes,
            cfg=self.ttp_cfg,
            pretrained_embeddings=ttp_data.get('technique_embeddings'),
            num_phases=ttp_data.get('num_phases', 14),
            num_subseq_features=ttp_data.get('num_feature_subseq', 0),
            global_feature_dim=ttp_data.get('global_feature_dim', 5)
        )

    # ------------------------------------------------------------------
    # 结果输出
    # ------------------------------------------------------------------
    def _save_and_print(self, all_results: dict):
        print(f"\n{'='*80}")
        print("消融实验最终报告".center(80))
        print(f"{'='*80}")

        summary = {}
        header = f"  {'实验组':<8} {'名称':<20} {'Acc':>8} {'±':>6} {'B-Acc':>8} {'±':>6} {'F1':>8} {'±':>6}"
        print(header)
        print(f"  {'-'*76}")

        full_acc = None
        for exp in self.experiments:
            folds = all_results[exp]
            accs  = [r['acc']  for r in folds]
            baccs = [r['bacc'] for r in folds]
            f1s   = [r['f1']   for r in folds]
            summary[exp] = {
                'acc_mean':  np.mean(accs),  'acc_std':  np.std(accs),
                'bacc_mean': np.mean(baccs), 'bacc_std': np.std(baccs),
                'f1_mean':   np.mean(f1s),   'f1_std':   np.std(f1s),
            }
            if exp == 'Full':
                full_acc = np.mean(accs)

            name = self.EXPERIMENT_NAMES.get(exp, exp)
            print(f"  {exp:<8} {name:<20} "
                  f"{np.mean(accs):>8.4f} {np.std(accs):>6.4f} "
                  f"{np.mean(baccs):>8.4f} {np.std(baccs):>6.4f} "
                  f"{np.mean(f1s):>8.4f} {np.std(f1s):>6.4f}")

        if full_acc is not None:
            print(f"\n  相对完整模型的Acc差距(↓负数=消融后性能下降):")
            for exp, s in summary.items():
                if exp == 'Full':
                    continue
                delta = s['acc_mean'] - full_acc
                symbol = '↑' if delta > 0 else '↓'
                print(f"    [{exp}] {self.EXPERIMENT_NAMES.get(exp, exp):<20}: "
                      f"{delta:+.4f} {symbol}")

        print(f"{'='*80}")

        with open(self.results_dir / "ablation_results.json", "w") as f:
            json.dump({'per_fold': all_results, 'summary': summary}, f, indent=2)
        print(f"\n[完成] 结果已保存至: {self.results_dir / 'ablation_results.json'}")


# ==================== run() 补丁：C3软投票路由修正 ====================
# 因为 _soft_vote 需要访问训练好的模型对象，在 run() 中做特殊处理

_original_run = AblationStudy.run

def _patched_run(self, ioc_data, ttp_data, class_weights, valid_idx):
    """
    重写 run()，在 C3 评估时直接调用 _soft_vote_with_models()。
    其余逻辑与原 run() 完全一致。
    """
    valid_y        = ioc_data['EVENT'].y[valid_idx].numpy()
    num_classes    = len(np.unique(valid_y))
    ioc_input_dims = EmbeddingGraphDataProcessor.build_input_dims(ioc_data)

    ttp_sequences = ttp_data.get('causal_sequences',
                                  ttp_data.get('technique_sequences'))
    all_ids       = [id_ for seq in ttp_sequences for id_ in seq]
    num_techniques = max(
        (max(all_ids) + 1) if all_ids else 369,
        ttp_data.get('num_techniques', 369)
    )

    skf         = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
    all_results = {exp: [] for exp in self.experiments}

    for fold, (temp_local_train_idx, local_test_idx) in enumerate(
            skf.split(valid_idx, valid_y)):

        print(f"\n{'='*70}")
        print(f"Fold {fold+1}/{self.config.n_folds}".center(70))
        print(f"{'='*70}")

        local_train_idx, local_val_idx = train_test_split(
            temp_local_train_idx,
            test_size=self.config.val_ratio,
            stratify=valid_y[temp_local_train_idx],
            random_state=42
        )
        global_train_idx = valid_idx[local_train_idx]
        global_val_idx   = valid_idx[local_val_idx]
        global_test_idx  = valid_idx[local_test_idx]

        need_std_ioc = any(e in self.experiments for e in ['Full', 'C0', 'C3'])
        need_std_ttp = any(e in self.experiments for e in ['Full', 'C1', 'C3'])

        ioc_model = ttp_model = ioc_a1_model = ttp_b1_model = ttp_b2_model = None

        if need_std_ioc:
            print(f"\n  训练标准IOC模型")
            ioc_model = IOCClassifier(
                metadata=ioc_data.metadata(), input_dims=ioc_input_dims,
                num_classes=num_classes, cfg=self.ioc_cfg
            ).to(self.config.device)
            ioc_model = self._train_ioc(
                ioc_model, ioc_data, global_train_idx, global_val_idx, class_weights
            )

        if 'A1' in self.experiments:
            print(f"\n  训练A1-IOC模型（无边类型嵌入）")
            ioc_a1_model = IOCClassifierNoEdgeEmbed(
                metadata=ioc_data.metadata(), input_dims=ioc_input_dims,
                num_classes=num_classes, cfg=self.ioc_cfg
            ).to(self.config.device)
            ioc_a1_model = self._train_ioc(
                ioc_a1_model, ioc_data, global_train_idx, global_val_idx, class_weights
            )

        if need_std_ttp:
            print(f"\n  训练标准TTP模型")
            ttp_model = self._build_standard_ttp(
                num_techniques, num_classes, ttp_data
            ).to(self.config.device)
            ttp_model = self._train_ttp(
                ttp_model, ttp_sequences,
                ttp_data.get('phase_sequences'),
                ttp_data.get('subseq_features'),
                ttp_data.get('global_features'),
                local_train_idx, local_val_idx, valid_y, class_weights
            )

        if 'B1' in self.experiments:
            print(f"\n  训练B1-TTP模型（随机嵌入）")
            ttp_b1_model = build_ttp_model_b1(
                num_techniques, num_classes, self.ttp_cfg, ttp_data
            ).to(self.config.device)
            ttp_b1_model = self._train_ttp(
                ttp_b1_model, ttp_sequences,
                ttp_data.get('phase_sequences'),
                ttp_data.get('subseq_features'),
                ttp_data.get('global_features'),
                local_train_idx, local_val_idx, valid_y, class_weights
            )

        if 'B2' in self.experiments:
            print(f"\n  训练B2-TTP模型（无阶段嵌入）")
            ttp_b2_model = TTPTransformerNoPhase(
                num_techniques=num_techniques, num_classes=num_classes,
                cfg=self.ttp_cfg,
                pretrained_embeddings=ttp_data.get('technique_embeddings'),
                num_subseq_features=ttp_data.get('num_feature_subseq', 0),
                global_feature_dim=ttp_data.get('global_feature_dim', 5)
            ).to(self.config.device)
            ttp_b2_model = self._train_ttp(
                ttp_b2_model, ttp_sequences,
                None,
                ttp_data.get('subseq_features'),
                ttp_data.get('global_features'),
                local_train_idx, local_val_idx, valid_y, class_weights
            )

        # ---- 评估 ----
        ioc_preds = ioc_labels = ttp_preds = None
        test_labels = valid_y[local_test_idx]

        if ioc_model is not None:
            _, ioc_preds, ioc_labels = self._eval_ioc(ioc_model, ioc_data, global_test_idx)

        if ttp_model is not None:
            _, ttp_preds, _ = self._eval_ttp(
                ttp_model, ttp_sequences,
                ttp_data.get('phase_sequences'),
                ttp_data.get('subseq_features'),
                ttp_data.get('global_features'),
                local_test_idx, valid_y
            )

        for exp in self.experiments:
            if exp == 'Full':
                preds  = np.where(ioc_preds == ioc_labels, ioc_preds, ttp_preds)
                labels = ioc_labels

            elif exp == 'C0':
                preds, labels = ioc_preds, ioc_labels

            elif exp == 'C1':
                preds, labels = ttp_preds, test_labels

            elif exp == 'C3':
                preds  = self._soft_vote_with_models(
                    ioc_model, ttp_model, ioc_data, global_test_idx,
                    ttp_sequences, ttp_data, local_test_idx, valid_y
                )
                labels = ioc_labels   # IOC loader 保证顺序一致

            elif exp == 'A1':
                _, preds, labels = self._eval_ioc(ioc_a1_model, ioc_data, global_test_idx)

            elif exp == 'B1':
                _, preds, labels_b1 = self._eval_ttp(
                    ttp_b1_model, ttp_sequences,
                    ttp_data.get('phase_sequences'),
                    ttp_data.get('subseq_features'),
                    ttp_data.get('global_features'),
                    local_test_idx, valid_y
                )
                labels = labels_b1

            elif exp == 'B2':
                _, preds, labels_b2 = self._eval_ttp(
                    ttp_b2_model, ttp_sequences,
                    None,
                    ttp_data.get('subseq_features'),
                    ttp_data.get('global_features'),
                    local_test_idx, valid_y
                )
                labels = labels_b2
            else:
                raise ValueError(f"未知实验组: {exp}")

            metrics = {
                'acc':  accuracy_score(labels, preds),
                'bacc': balanced_accuracy_score(labels, preds),
                'f1':   f1_score(labels, preds, average='macro', zero_division=0),
            }
            all_results[exp].append({'fold': fold, **metrics})
            print(f"    [{exp}] {self.EXPERIMENT_NAMES.get(exp, exp)}: "
                  f"Acc={metrics['acc']:.4f}, B-Acc={metrics['bacc']:.4f}, "
                  f"F1={metrics['f1']:.4f}")

    self._save_and_print(all_results)


# 替换 run 方法
AblationStudy.run = _patched_run


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='双模型OR融合消融实验')
    parser.add_argument('--ioc-data',      type=str,  default='./apt_kg_ioc.pt')
    parser.add_argument('--ttp-data',      type=str,  default='./apt_kg_ttp.pt')
    parser.add_argument('--top-k-classes', type=int,  default=15)
    parser.add_argument('--batch-size',    type=int,  default=128)
    parser.add_argument('--device',        type=str,  default='cuda:0')
    parser.add_argument(
        '--experiments', nargs='+',
        default=['Full', 'C0', 'C1', 'C3', 'B1', 'B2', 'A1'],
        choices=['Full', 'C0', 'C1', 'C3', 'B1', 'B2', 'A1'],
        help='指定要运行的实验组（默认全部）'
    )
    args = parser.parse_args()

    config = EmbeddingGraphConfig(
        data_path=args.ioc_data,
        ioc_data_path=args.ioc_data,
        ttp_data_path=args.ttp_data,
        top_k_classes=args.top_k_classes,
        batch_size=args.batch_size
    )
    if args.device:
        config.device = torch.device(args.device)

    ioc_cfg = IOCModelConfig()
    ttp_cfg = TTPModelConfig()

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("双模型OR融合 消融实验".center(70))
    print(f"实验组: {args.experiments}".center(70))
    print(f"设备: {config.device}".center(70))
    print("=" * 70)

    # 加载数据
    print(f"\n[数据] 加载IOC数据: {config.ioc_data_path}")
    ioc_data = torch.load(config.ioc_data_path, weights_only=False)
    print(f"[数据] 加载TTP序列: {config.ttp_data_path}")
    ttp_data = torch.load(config.ttp_data_path, weights_only=False)

    sequences = ttp_data.get('causal_sequences', ttp_data.get('technique_sequences'))
    print(f"  序列数: {len(sequences)}, 技术数: {ttp_data.get('num_techniques', 369)}")

    # 数据对齐检查
    assert len(sequences) == len(ioc_data['EVENT'].y), \
        f"样本数量不匹配: TTP={len(sequences)}, IOC={len(ioc_data['EVENT'].y)}"

    # 过滤top-k类别
    original_classes = len(np.unique(
        ioc_data['EVENT'].y[ioc_data['EVENT'].y != -1].numpy()
    ))
    if config.top_k_classes < original_classes:
        print(f"[过滤] Top-{config.top_k_classes} 类别")
        ioc_data  = filter_top_k_classes(ioc_data, config.top_k_classes)
        valid_idx = torch.where(ioc_data['EVENT'].y != -1)[0].numpy()
        seq_key   = 'causal_sequences' if 'causal_sequences' in ttp_data else 'technique_sequences'
        raw_seqs  = ttp_data.get('causal_sequences', ttp_data.get('technique_sequences'))

        ttp_data_filtered = {seq_key: [raw_seqs[i] for i in valid_idx],
                             'labels': ttp_data['labels'][valid_idx],
                             'num_techniques': ttp_data.get('num_techniques', 369)}
        for key in ['phase_sequences', 'global_features', 'technique_embeddings',
                    'num_phases', 'global_feature_dim', 'num_feature_subseq',
                    'semantic_dim', 'num_events', 'num_classes', 'apt_classes']:
            if key not in ttp_data:
                continue
            value = ttp_data[key]
            if key in ['phase_sequences', 'global_features']:
                ttp_data_filtered[key] = (
                    value[valid_idx] if isinstance(value, torch.Tensor)
                    else [value[i] for i in valid_idx]
                )
            else:
                ttp_data_filtered[key] = value
        ttp_data = ttp_data_filtered
    else:
        valid_idx = torch.where(ioc_data['EVENT'].y != -1)[0].numpy()

    y_valid       = ioc_data['EVENT'].y[valid_idx].numpy()
    weights       = compute_class_weight('balanced', classes=np.unique(y_valid), y=y_valid)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(config.device)

    print(f"[数据] 类别数: {config.top_k_classes}, 有效样本数: {len(y_valid)}")

    ablation = AblationStudy(config, ioc_cfg, ttp_cfg, args.experiments)
    ablation.run(ioc_data, ttp_data, class_weights, valid_idx)


if __name__ == "__main__":
    main()