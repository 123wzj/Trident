"""
双模型OR融合训练脚本
IOC图模型 + TTP语义向量模型
只要任一模型预测正确即为正确
重构：
  - 新增 IOCModelConfig / TTPModelConfig 两个独立配置类
  - 所有模型超参数从硬编码迁移至对应配置类，可通过命令行独立调节
  - 删除 LearnableFusionModule 及相关调用（USE_LEARNABLE_FUSION 始终为 False）
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
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
    Metrics,
    filter_top_k_classes
)


# ==================== 独立模型配置 ====================

@dataclass
class IOCModelConfig:
    """IOC图模型超参数（独立可调）"""
    hidden_dim: int = 256
    num_layers: int = 3
    num_bases: int = 8
    dropout: float = 0.2
    edge_type_embed_dim: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-4
    max_epochs: int = 150
    scheduler_patience: int = 15
    log_interval: int = 10          # 每隔多少 epoch 打印一次

    def to_dict(self):
        return asdict(self)


@dataclass
class TTPModelConfig:
    """TTP序列模型超参数（独立可调）"""
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3
    dropout: float = 0.2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    scheduler_patience: int = 15
    min_lr: float = 1e-6
    log_interval: int = 10          # 每隔多少 epoch 打印一次

    def to_dict(self):
        return asdict(self)


# ==================== 工具函数 ====================

def _build_edge_attr_dict(batch) -> dict:
    """[Bug2] 严格过滤 None 和空 tensor"""
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
    """[Bug7] 覆盖所有 node_types，缺失时打印警告"""
    result = {}
    for nt in batch.node_types:
        result[nt] = batch[nt].num_nodes
    return result


# ==================== IOC模型 ====================

class IOCClassifier(nn.Module):
    """IOC图分类器（RGCN + 边类型嵌入）"""

    def __init__(self, metadata, input_dims, num_classes: int,
                 cfg: IOCModelConfig):
        super().__init__()
        hidden_dim          = cfg.hidden_dim
        self.hidden_dim     = hidden_dim
        self.num_layers     = cfg.num_layers
        self.num_classes    = num_classes
        self.node_types     = metadata[0]
        self.edge_types     = metadata[1]
        self.dropout        = cfg.dropout

        # 输入投影层
        self.input_projs = nn.ModuleDict()
        for node_type, dim in input_dims.items():
            if dim is not None and dim > 0:
                self.input_projs[node_type] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                )

        # 边类型嵌入
        self.edge_type_embedding = nn.Embedding(
            num_embeddings=len(metadata[1]),
            embedding_dim=cfg.edge_type_embed_dim
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(cfg.edge_type_embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout)
        )
        self.edge_type_map = {etype: i for i, etype in enumerate(self.edge_types)}

        # RGCN 层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(RGCNConv(
                in_channels=hidden_dim, out_channels=hidden_dim,
                num_relations=len(metadata[1]), num_bases=cfg.num_bases
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict):
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type] is not None:
                if node_type in self.input_projs:
                    h_dict[node_type] = self.input_projs[node_type](x_dict[node_type])
                else:
                    num_nodes = num_nodes_dict.get(node_type, 0)
                    if num_nodes == 0:
                        print(f"[警告] node_type={node_type} 不在 num_nodes_dict 中，补零")
                    h_dict[node_type] = torch.zeros(
                        max(num_nodes, 1), self.hidden_dim,
                        device=x_dict[list(x_dict.keys())[0]].device
                    )
            else:
                num_nodes = num_nodes_dict.get(node_type, 0)
                if num_nodes == 0:
                    print(f"[警告] node_type={node_type} 无特征且不在 num_nodes_dict 中，补零")
                device = x_dict[list(x_dict.keys())[0]].device if x_dict else 'cpu'
                h_dict[node_type] = torch.zeros(
                    max(num_nodes, 1), self.hidden_dim, device=device
                )

        x_all, node_offsets = self._concat_node_features(h_dict)
        edge_index_all, edge_type_all, edge_weights = self._build_global_edges(
            edge_index_dict, edge_attr_dict, node_offsets, x_all.device
        )
        if edge_index_all is None:
            return h_dict['EVENT']

        edge_embeds = self.edge_type_embedding(edge_type_all)
        edge_features = torch.cat([edge_embeds, edge_weights.unsqueeze(1)], dim=-1)
        edge_message_enhancement = self.edge_mlp(edge_features)
        h = self._message_passing(x_all, edge_index_all, edge_type_all, edge_message_enhancement)

        event_start = node_offsets['EVENT']
        event_end   = event_start + h_dict['EVENT'].shape[0]
        return h[event_start:event_end]

    def _concat_node_features(self, h_dict):
        x_all, node_offsets, curr_offset = [], {}, 0
        for ntype in self.node_types:
            feat = h_dict[ntype]
            x_all.append(feat)
            node_offsets[ntype] = curr_offset
            curr_offset += feat.shape[0]
        return torch.cat(x_all, dim=0), node_offsets

    def _build_global_edges(self, edge_index_dict, edge_attr_dict, node_offsets, device):
        edge_indices, edge_types_list, edge_weights_list = [], [], []
        for edge_key, edge_index in edge_index_dict.items():
            if edge_index is None or edge_index.numel() == 0:
                continue
            src_t, _, dst_t = edge_key
            rel_id = self.edge_type_map[edge_key]
            new_idx = edge_index.clone()
            new_idx[0] += node_offsets[src_t]
            new_idx[1] += node_offsets[dst_t]
            edge_indices.append(new_idx)
            edge_types_list.append(
                torch.full((edge_index.shape[1],), rel_id, dtype=torch.long, device=device)
            )
            ea = edge_attr_dict.get(edge_key, None)
            if ea is not None and ea.numel() > 0 and ea.shape[1] >= 2:
                weights = ea[:, 1].to(device)
            else:
                weights = torch.ones(edge_index.shape[1], device=device)
            edge_weights_list.append(weights)

        if not edge_indices:
            return None, None, None
        return (
            torch.cat(edge_indices, dim=1),
            torch.cat(edge_types_list, dim=0),
            torch.cat(edge_weights_list, dim=0)
        )

    def _message_passing(self, h, edge_index, edge_type, edge_message_enhancement=None):
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_type)
            if edge_message_enhancement is not None:
                src, dst = edge_index
                edge_enhancement_aggregated = torch.zeros_like(h)
                edge_enhancement_aggregated.scatter_add_(
                    0, dst.unsqueeze(1).expand(-1, h.shape[1]), edge_message_enhancement
                )
                node_degrees = torch.zeros_like(h)
                node_degrees.scatter_add_(
                    0, dst.unsqueeze(1).expand(-1, h.shape[1]),
                    torch.ones_like(edge_message_enhancement)
                )
                node_degrees = torch.clamp(node_degrees, min=1)
                h_new = h_new + 0.1 * (edge_enhancement_aggregated / node_degrees)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h
        return h


# ==================== TTP模型 ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TTPTransformer(nn.Module):
    """阶段感知 + 子序列特征 + 全局特征的TTP序列模型"""

    def __init__(self, num_techniques: int, num_classes: int,
                 cfg: TTPModelConfig,
                 pretrained_embeddings=None,
                 num_phases: int = 14,
                 num_subseq_features: int = 0,
                 global_feature_dim: int = 6):
        super().__init__()
        d_model          = cfg.d_model
        self.num_classes = num_classes
        self.num_phases  = num_phases

        # 技术语义嵌入
        if pretrained_embeddings is not None:
            self.tech_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.float(), padding_idx=0, freeze=False
            )
            semantic_dim = pretrained_embeddings.shape[1]
            self.semantic_proj = (
                nn.Sequential(nn.Linear(semantic_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
                if semantic_dim != d_model else nn.Identity()
            )
            print(f"    [TTP模型] 预训练语义嵌入: {semantic_dim}维 → {d_model}维")
        else:
            self.tech_embedding = nn.Embedding(num_techniques, d_model, padding_idx=0)
            self.semantic_proj  = nn.Identity()
            print(f"    [TTP模型] 随机初始化嵌入")

        # 阶段嵌入
        phase_embed_dim = 16
        self.phase_embedding = nn.Embedding(num_phases, phase_embed_dim, padding_idx=0)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + phase_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cfg.nhead,
            dim_feedforward=d_model * 4,
            dropout=cfg.dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # 子序列特征
        self.use_subseq = num_subseq_features > 0
        self.subseq_mlp = (
            nn.Sequential(
                nn.Linear(num_subseq_features, d_model // 2),
                nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(cfg.dropout)
            ) if self.use_subseq else None
        )

        # 全局特征
        self.use_global = global_feature_dim > 0
        self.global_mlp = (
            nn.Sequential(
                nn.Linear(global_feature_dim, d_model // 4),
                nn.LayerNorm(d_model // 4), nn.GELU(), nn.Dropout(cfg.dropout)
            ) if self.use_global else None
        )

        # 分类器
        classifier_input_dim = d_model
        if self.use_subseq:
            classifier_input_dim += d_model // 2
        if self.use_global:
            classifier_input_dim += d_model // 4

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.LayerNorm(classifier_input_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        print(f"    [TTP模型] 阶段数: {num_phases}, 子序列特征: {num_subseq_features}, "
              f"全局特征: {global_feature_dim}, 分类器输入: {classifier_input_dim}维")

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                if not hasattr(module, 'weight') or module.weight.shape[1] != 384:
                    nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, technique_sequences, phase_sequences=None,
                subseq_features=None, global_features=None, attention_mask=None):
        x = self.semantic_proj(self.tech_embedding(technique_sequences))
        if phase_sequences is not None:
            x = self.fusion_proj(torch.cat([x, self.phase_embedding(phase_sequences)], dim=-1))
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


# ==================== 双模型OR融合训练器 ====================

class DualORFusionTrainer:
    """双模型OR融合训练器"""

    def __init__(self, config: EmbeddingGraphConfig,
                 ioc_cfg: IOCModelConfig,
                 ttp_cfg: TTPModelConfig):
        self.config  = config
        self.ioc_cfg = ioc_cfg
        self.ttp_cfg = ttp_cfg

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path("../v2/results") / f"dual_or_fusion_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(self.results_dir / "config.json", "w") as f:
            json.dump({
                "base":  config.to_dict(),
                "ioc":   ioc_cfg.to_dict(),
                "ttp":   ttp_cfg.to_dict(),
            }, f, indent=2)

        print(f"[双模型OR融合] 结果目录: {self.results_dir}")
        print(f"  IOC配置: {ioc_cfg.to_dict()}")
        print(f"  TTP配置: {ttp_cfg.to_dict()}")

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def train_and_evaluate(self, ioc_data: Any, ttp_data: dict,
                           class_weights: torch.Tensor,
                           valid_idx: np.ndarray):
        print(f"\n{'='*70}")
        print("开始双模型OR融合训练".center(70))
        print(f"{'='*70}")

        valid_y      = ioc_data['EVENT'].y[valid_idx].numpy()
        num_classes  = len(np.unique(valid_y))
        ttp_sequences = ttp_data.get('causal_sequences', ttp_data.get('technique_sequences'))

        all_ids = [id_ for seq in ttp_sequences for id_ in seq]
        num_techniques = max(
            (max(all_ids) + 1) if all_ids else 369,
            ttp_data.get('num_techniques', 369)
        )
        seq_type = "因果序列" if 'causal_sequences' in ttp_data else "原始序列"
        print(f"  [TTP数据] 序列类型: {seq_type}, 技术ID范围: 1~{max(all_ids)}, "
              f"num_techniques={num_techniques}")

        ioc_input_dims = EmbeddingGraphDataProcessor.build_input_dims(ioc_data)
        skf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (temp_local_train_idx, local_test_idx) in enumerate(
                skf.split(valid_idx, valid_y)):

            print(f"\n{'='*70}")
            print(f"Fold {fold+1}/{self.config.n_folds}".center(70))
            print(f"{'='*70}")

            # [Bug3] 明确区分局部位置索引与全局节点索引
            local_train_idx, local_val_idx = train_test_split(
                temp_local_train_idx,
                test_size=self.config.val_ratio,
                stratify=valid_y[temp_local_train_idx],
                random_state=42
            )
            global_train_idx = valid_idx[local_train_idx]
            global_val_idx   = valid_idx[local_val_idx]
            global_test_idx  = valid_idx[local_test_idx]

            # -------- IOC --------
            print(f"\n  [1/2] 训练IOC模型")
            ioc_model = IOCClassifier(
                metadata=ioc_data.metadata(),
                input_dims=ioc_input_dims,
                num_classes=num_classes,
                cfg=self.ioc_cfg
            ).to(self.config.device)
            ioc_model = self._train_ioc_model(
                ioc_model, ioc_data, global_train_idx, global_val_idx,
                class_weights, fold
            )

            # -------- TTP --------
            print(f"\n  [2/2] 训练TTP模型")
            num_phases          = ttp_data.get('num_phases', 14)
            num_subseq_features = ttp_data.get('num_feature_subseq', 0)
            global_feature_dim  = ttp_data.get('global_feature_dim', 5)
            phase_sequences     = ttp_data.get('phase_sequences', None)
            subseq_features     = ttp_data.get('subseq_features', None)
            global_features     = ttp_data.get('global_features', None)
            pretrained_emb      = ttp_data.get('technique_embeddings', None)

            ttp_model = TTPTransformer(
                num_techniques=num_techniques,
                num_classes=num_classes,
                cfg=self.ttp_cfg,
                pretrained_embeddings=pretrained_emb,
                num_phases=num_phases,
                num_subseq_features=num_subseq_features,
                global_feature_dim=global_feature_dim
            ).to(self.config.device)
            ttp_model = self._train_ttp_model(
                ttp_model, ttp_sequences, phase_sequences, subseq_features, global_features,
                local_train_idx, local_val_idx, valid_y, class_weights, fold
            )

            # -------- OR 融合评估 --------
            print(f"\n  >>> OR融合评估")
            ioc_acc, ioc_preds, ioc_labels = self._evaluate_ioc_model(
                ioc_model, ioc_data, global_test_idx
            )
            ttp_acc, ttp_preds, _ = self._evaluate_ttp_model(
                ttp_model, ttp_sequences, phase_sequences, subseq_features,
                global_features, local_test_idx, valid_y
            )

            or_preds = np.where(ioc_preds == ioc_labels, ioc_preds, ttp_preds)
            or_acc   = accuracy_score(ioc_labels, or_preds)
            or_bacc  = balanced_accuracy_score(ioc_labels, or_preds)
            or_f1    = f1_score(ioc_labels, or_preds, average='macro')

            print(f"\n    结果对比:")
            print(f"      IOC单独:  Acc={ioc_acc:.4f}")
            print(f"      TTP单独:  Acc={ttp_acc:.4f}")
            print(f"      OR融合:   Acc={or_acc:.4f}, B-Acc={or_bacc:.4f}, F1={or_f1:.4f}")
            print(f"      提升:     +{or_acc - max(ioc_acc, ttp_acc):.4f}")

            fold_results.append({
                'fold': fold, 'ioc_acc': ioc_acc, 'ttp_acc': ttp_acc,
                'or_acc': or_acc, 'or_bacc': or_bacc, 'or_f1_macro': or_f1
            })
            torch.save(ioc_model.state_dict(), self.results_dir / f"ioc_model_fold{fold}.pt")
            torch.save(ttp_model.state_dict(), self.results_dir / f"ttp_model_fold{fold}.pt")

        self._print_final_results(fold_results)

    # ------------------------------------------------------------------
    # IOC 训练
    # ------------------------------------------------------------------
    def _train_ioc_model(self, model: nn.Module, ioc_data: Any,
                         train_idx: np.ndarray, val_idx: np.ndarray,
                         class_weights: torch.Tensor, fold: int) -> nn.Module:
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
                event_emb = model(x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict)
                loss = criterion(model.classifier(event_emb[:bs]), batch['EVENT'].y[:bs])
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
                    event_emb = model(x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict)
                    val_preds.append(model.classifier(event_emb[:bs]).argmax(1).cpu())
                    val_labels.append(batch['EVENT'].y[:bs].cpu())

            val_bacc = balanced_accuracy_score(
                torch.cat(val_labels).numpy(), torch.cat(val_preds).numpy()
            )
            scheduler.step(val_bacc)

            if val_bacc > best_bacc:
                best_bacc = val_bacc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

            if (epoch + 1) % cfg.log_interval == 0:
                print(f"      Epoch {epoch+1:03d} | Val B-Acc: {val_bacc:.4f}")

        model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    # TTP 训练
    # ------------------------------------------------------------------
    def _train_ttp_model(self, model: nn.Module, ttp_sequences: list,
                         phase_sequences, subseq_features, global_features,
                         local_train_idx: np.ndarray, local_val_idx: np.ndarray,
                         valid_y: np.ndarray, class_weights: torch.Tensor,
                         fold: int) -> nn.Module:
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

        train_dataset = list(zip(local_train_idx.tolist(), valid_y[local_train_idx].tolist()))
        train_loader  = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
        )

        # 验证集一次性准备
        val_seqs   = [ttp_sequences[i] for i in local_val_idx]
        val_phases = [phase_sequences[i] for i in local_val_idx] if phase_sequences else None
        val_labels = torch.tensor(valid_y[local_val_idx], dtype=torch.long).to(device)

        val_padded = pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in val_seqs],
            batch_first=True, padding_value=0
        ).to(device)
        val_mask = torch.zeros_like(val_padded)
        for i, seq in enumerate(val_seqs):
            val_mask[i, :len(seq)] = 1

        val_phase_padded = None
        if val_phases:
            val_phase_padded = pad_sequence(
                [torch.tensor(p, dtype=torch.long) for p in val_phases],
                batch_first=True, padding_value=0
            ).to(device)

        # [Bug4] torch.tensor 索引
        idx_val  = torch.tensor(local_val_idx)
        val_subseq  = subseq_features[idx_val].to(device)  if subseq_features  is not None else None
        val_global  = global_features[idx_val].to(device)  if global_features  is not None else None

        best_bacc, best_state, patience_counter = 0, None, 0
        torch.manual_seed(42 + fold)

        for epoch in range(cfg.max_epochs):
            model.train()
            total_loss = 0
            for batch_indices, batch_labels_raw in train_loader:
                idx_list    = list(batch_indices)
                batch_seqs  = [ttp_sequences[i] for i in idx_list]
                batch_phases= [phase_sequences[i] for i in idx_list] if phase_sequences else None

                padded = pad_sequence(
                    [torch.tensor(s, dtype=torch.long) for s in batch_seqs],
                    batch_first=True, padding_value=0
                ).to(device)
                mask = torch.zeros_like(padded)
                for i, seq in enumerate(batch_seqs):
                    mask[i, :len(seq)] = 1

                padded_phases = None
                if batch_phases:
                    padded_phases = pad_sequence(
                        [torch.tensor(p, dtype=torch.long) for p in batch_phases],
                        batch_first=True, padding_value=0
                    ).to(device)

                # [Bug4]
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
                best_bacc = val_bacc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                # [Bug8] 使用 config.patience
                if patience_counter >= self.config.patience:
                    break

            if (epoch + 1) % cfg.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"      Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f}"
                      f" | Val B-Acc: {val_bacc:.4f} | LR: {lr:.6f}")

        model.load_state_dict(best_state)
        return model

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    def _evaluate_ioc_model(self, model, ioc_data, global_test_idx):
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
                event_emb = model(x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict)
                all_preds.append(model.classifier(event_emb[:bs]).argmax(1).cpu())
                all_labels.append(batch['EVENT'].y[:bs].cpu())

        preds  = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        return accuracy_score(labels, preds), preds, labels

    def _evaluate_ttp_model(self, model, ttp_sequences, phase_sequences,
                             subseq_features, global_features,
                             local_test_idx, valid_y):
        """[Bug3] 接收局部位置索引"""
        model.eval()
        device = self.config.device

        test_seqs   = [ttp_sequences[i] for i in local_test_idx]
        test_phases = [phase_sequences[i] for i in local_test_idx] if phase_sequences else None
        test_y      = valid_y[local_test_idx]

        test_padded = pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in test_seqs],
            batch_first=True, padding_value=0
        ).to(device)
        test_mask = torch.zeros_like(test_padded)
        for i, seq in enumerate(test_seqs):
            test_mask[i, :len(seq)] = 1

        test_phase_padded = None
        if test_phases:
            test_phase_padded = pad_sequence(
                [torch.tensor(p, dtype=torch.long) for p in test_phases],
                batch_first=True, padding_value=0
            ).to(device)

        # [Bug4]
        idx_t       = torch.tensor(local_test_idx)
        test_subseq = subseq_features[idx_t].to(device) if subseq_features is not None else None
        test_global = global_features[idx_t].to(device) if global_features is not None else None

        with torch.no_grad():
            logits = model(test_padded, test_phase_padded, test_subseq, test_global, test_mask)
            preds  = logits.argmax(1).cpu().numpy()

        return accuracy_score(test_y, preds), preds, test_y

    # ------------------------------------------------------------------
    # 结果输出
    # ------------------------------------------------------------------
    def _print_final_results(self, fold_results):
        print(f"\n{'='*70}")
        print("双模型OR融合最终报告".center(70))
        print(f"{'='*70}")
        print(f"\n  {'Fold':<8} {'IOC Acc':<12} {'TTP Acc':<12} "
              f"{'OR Acc':<12} {'OR B-Acc':<12}")
        print(f"  {'-'*60}")
        for r in fold_results:
            print(f"  {r['fold']:<8} {r['ioc_acc']:<12.4f} {r['ttp_acc']:<12.4f} "
                  f"{r['or_acc']:<12.4f} {r['or_bacc']:<12.4f}")

        or_accs  = [r['or_acc']      for r in fold_results]
        or_baccs = [r['or_bacc']     for r in fold_results]
        or_f1s   = [r['or_f1_macro'] for r in fold_results]
        print(f"\n  平均值:")
        print(f"    OR Acc:   {np.mean(or_accs):.4f} ± {np.std(or_accs):.4f}")
        print(f"    OR B-Acc: {np.mean(or_baccs):.4f} ± {np.std(or_baccs):.4f}")
        print(f"    OR F1:    {np.mean(or_f1s):.4f} ± {np.std(or_f1s):.4f}")
        print(f"{'='*70}")

        with open(self.results_dir / "final_results.json", "w") as f:
            json.dump(fold_results, f, indent=2)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='双模型OR融合训练')
    parser.add_argument('--ioc-data',      type=str,
                        default='./apt_kg_ioc.pt')
    parser.add_argument('--ttp-data',      type=str,
                        default='./apt_kg_ttp.pt')
    parser.add_argument('--top-k-classes', type=int, default=15)
    parser.add_argument('--batch-size',    type=int, default=128)
    parser.add_argument('--device',        type=str, default="cuda:0")
    args = parser.parse_args()

    # 模型超参数直接从配置类默认值读取，如需调整请修改 IOCModelConfig / TTPModelConfig
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
    print("双模型OR融合训练 (Transformer TTP)".center(70))
    print(f"设备: {config.device}".center(70))
    print("=" * 70)

    # 加载数据
    print(f"\n[数据] 加载IOC数据: {config.ioc_data_path}")
    ioc_data = torch.load(config.ioc_data_path, weights_only=False)
    print(f"[数据] 加载TTP序列: {config.ttp_data_path}")
    ttp_data = torch.load(config.ttp_data_path, weights_only=False)

    sequences = ttp_data.get('causal_sequences', ttp_data.get('technique_sequences'))
    seq_type  = "因果序列" if 'causal_sequences' in ttp_data else "原始序列"
    print(f"  序列类型: {seq_type}, 序列数: {len(sequences)}, "
          f"技术数: {ttp_data.get('num_techniques', 369)}")

    # 数据对齐验证
    if len(sequences) == len(ioc_data['EVENT'].y):
        print(f"  [OK] 样本数量一致: {len(sequences)}")
    else:
        print(f"  [ERROR] 样本数量不匹配! TTP:{len(sequences)} vs "
              f"IOC:{len(ioc_data['EVENT'].y)}")

    if 'labels' in ttp_data:
        sample_size  = min(5, len(ttp_data['labels']))
        labels_match = sum(
            ioc_data['EVENT'].y[i].item() == (
                ttp_data['labels'][i].item()
                if isinstance(ttp_data['labels'], torch.Tensor)
                else ttp_data['labels'][i]
            )
            for i in range(sample_size)
        )
        status = "[OK]" if labels_match == sample_size else "[ERROR]"
        print(f"  {status} 抽样标签一致: {labels_match}/{sample_size}")

    # 过滤 top-k 类别
    original_classes = len(np.unique(
        ioc_data['EVENT'].y[ioc_data['EVENT'].y != -1].numpy()
    ))
    if config.top_k_classes < original_classes:
        print(f"[过滤] Top-{config.top_k_classes} 类别")
        ioc_data  = filter_top_k_classes(ioc_data, config.top_k_classes)
        valid_idx = torch.where(ioc_data['EVENT'].y != -1)[0].numpy()
        raw_seqs  = ttp_data.get('causal_sequences', ttp_data.get('technique_sequences'))
        seq_key   = 'causal_sequences' if 'causal_sequences' in ttp_data else 'technique_sequences'

        # [Bug6] 序列键只赋值一次
        ttp_data_filtered = {
            seq_key:          [raw_seqs[i] for i in valid_idx],
            'labels':         ttp_data['labels'][valid_idx],
            'num_techniques': ttp_data.get('num_techniques', 369),
        }
        for key in ['phase_sequences', 'global_features', 'technique_embeddings',
                    'num_phases', 'global_feature_dim', 'semantic_dim',
                    'num_events', 'num_classes', 'apt_classes', 'padding_value',
                    'seq_stats', 'tactic_mapping', 'tactic_phase_order', 'sequence_type']:
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

    print(f"[数据] 类别数: {config.top_k_classes}, 样本数: {len(y_valid)}")

    trainer = DualORFusionTrainer(config, ioc_cfg, ttp_cfg)
    trainer.train_and_evaluate(ioc_data, ttp_data, class_weights, valid_idx)

    print(f"\n[完成] 结果已保存至: {trainer.results_dir}")


if __name__ == "__main__":
    main()