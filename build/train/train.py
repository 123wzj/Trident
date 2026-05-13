"""Dual-branch APT attribution training with entropy-aware decision fusion."""
import json
import random
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
from torch_geometric.data import HeteroData
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


@dataclass
class EmbeddingGraphConfig:
    """Shared training settings."""

    ioc_data_path: str = "./apt_kg_ioc.pt"
    ttp_data_path: str = "./apt_kg_ttp.pt"
    top_k_classes: int = 15
    seed: int = 42
    n_folds: int = 5
    val_ratio: float = 0.2
    patience: int = 50
    batch_size: int = 128
    num_neighbors: List[int] = field(default_factory=lambda: [30, 20])
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            result[key] = str(value) if isinstance(value, torch.device) else value
        return result


class FocalLoss(nn.Module):
    """Focal loss with optional class weights."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


class EmbeddingGraphDataProcessor:
    """Small data helper used by this training script."""

    @staticmethod
    def build_input_dims(data: HeteroData) -> Dict[str, Optional[int]]:
        input_dims = {}
        print("\n[Data] Node feature dimensions:")
        for node_type in data.node_types:
            x = data[node_type].x
            if x is None or x.shape[1] == 0:
                input_dims[node_type] = None
                print(f"  {node_type}: None")
            else:
                input_dims[node_type] = int(x.shape[1])
                print(f"  {node_type}: {x.shape[1]}")
        return input_dims


def filter_top_k_classes(data: HeteroData, top_k: int = 10) -> HeteroData:
    """Keep only the top-k classes by EVENT sample count and remap labels."""
    print(f"\n[Data] Keeping top-{top_k} classes by sample count...")
    valid_idx = data["EVENT"].y != -1
    valid_y = data["EVENT"].y[valid_idx]
    unique_labels, counts = torch.unique(valid_y, return_counts=True)
    class_counts = sorted(
        zip(unique_labels.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True
    )
    top_k_labels = [label for label, _ in class_counts[:top_k]]
    top_k_label_set = set(top_k_labels)
    filtered_mask = torch.zeros(len(data["EVENT"].y), dtype=torch.bool)
    for idx, label in enumerate(data["EVENT"].y):
        if label != -1 and label.item() in top_k_label_set:
            filtered_mask[idx] = True
    old_to_new = {
        old_label: new_label for new_label, old_label in enumerate(top_k_labels)
    }
    new_y = torch.full((len(data["EVENT"].y),), -1, dtype=torch.long)
    for idx in range(len(data["EVENT"].y)):
        if filtered_mask[idx]:
            new_y[idx] = old_to_new[data["EVENT"].y[idx].item()]
    data["EVENT"].y = new_y
    old_classes = getattr(
        data, "_apt_classes", [f"Class_{i}" for i in range(len(unique_labels))]
    )
    data._apt_classes = np.array([old_classes[label] for label in top_k_labels])
    print(f"  Kept samples: {int(filtered_mask.sum())}")
    print(f"  Kept classes: {len(top_k_labels)}")
    return data


@dataclass
class IOCModelConfig:
    """Fixed IOC branch hyperparameters."""

    hidden_dim: int = 256
    num_layers: int = 3
    num_bases: int = 8
    dropout: float = 0.2
    edge_type_embed_dim: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-4
    max_epochs: int = 150
    scheduler_patience: int = 15
    log_interval: int = 10

    def to_dict(self):
        return asdict(self)


@dataclass
class TTPModelConfig:
    """Fixed TTP branch hyperparameters."""

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
    log_interval: int = 10

    def to_dict(self):
        return asdict(self)


def _build_edge_attr_dict(batch) -> dict:
    """Return non-empty edge attributes from a sampled batch."""
    result = {}
    for et in batch.edge_types:
        if not hasattr(batch[et], "edge_attr"):
            continue
        ea = batch[et].edge_attr
        if ea is None or ea.numel() == 0:
            continue
        result[et] = ea
    return result


def _build_num_nodes_dict(batch) -> dict:
    """Return node counts for all node types in a sampled batch."""
    result = {}
    for nt in batch.node_types:
        result[nt] = batch[nt].num_nodes
    return result


def _set_global_seed(seed: int):
    """Set seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _align_optional_field(
    field, valid_idx: np.ndarray, total_events: int, field_name: str
):
    """Align optional tensor/list field to valid-index space."""
    if field is None:
        return None
    field_len = len(field)
    valid_len = len(valid_idx)
    if field_len == valid_len:
        return field
    if field_len == total_events:
        if isinstance(field, torch.Tensor):
            return field[torch.tensor(valid_idx, dtype=torch.long)]
        return [field[i] for i in valid_idx]
    raise ValueError(
        f"{field_name} length mismatch: got {field_len}, expected {valid_len} (valid) or {total_events} (global)"
    )


class IOCClassifier(nn.Module):
    """RGCN classifier for IOC graph evidence."""

    def __init__(self, metadata, input_dims, num_classes: int, cfg: IOCModelConfig):
        super().__init__()
        hidden_dim = cfg.hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = cfg.num_layers
        self.num_classes = num_classes
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.dropout = cfg.dropout
        self.input_projs = nn.ModuleDict()
        for node_type, dim in input_dims.items():
            if dim is not None and dim > 0:
                self.input_projs[node_type] = nn.Sequential(
                    nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
                )
        self.edge_type_embedding = nn.Embedding(
            num_embeddings=len(metadata[1]), embedding_dim=cfg.edge_type_embed_dim
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(cfg.edge_type_embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.edge_type_map = {etype: i for i, etype in enumerate(self.edge_types)}
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(
                RGCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_relations=len(metadata[1]),
                    num_bases=cfg.num_bases,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim // 2, num_classes),
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
                        print(
                            f"[WARN] Missing node count for {node_type}; using zeros."
                        )
                    h_dict[node_type] = torch.zeros(
                        max(num_nodes, 1),
                        self.hidden_dim,
                        device=x_dict[list(x_dict.keys())[0]].device,
                    )
            else:
                num_nodes = num_nodes_dict.get(node_type, 0)
                if num_nodes == 0:
                    print(f"[WARN] Missing features for {node_type}; using zeros.")
                device = x_dict[list(x_dict.keys())[0]].device if x_dict else "cpu"
                h_dict[node_type] = torch.zeros(
                    max(num_nodes, 1), self.hidden_dim, device=device
                )
        x_all, node_offsets = self._concat_node_features(h_dict)
        edge_index_all, edge_type_all, edge_weights = self._build_global_edges(
            edge_index_dict, edge_attr_dict, node_offsets, x_all.device
        )
        if edge_index_all is None:
            return h_dict["EVENT"]
        edge_embeds = self.edge_type_embedding(edge_type_all)
        edge_features = torch.cat([edge_embeds, edge_weights.unsqueeze(1)], dim=-1)
        edge_message_enhancement = self.edge_mlp(edge_features)
        h = self._message_passing(
            x_all, edge_index_all, edge_type_all, edge_message_enhancement
        )
        event_start = node_offsets["EVENT"]
        event_end = event_start + h_dict["EVENT"].shape[0]
        return h[event_start:event_end]

    def _concat_node_features(self, h_dict):
        x_all, node_offsets, curr_offset = [], {}, 0
        for ntype in self.node_types:
            feat = h_dict[ntype]
            x_all.append(feat)
            node_offsets[ntype] = curr_offset
            curr_offset += feat.shape[0]
        return torch.cat(x_all, dim=0), node_offsets

    def _build_global_edges(
        self, edge_index_dict, edge_attr_dict, node_offsets, device
    ):
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
                torch.full(
                    (edge_index.shape[1],), rel_id, dtype=torch.long, device=device
                )
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
            torch.cat(edge_weights_list, dim=0),
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
                    0,
                    dst.unsqueeze(1).expand(-1, h.shape[1]),
                    torch.ones_like(edge_message_enhancement),
                )
                node_degrees = torch.clamp(node_degrees, min=1)
                h_new = h_new + 0.1 * (edge_enhancement_aggregated / node_degrees)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h
        return h


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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TTPTransformer(nn.Module):
    """Transformer classifier for TTP sequence evidence."""

    def __init__(
        self,
        num_techniques: int,
        num_classes: int,
        cfg: TTPModelConfig,
        pretrained_embeddings=None,
        num_phases: int = 14,
        num_subseq_features: int = 0,
        global_feature_dim: int = 6,
    ):
        super().__init__()
        d_model = cfg.d_model
        self.num_classes = num_classes
        self.num_phases = num_phases
        if pretrained_embeddings is not None:
            self.tech_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.float(), padding_idx=0, freeze=False
            )
            semantic_dim = pretrained_embeddings.shape[1]
            self.semantic_proj = (
                nn.Sequential(
                    nn.Linear(semantic_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
                )
                if semantic_dim != d_model
                else nn.Identity()
            )
            print(f"    [TTP] pretrained embeddings: {semantic_dim} -> {d_model}")
        else:
            self.tech_embedding = nn.Embedding(num_techniques, d_model, padding_idx=0)
            self.semantic_proj = nn.Identity()
            print("    [TTP] random technique embeddings")
        phase_embed_dim = 16
        self.phase_embedding = nn.Embedding(num_phases, phase_embed_dim, padding_idx=0)
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + phase_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.nhead,
            dim_feedforward=d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )
        self.use_subseq = num_subseq_features > 0
        self.subseq_mlp = (
            nn.Sequential(
                nn.Linear(num_subseq_features, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            if self.use_subseq
            else None
        )
        self.use_global = global_feature_dim > 0
        self.global_mlp = (
            nn.Sequential(
                nn.Linear(global_feature_dim, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            if self.use_global
            else None
        )
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
            nn.Linear(classifier_input_dim // 2, num_classes),
        )
        print(
            f"    [TTP] phases={num_phases}, subseq_features={num_subseq_features}, "
            f"global_features={global_feature_dim}, classifier_input={classifier_input_dim}"
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                if not hasattr(module, "weight") or module.weight.shape[1] != 384:
                    nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        technique_sequences,
        phase_sequences=None,
        subseq_features=None,
        global_features=None,
        attention_mask=None,
    ):
        x = self.semantic_proj(self.tech_embedding(technique_sequences))
        if phase_sequences is not None:
            x = self.fusion_proj(
                torch.cat([x, self.phase_embedding(phase_sequences)], dim=-1)
            )
        x = self.pos_encoding(x)
        src_key_padding_mask = (
            (attention_mask == 0) if attention_mask is not None else None
        )
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(
                min=1
            )
        else:
            pooled = encoded.mean(1)
        features = [pooled]
        if self.use_subseq and subseq_features is not None:
            features.append(self.subseq_mlp(subseq_features))
        if self.use_global and global_features is not None:
            features.append(self.global_mlp(global_features))
        return self.classifier(torch.cat(features, dim=-1))


# Dual-branch trainer
class DualDecisionFusionTrainer:
    """Dual-branch trainer with entropy-aware decision fusion."""

    def __init__(
        self,
        config: EmbeddingGraphConfig,
        ioc_cfg: IOCModelConfig,
        ttp_cfg: TTPModelConfig,
    ):
        self.config = config
        self.ioc_cfg = ioc_cfg
        self.ttp_cfg = ttp_cfg
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / f"dual_decision_fusion_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / "config.json", "w") as f:
            json.dump(
                {
                    "base": config.to_dict(),
                    "ioc": ioc_cfg.to_dict(),
                    "ttp": ttp_cfg.to_dict(),
                },
                f,
                indent=2,
            )
        print(f"[Dual decision fusion] Results directory: {self.results_dir}")
        print(f"  IOC config: {ioc_cfg.to_dict()}")
        print(f"  TTP config: {ttp_cfg.to_dict()}")
        print("  Fusion: entropy-aware decision fusion")

    def train_and_evaluate(self, ioc_data: Any, ttp_data: dict, valid_idx: np.ndarray):
        print(f"\n{'='*70}")
        print("Start dual-branch training".center(70))
        print(f"{'='*70}")
        valid_y = ioc_data["EVENT"].y[valid_idx].numpy()
        num_classes = len(np.unique(valid_y))
        ttp_sequences = ttp_data.get(
            "causal_sequences", ttp_data.get("technique_sequences")
        )
        if ttp_sequences is None:
            raise KeyError(
                "TTP data must contain 'causal_sequences' or 'technique_sequences'."
            )
        all_ids = [id_ for seq in ttp_sequences for id_ in seq]
        num_techniques = max(
            (max(all_ids) + 1) if all_ids else 369, ttp_data.get("num_techniques", 369)
        )
        seq_type = "causal" if "causal_sequences" in ttp_data else "raw"
        print(
            f"  [TTP data] sequence_type={seq_type}, "
            f"num_techniques={num_techniques}"
        )
        ioc_input_dims = EmbeddingGraphDataProcessor.build_input_dims(ioc_data)
        skf = StratifiedKFold(
            n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed
        )
        fold_results = []
        for fold, (temp_local_train_idx, local_test_idx) in enumerate(
            skf.split(valid_idx, valid_y)
        ):
            print(f"\n{'='*70}")
            print(f"Fold {fold+1}/{self.config.n_folds}".center(70))
            print(f"{'='*70}")
            local_train_idx, local_val_idx = train_test_split(
                temp_local_train_idx,
                test_size=self.config.val_ratio,
                stratify=valid_y[temp_local_train_idx],
                random_state=self.config.seed,
            )
            global_train_idx = valid_idx[local_train_idx]
            global_val_idx = valid_idx[local_val_idx]
            global_test_idx = valid_idx[local_test_idx]
            # Fold-level class weights prevent leakage from test distribution.
            fold_class_weights = self._compute_fold_class_weights(
                valid_y[local_train_idx], num_classes
            )
            _set_global_seed(self.config.seed + fold)
            print("\n  [1/2] Train IOC branch")
            ioc_model = IOCClassifier(
                metadata=ioc_data.metadata(),
                input_dims=ioc_input_dims,
                num_classes=num_classes,
                cfg=self.ioc_cfg,
            ).to(self.config.device)
            ioc_model = self._train_ioc_model(
                ioc_model,
                ioc_data,
                global_train_idx,
                global_val_idx,
                fold_class_weights,
                fold,
            )
            print("\n  [2/2] Train TTP branch")
            num_phases = ttp_data.get("num_phases", 14)
            num_subseq_features = ttp_data.get(
                "num_feature_subseq", ttp_data.get("num_subseq_features", 0)
            )
            global_feature_dim = ttp_data.get("global_feature_dim", 0)
            phase_sequences = ttp_data.get("phase_sequences", None)
            subseq_features = ttp_data.get("subseq_features", None)
            global_features = ttp_data.get("global_features", None)
            pretrained_emb = ttp_data.get("technique_embeddings", None)
            inferred_subseq_dim = (
                int(subseq_features.shape[1])
                if isinstance(subseq_features, torch.Tensor)
                else (len(subseq_features[0]) if subseq_features else 0)
            )
            inferred_global_dim = (
                int(global_features.shape[1])
                if isinstance(global_features, torch.Tensor)
                else (len(global_features[0]) if global_features else 0)
            )
            if num_subseq_features == 0 and inferred_subseq_dim > 0:
                num_subseq_features = inferred_subseq_dim
            if global_feature_dim == 0 and inferred_global_dim > 0:
                global_feature_dim = inferred_global_dim
            if num_subseq_features == 0:
                print("    [TTP] subsequence features are disabled.")
            ttp_model = TTPTransformer(
                num_techniques=num_techniques,
                num_classes=num_classes,
                cfg=self.ttp_cfg,
                pretrained_embeddings=pretrained_emb,
                num_phases=num_phases,
                num_subseq_features=num_subseq_features,
                global_feature_dim=global_feature_dim,
            ).to(self.config.device)
            ttp_model = self._train_ttp_model(
                ttp_model,
                ttp_sequences,
                phase_sequences,
                subseq_features,
                global_features,
                local_train_idx,
                local_val_idx,
                valid_y,
                fold_class_weights,
                fold,
                cfg_override=self.ttp_cfg,
            )
            ttp_val_acc, _, _, _ = self._evaluate_ttp_model(
                ttp_model,
                ttp_sequences,
                phase_sequences,
                subseq_features,
                global_features,
                local_val_idx,
                valid_y,
            )
            print(f"    [TTP] Val Acc={ttp_val_acc:.4f} | fixed parameters")
            # -------- Decision fusion evaluation --------
            _, _, ioc_val_labels, ioc_val_probs = self._evaluate_ioc_model(
                ioc_model, ioc_data, global_val_idx
            )
            _, _, ttp_val_labels, ttp_val_probs = self._evaluate_ttp_model(
                ttp_model,
                ttp_sequences,
                phase_sequences,
                subseq_features,
                global_features,
                local_val_idx,
                valid_y,
            )
            if not np.array_equal(ioc_val_labels, ttp_val_labels):
                raise ValueError(
                    "IOC/TTP validation labels are not aligned for decision fusion."
                )
            (
                val_fused_probs,
                val_graph_weight,
                val_graph_conf,
                val_seq_conf,
            ) = self._entropy_aware_fusion(ioc_val_probs, ttp_val_probs)
            val_ioc_acc = accuracy_score(ioc_val_labels, ioc_val_probs.argmax(axis=1))
            val_ttp_acc = accuracy_score(ioc_val_labels, ttp_val_probs.argmax(axis=1))
            val_fusion_acc = accuracy_score(
                ioc_val_labels, val_fused_probs.argmax(axis=1)
            )
            print(
                f"    [ValFusion] IOC Acc={val_ioc_acc:.4f}, TTP Acc={val_ttp_acc:.4f}, "
                f"Fusion Acc={val_fusion_acc:.4f}, mean graph weight={val_graph_weight.mean():.4f}"
            )
            print("\n  >>> Decision fusion evaluation")
            ioc_acc, ioc_preds, ioc_labels, ioc_probs = self._evaluate_ioc_model(
                ioc_model, ioc_data, global_test_idx
            )
            ttp_acc, ttp_preds, ttp_labels, ttp_probs = self._evaluate_ttp_model(
                ttp_model,
                ttp_sequences,
                phase_sequences,
                subseq_features,
                global_features,
                local_test_idx,
                valid_y,
            )
            if not np.array_equal(ioc_labels, ttp_labels):
                raise ValueError(
                    "IOC/TTP test labels are not aligned for decision fusion."
                )
            (
                fused_probs,
                graph_weight,
                graph_conf,
                seq_conf,
            ) = self._entropy_aware_fusion(ioc_probs, ttp_probs)
            or_preds = fused_probs.argmax(axis=1)
            fusion_mode = "entropy_aware"
            or_acc = accuracy_score(ioc_labels, or_preds)
            or_bacc = balanced_accuracy_score(ioc_labels, or_preds)
            or_f1 = f1_score(ioc_labels, or_preds, average="macro")
            oracle_mask = (ioc_preds == ioc_labels) | (ttp_preds == ioc_labels)
            oracle_acc = float(np.mean(oracle_mask))
            print("\n    Results:")
            print(f"      IOC:    Acc={ioc_acc:.4f}")
            print(f"      TTP:    Acc={ttp_acc:.4f}")
            print(
                f"      Fusion: Acc={or_acc:.4f}, B-Acc={or_bacc:.4f}, F1={or_f1:.4f}"
            )
            print(f"      Mode:   {fusion_mode}")
            print(f"      Oracle: Acc={oracle_acc:.4f}")
            print(f"      Gain:   +{or_acc - max(ioc_acc, ttp_acc):.4f}")
            fold_results.append(
                {
                    "fold": fold,
                    "ioc_acc": ioc_acc,
                    "ttp_acc": ttp_acc,
                    "or_acc": or_acc,
                    "or_bacc": or_bacc,
                    "or_f1_macro": or_f1,
                    "oracle_acc": oracle_acc,
                    "val_ioc_acc": float(val_ioc_acc),
                    "val_ttp_acc": float(val_ttp_acc),
                    "val_fusion_acc": float(val_fusion_acc),
                    "fusion_mode": fusion_mode,
                    "mean_graph_weight": float(graph_weight.mean()),
                    "mean_seq_weight": float((1.0 - graph_weight).mean()),
                    "mean_graph_conf": float(graph_conf.mean()),
                    "mean_seq_conf": float(seq_conf.mean()),
                }
            )
            torch.save(
                ioc_model.state_dict(), self.results_dir / f"ioc_model_fold{fold}.pt"
            )
            torch.save(
                ttp_model.state_dict(), self.results_dir / f"ttp_model_fold{fold}.pt"
            )
        self._print_final_results(fold_results)

    def _compute_fold_class_weights(
        self, train_y: np.ndarray, num_classes: int
    ) -> torch.Tensor:
        """Build class weights from train split only to avoid leakage."""
        weights = np.ones(num_classes, dtype=np.float32)
        train_classes = np.unique(train_y)
        balanced = compute_class_weight(
            class_weight="balanced", classes=train_classes, y=train_y
        )
        weights[train_classes] = balanced.astype(np.float32)
        return torch.tensor(weights, dtype=torch.float32).to(self.config.device)

    def _branch_confidence(self, probs: np.ndarray) -> np.ndarray:
        """Estimate branch confidence with normalized negative entropy."""
        safe_probs = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(safe_probs * np.log(safe_probs), axis=1)
        normalizer = np.log(max(probs.shape[1], 2))
        confidence = 1.0 - entropy / normalizer
        return np.clip(confidence, 0.0, 1.0)

    def _entropy_aware_fusion(
        self, graph_probs: np.ndarray, seq_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fuse posterior distributions using entropy-derived dynamic weights."""
        graph_conf = self._branch_confidence(graph_probs)
        seq_conf = self._branch_confidence(seq_probs)
        denom = graph_conf + seq_conf
        graph_weight = np.divide(
            graph_conf,
            denom,
            out=np.full_like(graph_conf, 0.5, dtype=np.float64),
            where=denom > 1e-12,
        )
        fused_probs = (
            graph_weight[:, None] * graph_probs
            + (1.0 - graph_weight[:, None]) * seq_probs
        )
        fused_probs = fused_probs / np.clip(
            fused_probs.sum(axis=1, keepdims=True), 1e-12, None
        )
        return fused_probs, graph_weight, graph_conf, seq_conf

    def _train_ioc_model(
        self,
        model: nn.Module,
        ioc_data: Any,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        class_weights: torch.Tensor,
        fold: int,
    ) -> nn.Module:
        cfg = self.ioc_cfg
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=cfg.scheduler_patience
        )
        criterion = FocalLoss(alpha=class_weights, gamma=3, label_smoothing=0.1)
        train_loader = NeighborLoader(
            ioc_data,
            num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            input_nodes=("EVENT", torch.tensor(train_idx, dtype=torch.long)),
            shuffle=True,
        )
        val_loader = NeighborLoader(
            ioc_data,
            num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            input_nodes=("EVENT", torch.tensor(val_idx, dtype=torch.long)),
            shuffle=False,
        )
        best_bacc, best_state, patience_counter = (
            -1.0,
            {k: v.cpu().clone() for k, v in model.state_dict().items()},
            0,
        )
        for epoch in range(cfg.max_epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(self.config.device)
                bs = batch["EVENT"].batch_size
                x_dict = {
                    nt: batch[nt].x
                    for nt in batch.node_types
                    if nt in ["IP", "domain", "URL", "File", "CVE", "ASN", "EVENT"]
                }
                edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
                edge_attr_dict = _build_edge_attr_dict(batch)
                num_nodes_dict = _build_num_nodes_dict(batch)
                optimizer.zero_grad()
                event_emb = model(
                    x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict
                )
                loss = criterion(
                    model.classifier(event_emb[:bs]), batch["EVENT"].y[:bs]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_preds, val_labels = [], []
                for batch in val_loader:
                    batch = batch.to(self.config.device)
                    bs = batch["EVENT"].batch_size
                    x_dict = {
                        nt: batch[nt].x
                        for nt in batch.node_types
                        if nt in ["IP", "domain", "URL", "File", "CVE", "ASN", "EVENT"]
                    }
                    edge_index_dict = {
                        et: batch[et].edge_index for et in batch.edge_types
                    }
                    edge_attr_dict = _build_edge_attr_dict(batch)
                    num_nodes_dict = _build_num_nodes_dict(batch)
                    event_emb = model(
                        x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict
                    )
                    val_preds.append(model.classifier(event_emb[:bs]).argmax(1).cpu())
                    val_labels.append(batch["EVENT"].y[:bs].cpu())
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

    def _train_ttp_model(
        self,
        model: nn.Module,
        ttp_sequences: list,
        phase_sequences,
        subseq_features,
        global_features,
        local_train_idx: np.ndarray,
        local_val_idx: np.ndarray,
        valid_y: np.ndarray,
        class_weights: torch.Tensor,
        fold: int,
        cfg_override: Optional[TTPModelConfig] = None,
    ) -> nn.Module:
        cfg = cfg_override or self.ttp_cfg
        device = self.config.device

        def collate_fn(batch):
            indices, labels = zip(*batch)
            return indices, labels

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_lr,
        )
        criterion = FocalLoss(alpha=class_weights, gamma=3, label_smoothing=0.1)
        train_dataset = list(
            zip(local_train_idx.tolist(), valid_y[local_train_idx].tolist())
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_seqs = [ttp_sequences[i] for i in local_val_idx]
        val_phases = (
            [phase_sequences[i] for i in local_val_idx] if phase_sequences else None
        )
        val_labels = torch.tensor(valid_y[local_val_idx], dtype=torch.long).to(device)
        val_padded = pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in val_seqs],
            batch_first=True,
            padding_value=0,
        ).to(device)
        val_mask = torch.zeros_like(val_padded)
        for i, seq in enumerate(val_seqs):
            val_mask[i, : len(seq)] = 1
        val_phase_padded = None
        if val_phases:
            val_phase_padded = pad_sequence(
                [torch.tensor(p, dtype=torch.long) for p in val_phases],
                batch_first=True,
                padding_value=0,
            ).to(device)
        idx_val = torch.tensor(local_val_idx)
        val_subseq = (
            subseq_features[idx_val].to(device) if subseq_features is not None else None
        )
        val_global = (
            global_features[idx_val].to(device) if global_features is not None else None
        )
        best_bacc, best_state, patience_counter = (
            -1.0,
            {k: v.cpu().clone() for k, v in model.state_dict().items()},
            0,
        )
        for epoch in range(cfg.max_epochs):
            model.train()
            total_loss = 0
            for batch_indices, batch_labels_raw in train_loader:
                idx_list = list(batch_indices)
                batch_seqs = [ttp_sequences[i] for i in idx_list]
                batch_phases = (
                    [phase_sequences[i] for i in idx_list] if phase_sequences else None
                )
                padded = pad_sequence(
                    [torch.tensor(s, dtype=torch.long) for s in batch_seqs],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                mask = torch.zeros_like(padded)
                for i, seq in enumerate(batch_seqs):
                    mask[i, : len(seq)] = 1
                padded_phases = None
                if batch_phases:
                    padded_phases = pad_sequence(
                        [torch.tensor(p, dtype=torch.long) for p in batch_phases],
                        batch_first=True,
                        padding_value=0,
                    ).to(device)
                idx_t = torch.tensor(idx_list)
                batch_subseq = (
                    subseq_features[idx_t].to(device)
                    if subseq_features is not None
                    else None
                )
                batch_global = (
                    global_features[idx_t].to(device)
                    if global_features is not None
                    else None
                )
                batch_labels_t = torch.tensor(
                    list(batch_labels_raw), dtype=torch.long
                ).to(device)
                logits = model(padded, padded_phases, batch_subseq, batch_global, mask)
                loss = criterion(logits, batch_labels_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            with torch.no_grad():
                val_logits = model(
                    val_padded, val_phase_padded, val_subseq, val_global, val_mask
                )
                val_bacc = balanced_accuracy_score(
                    val_labels.cpu().numpy(), val_logits.argmax(1).cpu().numpy()
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
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"      Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f}"
                    f" | Val B-Acc: {val_bacc:.4f} | LR: {lr:.6f}"
                )
        model.load_state_dict(best_state)
        return model

    def _evaluate_ioc_model(self, model, ioc_data, global_test_idx):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        test_loader = NeighborLoader(
            ioc_data,
            num_neighbors=self.config.num_neighbors,
            batch_size=self.config.batch_size,
            input_nodes=("EVENT", torch.tensor(global_test_idx, dtype=torch.long)),
            shuffle=False,
        )
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.config.device)
                bs = batch["EVENT"].batch_size
                x_dict = {
                    nt: batch[nt].x
                    for nt in batch.node_types
                    if nt in ["IP", "domain", "URL", "File", "CVE", "ASN", "EVENT"]
                }
                edge_index_dict = {et: batch[et].edge_index for et in batch.edge_types}
                edge_attr_dict = _build_edge_attr_dict(batch)
                num_nodes_dict = _build_num_nodes_dict(batch)
                event_emb = model(
                    x_dict, edge_index_dict, edge_attr_dict, num_nodes_dict
                )
                logits = model.classifier(event_emb[:bs])
                all_preds.append(logits.argmax(1).cpu())
                all_probs.append(F.softmax(logits, dim=1).cpu())
                all_labels.append(batch["EVENT"].y[:bs].cpu())
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        return accuracy_score(labels, preds), preds, labels, probs

    def _evaluate_ttp_model(
        self,
        model,
        ttp_sequences,
        phase_sequences,
        subseq_features,
        global_features,
        local_test_idx,
        valid_y,
    ):
        """Evaluate the TTP branch with local sample indices."""
        model.eval()
        device = self.config.device
        test_seqs = [ttp_sequences[i] for i in local_test_idx]
        test_phases = (
            [phase_sequences[i] for i in local_test_idx] if phase_sequences else None
        )
        test_y = valid_y[local_test_idx]
        test_padded = pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in test_seqs],
            batch_first=True,
            padding_value=0,
        ).to(device)
        test_mask = torch.zeros_like(test_padded)
        for i, seq in enumerate(test_seqs):
            test_mask[i, : len(seq)] = 1
        test_phase_padded = None
        if test_phases:
            test_phase_padded = pad_sequence(
                [torch.tensor(p, dtype=torch.long) for p in test_phases],
                batch_first=True,
                padding_value=0,
            ).to(device)
        idx_t = torch.tensor(local_test_idx)
        test_subseq = (
            subseq_features[idx_t].to(device) if subseq_features is not None else None
        )
        test_global = (
            global_features[idx_t].to(device) if global_features is not None else None
        )
        with torch.no_grad():
            logits = model(
                test_padded, test_phase_padded, test_subseq, test_global, test_mask
            )
            preds = logits.argmax(1).cpu().numpy()
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return accuracy_score(test_y, preds), preds, test_y, probs

    def _print_final_results(self, fold_results):
        print(f"\n{'='*70}")
        print("Dual decision fusion final report".center(70))
        print(f"{'='*70}")
        print(
            f"\n  {'Fold':<8} {'IOC Acc':<12} {'TTP Acc':<12} "
            f"{'Fusion Acc':<12} {'Fusion B-Acc':<12}"
        )
        print(f"  {'-'*60}")
        for r in fold_results:
            print(
                f"  {r['fold']:<8} {r['ioc_acc']:<12.4f} {r['ttp_acc']:<12.4f} "
                f"{r['or_acc']:<12.4f} {r['or_bacc']:<12.4f}"
            )
        or_accs = [r["or_acc"] for r in fold_results]
        or_baccs = [r["or_bacc"] for r in fold_results]
        or_f1s = [r["or_f1_macro"] for r in fold_results]
        oracle_accs = [r["oracle_acc"] for r in fold_results]
        mean_or_acc = float(np.mean(or_accs))
        std_or_acc = float(np.std(or_accs))
        mean_oracle_acc = float(np.mean(oracle_accs))
        print("\n  Means:")
        print(f"    Fusion Acc:   {mean_or_acc:.4f} +/- {std_or_acc:.4f}")
        print(f"    Fusion B-Acc: {np.mean(or_baccs):.4f} +/- {np.std(or_baccs):.4f}")
        print(f"    Fusion F1:    {np.mean(or_f1s):.4f} +/- {np.std(or_f1s):.4f}")
        print(f"    Oracle Acc: {mean_oracle_acc:.4f}")
        if mean_oracle_acc < 0.80:
            print("    [Gate] Current branches cannot reach 80% oracle accuracy.")
        elif mean_or_acc < 0.80:
            print(
                "    [Gate] Fusion is below 80% although oracle accuracy is reachable."
            )
        elif std_or_acc > 0.03:
            print("    [Gate] Accuracy variance is still high.")
        else:
            print("    [Gate] Target reached.")
        print(f"{'='*70}")
        with open(self.results_dir / "final_results.json", "w") as f:
            json.dump(
                {
                    "folds": fold_results,
                    "summary": {
                        "mean_or_acc": mean_or_acc,
                        "std_or_acc": std_or_acc,
                        "mean_oracle_acc": mean_oracle_acc,
                        "accept_acc_80": mean_or_acc >= 0.80,
                        "accept_std_003": std_or_acc <= 0.03,
                        "oracle_reachable": mean_oracle_acc >= 0.80,
                        "fusion": "entropy_aware",
                    },
                },
                f,
                indent=2,
            )


def main():
    parser = argparse.ArgumentParser(description="Dual decision fusion training")
    parser.add_argument("--ioc-data", type=str, default="./apt_kg_ioc.pt")
    parser.add_argument("--ttp-data", type=str, default="./apt_kg_ttp.pt")
    parser.add_argument("--top-k-classes", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    config = EmbeddingGraphConfig(
        ioc_data_path=args.ioc_data,
        ttp_data_path=args.ttp_data,
        top_k_classes=args.top_k_classes,
        batch_size=args.batch_size,
    )
    if args.device:
        config.device = torch.device(args.device)
    ioc_cfg = IOCModelConfig()
    ttp_cfg = TTPModelConfig()
    _set_global_seed(config.seed)
    print("=" * 70)
    print("Dual decision fusion training (Transformer TTP)".center(70))
    print(f"Device: {config.device}".center(70))
    print("Fusion: entropy-aware decision fusion".center(70))
    print("=" * 70)
    print(f"\n[Data] Load IOC graph: {config.ioc_data_path}")
    ioc_data = torch.load(config.ioc_data_path, weights_only=False)
    print(f"[Data] Load TTP sequences: {config.ttp_data_path}")
    ttp_data = torch.load(config.ttp_data_path, weights_only=False)
    sequences = ttp_data.get("causal_sequences", ttp_data.get("technique_sequences"))
    if sequences is None:
        raise KeyError(
            "TTP data must contain 'causal_sequences' or 'technique_sequences'."
        )
    seq_type = "causal" if "causal_sequences" in ttp_data else "raw"
    print(
        f"  Sequence type: {seq_type}, sequences: {len(sequences)}, "
        f"techniques: {ttp_data.get('num_techniques', 369)}"
    )
    total_events = len(ioc_data["EVENT"].y)
    valid_event_count = int((ioc_data["EVENT"].y != -1).sum().item())
    original_valid_idx = torch.where(ioc_data["EVENT"].y != -1)[0].numpy()
    if len(sequences) in {total_events, valid_event_count}:
        print(f"  [OK] Sample count is usable: {len(sequences)}")
    else:
        raise ValueError(
            f"TTP sample count mismatch: got {len(sequences)}, expected "
            f"{valid_event_count} valid events or {total_events} total events."
        )
    if "labels" in ttp_data:
        ttp_labels = ttp_data["labels"]
        label_count = len(ttp_labels)
        if label_count == total_events:
            ioc_label_idx = np.arange(min(5, total_events))
            ttp_label_idx = ioc_label_idx
        elif label_count == valid_event_count:
            sample_size = min(5, valid_event_count)
            ioc_label_idx = original_valid_idx[:sample_size]
            ttp_label_idx = np.arange(sample_size)
        else:
            raise ValueError(
                f"TTP label count mismatch: got {label_count}, expected "
                f"{valid_event_count} valid labels or {total_events} total labels."
            )
        labels_match = sum(
            ioc_data["EVENT"].y[int(ioc_i)].item()
            == (
                ttp_labels[int(ttp_i)].item()
                if isinstance(ttp_labels, torch.Tensor)
                else ttp_labels[int(ttp_i)]
            )
            for ioc_i, ttp_i in zip(ioc_label_idx, ttp_label_idx)
        )
        sample_size = len(ioc_label_idx)
        status = "[OK]" if labels_match == sample_size else "[ERROR]"
        print(f"  {status} Sampled labels match: {labels_match}/{sample_size}")
        if labels_match != sample_size:
            raise ValueError("Sampled IOC/TTP labels are not aligned.")
    original_classes = len(
        np.unique(ioc_data["EVENT"].y[ioc_data["EVENT"].y != -1].numpy())
    )
    if config.top_k_classes < original_classes:
        print(f"[Filter] Keep top-{config.top_k_classes} classes.")
        ioc_data = filter_top_k_classes(ioc_data, config.top_k_classes)
        valid_idx = torch.where(ioc_data["EVENT"].y != -1)[0].numpy()
        raw_seqs = ttp_data.get("causal_sequences", ttp_data.get("technique_sequences"))
        seq_key = (
            "causal_sequences"
            if "causal_sequences" in ttp_data
            else "technique_sequences"
        )
        raw_seqs = _align_optional_field(
            raw_seqs, original_valid_idx, total_events, seq_key
        )
        original_valid_pos = {
            int(global_idx): pos for pos, global_idx in enumerate(original_valid_idx)
        }
        keep_positions = np.array(
            [original_valid_pos[int(global_idx)] for global_idx in valid_idx],
            dtype=np.int64,
        )
        ttp_data_filtered = {
            seq_key: [raw_seqs[i] for i in keep_positions],
            "num_techniques": ttp_data.get("num_techniques", 369),
        }
        if "labels" in ttp_data:
            ttp_data_filtered["labels"] = ioc_data["EVENT"].y[valid_idx].clone()
        for key in [
            "phase_sequences",
            "global_features",
            "subseq_features",
            "technique_embeddings",
            "num_phases",
            "global_feature_dim",
            "semantic_dim",
            "num_events",
            "num_classes",
            "apt_classes",
            "padding_value",
            "seq_stats",
            "tactic_mapping",
            "tactic_phase_order",
            "sequence_type",
        ]:
            if key not in ttp_data:
                continue
            value = ttp_data[key]
            if key in ["phase_sequences", "global_features", "subseq_features"]:
                value = _align_optional_field(
                    value, original_valid_idx, total_events, key
                )
                ttp_data_filtered[key] = (
                    value[torch.tensor(keep_positions, dtype=torch.long)]
                    if isinstance(value, torch.Tensor)
                    else [value[i] for i in keep_positions]
                )
            else:
                ttp_data_filtered[key] = value
        ttp_data = ttp_data_filtered
    else:
        valid_idx = torch.where(ioc_data["EVENT"].y != -1)[0].numpy()
    for seq_key in ["causal_sequences", "technique_sequences"]:
        if seq_key in ttp_data:
            ttp_data[seq_key] = _align_optional_field(
                ttp_data[seq_key], valid_idx, total_events, seq_key
            )
    for field_name in [
        "labels",
        "phase_sequences",
        "subseq_features",
        "global_features",
    ]:
        if field_name in ttp_data:
            ttp_data[field_name] = _align_optional_field(
                ttp_data[field_name], valid_idx, total_events, field_name
            )
    y_valid = ioc_data["EVENT"].y[valid_idx].numpy()
    print(f"[Data] classes={config.top_k_classes}, samples={len(y_valid)}")
    trainer = DualDecisionFusionTrainer(config, ioc_cfg, ttp_cfg)
    trainer.train_and_evaluate(ioc_data, ttp_data, valid_idx)
    print(f"\n[Done] Results saved to: {trainer.results_dir}")


if __name__ == "__main__":
    main()
