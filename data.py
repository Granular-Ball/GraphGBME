from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import dgl
import numpy as np
import torch
from scipy import sparse
from scipy.io import loadmat


@dataclass
class DatasetBundle:
    graph: dgl.DGLGraph
    features: torch.Tensor
    labels: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    minority_class: int
    minority_mean: torch.Tensor
    minority_std: torch.Tensor
    supervision_metadata: dict | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_graph(zip_path: Path, cache_dir: Path, member: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / member
    if not output.exists() or output.stat().st_size == 0:
        with zipfile.ZipFile(zip_path) as archive:
            if member not in archive.namelist():
                raise ValueError(f"{zip_path} 中没有 {member!r}")
            archive.extract(member, cache_dir)
    return output


def stratified_split(
    labels: torch.Tensor, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic per-class 7:1:2 split."""
    generator = torch.Generator().manual_seed(seed)
    train, val, test = [], [], []
    for class_id in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        n_train = int(indices.numel() * 0.7)
        n_val = int(indices.numel() * 0.1)
        train.append(indices[:n_train])
        val.append(indices[n_train : n_train + n_val])
        test.append(indices[n_train + n_val :])

    def shuffled(parts: list[torch.Tensor]) -> torch.Tensor:
        result = torch.cat(parts)
        return result[torch.randperm(result.numel(), generator=generator)]

    return shuffled(train), shuffled(val), shuffled(test)


def apply_supervised_minority_ratio(
    data: DatasetBundle,
    target_ratio: float | str,
    seed: int,
    eps: float = 1e-6,
) -> dict:
    """Reduce minority *supervision* without removing graph nodes or edges.

    Majority training nodes are all retained.  A deterministic prefix of a
    seed-specific minority permutation is retained as labelled training nodes;
    the remainder stays in the graph and can still be sampled as message-passing
    context.  Feature normalization is refit using only retained labelled
    minority nodes so hidden labels cannot influence preprocessing.
    """
    original_train_idx = data.train_idx.long().cpu()
    labels = data.labels.long().cpu()
    minority_mask = labels[original_train_idx] == data.minority_class
    minority_nodes = original_train_idx[minority_mask]
    majority_nodes = original_train_idx[~minority_mask]
    original_ratio = minority_nodes.numel() / original_train_idx.numel()

    if target_ratio == "original":
        requested_ratio = original_ratio
        retained_minority = minority_nodes
    else:
        requested_ratio = float(target_ratio)
        if not 0.0 < requested_ratio < 1.0:
            raise ValueError("supervised minority ratio must be in (0, 1) or 'original'")
        if requested_ratio > original_ratio + 1e-12:
            raise ValueError(
                "supervised minority ratio can only reduce the original ratio "
                f"({original_ratio:.8f}), got {requested_ratio:.8f}"
            )
        retain_count = int(
            requested_ratio / (1.0 - requested_ratio) * majority_nodes.numel()
        )
        retain_count = max(1, min(retain_count, minority_nodes.numel()))
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(minority_nodes.numel(), generator=generator)
        retained_minority = minority_nodes[permutation[:retain_count]]

    # Recover pre-normalized features, then fit the scaler using retained labels.
    raw_features = data.features * data.minority_std + data.minority_mean
    new_mean = raw_features[retained_minority].mean(dim=0)
    new_std = raw_features[retained_minority].std(dim=0, unbiased=False).clamp_min(eps)
    data.features = (raw_features - new_mean) / new_std
    data.minority_mean = new_mean
    data.minority_std = new_std

    supervised = torch.cat((majority_nodes, retained_minority))
    generator = torch.Generator().manual_seed(seed + 1)
    data.train_idx = supervised[torch.randperm(supervised.numel(), generator=generator)]
    retained_ratio = retained_minority.numel() / data.train_idx.numel()
    metadata = {
        "strategy": "minority_label_subsampling_graph_unchanged",
        "requested_ratio": target_ratio,
        "original_ratio": original_ratio,
        "actual_ratio": retained_ratio,
        "original_train_nodes": int(original_train_idx.numel()),
        "supervised_train_nodes": int(data.train_idx.numel()),
        "majority_train_nodes": int(majority_nodes.numel()),
        "original_minority_train_nodes": int(minority_nodes.numel()),
        "retained_minority_train_nodes": int(retained_minority.numel()),
        "context_only_minority_nodes": int(minority_nodes.numel() - retained_minority.numel()),
        "graph_nodes": int(data.graph.num_nodes()),
        "graph_edges": int(data.graph.num_edges()),
        "graph_structure_changed": False,
        "seed": int(seed),
    }
    data.supervision_metadata = metadata
    return metadata


def load_dataset(
    zip_path: str | Path,
    cache_dir: str | Path,
    member: str,
    seed: int = 42,
    eps: float = 1e-6,
) -> DatasetBundle:
    graph_file = _extract_graph(Path(zip_path), Path(cache_dir), member)
    graph = dgl.load_graphs(str(graph_file))[0][0]
    labels_raw = graph.ndata["label"]
    labels = labels_raw.argmax(dim=1) if labels_raw.ndim == 2 else labels_raw.long()
    labels = labels.long().cpu()
    features = graph.ndata["feature"].float().cpu()
    train_idx, val_idx, test_idx = stratified_split(labels, seed)

    # Determine the minority class from training labels only.  Validation and
    # test labels cannot influence granular-ball construction or preprocessing.
    classes, counts = torch.unique(labels[train_idx], return_counts=True)
    minority_class = int(classes[counts.argmin()].item())
    minority_train = train_idx[labels[train_idx] == minority_class]
    # Only minority training nodes fit the scaler. The same transformation is then
    # applied to all nodes, so validation/test labels never influence preprocessing.
    mean = features[minority_train].mean(dim=0)
    std = features[minority_train].std(dim=0, unbiased=False).clamp_min(eps)
    features = (features - mean) / std

    graph.ndata.pop("feature")
    graph.ndata.pop("label")
    return DatasetBundle(
        graph=graph,
        features=features,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        minority_class=minority_class,
        minority_mean=mean,
        minority_std=std,
    )


def load_mat_dataset(
    mat_path: str | Path,
    cache_dir: str | Path,
    seed: int = 42,
    eps: float = 1e-6,
) -> DatasetBundle:
    """Load a fraud-detection ``.mat`` file with homo/features/label fields."""
    # Kept in the shared loader signature for interchangeability with zip loaders.
    del cache_dir
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {mat_path}")

    values = loadmat(mat_path)
    required = ("homo", "features", "label")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"{mat_path} is missing required fields: {missing}")

    adjacency = values["homo"]
    if not sparse.issparse(adjacency):
        adjacency = sparse.coo_matrix(adjacency)
    adjacency = adjacency.tocoo(copy=False)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()

    raw_features = values["features"]
    if sparse.issparse(raw_features):
        raw_features = raw_features.toarray()
    features = torch.as_tensor(np.asarray(raw_features), dtype=torch.float32)
    labels = torch.as_tensor(
        np.asarray(values["label"]).reshape(-1), dtype=torch.long
    )
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"homo adjacency must be square, got {adjacency.shape}")
    if features.ndim != 2:
        raise ValueError(f"features must be a matrix, got shape {features.shape}")
    if adjacency.shape[0] != labels.numel() or features.shape[0] != labels.numel():
        raise ValueError(
            "homo, features, and label must contain the same number of nodes"
        )
    classes = torch.unique(labels, sorted=True)
    if classes.numel() != 2:
        raise ValueError(f"binary labels are required, got {classes.tolist()}")

    graph = dgl.from_scipy(adjacency)
    train_idx, val_idx, test_idx = stratified_split(labels, seed)
    train_classes, counts = torch.unique(labels[train_idx], return_counts=True)
    minority_class = int(train_classes[counts.argmin()].item())
    minority_train = train_idx[labels[train_idx] == minority_class]
    mean = features[minority_train].mean(dim=0)
    std = features[minority_train].std(dim=0, unbiased=False).clamp_min(eps)
    features = (features - mean) / std

    return DatasetBundle(
        graph=graph,
        features=features,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        minority_class=minority_class,
        minority_mean=mean,
        minority_std=std,
    )


def load_tfinance(
    zip_path: str | Path,
    cache_dir: str | Path,
    seed: int = 42,
    eps: float = 1e-6,
) -> DatasetBundle:
    return load_dataset(zip_path, cache_dir, "tfinance", seed, eps)


def load_tsocial(
    zip_path: str | Path,
    cache_dir: str | Path,
    seed: int = 42,
    eps: float = 1e-6,
) -> DatasetBundle:
    return load_dataset(zip_path, cache_dir, "tsocial", seed, eps)


def load_amazon(
    mat_path: str | Path,
    cache_dir: str | Path,
    seed: int = 42,
    eps: float = 1e-6,
) -> DatasetBundle:
    return load_mat_dataset(mat_path, cache_dir, seed, eps)
