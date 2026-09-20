from __future__ import annotations

import argparse
import json
import tempfile
import time
import warnings
from pathlib import Path

import dgl
import numpy as np
import torch
from dgl.base import DGLWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from torch import nn


warnings.filterwarnings(
    "ignore",
    message=r"Dataloader CPU affinity opt is not enabled.*",
    category=DGLWarning,
)


REQUIRED_TEST_OUTPUT_FIELDS = (
    "accuracy",
    "minority_recall",
    "minority_precision",
    "minority_f1",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "g_mean",
    "threshold",
    "tn",
    "fp",
    "fn",
    "tp",
    "test_inference_time_per_sample_ms",
)


def validate_test_output_fields(result: dict) -> None:
    """Enforce the shared test-output contract used by all training scripts."""
    missing = [key for key in REQUIRED_TEST_OUTPUT_FIELDS if key not in result]
    if missing:
        raise RuntimeError(f"test result is missing required fields: {missing}")


def parse_minority_ratio(value: str) -> float | str:
    """Accept a numeric target ratio or the training split's original ratio."""
    if value.lower() == "original":
        return "original"
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "minority ratio must be a number in [0, 1] or 'original'"
        ) from exc
    if not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("minority ratio must be in [0, 1]")
    return ratio

try:
    from .ball_balanced_seed_sampler import BallBalancedSeedSampler
    from .data import (
        DatasetBundle,
        apply_supervised_minority_ratio,
        load_amazon,
        load_tfinance,
        load_tsocial,
        seed_everything,
    )
    from .granular_ball import GranularBallBuilder, build_two_hop_minority_hypergraph
    from .model import BallGraphSAGE
except ImportError:  # Support the current flat project layout via ``python train.py``.
    from ball_balanced_seed_sampler import BallBalancedSeedSampler
    from data import (
        DatasetBundle,
        apply_supervised_minority_ratio,
        load_amazon,
        load_tfinance,
        load_tsocial,
        seed_everything,
    )
    from granular_ball import GranularBallBuilder, build_two_hop_minority_hypergraph
    from model import BallGraphSAGE


DATASET_CONFIGS = {
    "tfinance": {
        "data": Path("datasets/T-Finance/tfinance.zip"),
        "output_dir": Path("outputs/tfinance"),
        "loader": load_tfinance,
    },
    "tsocial": {
        "data": Path("datasets/T-Social/tsocial.zip"),
        "output_dir": Path("outputs/tsocial"),
        "loader": load_tsocial,
    },
    "amazon": {
        "data": Path("datasets/Amazon/Amazon.mat"),
        "output_dir": Path("outputs/amazon"),
        "loader": load_amazon,
    },
}


def parse_dataset_name(value: str) -> str:
    """Accept human-readable and compact dataset spellings."""
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    if normalized not in DATASET_CONFIGS:
        choices = ", ".join(DATASET_CONFIGS)
        raise argparse.ArgumentTypeError(f"dataset must be one of: {choices}")
    return normalized


def resolve_dataset_config(args: argparse.Namespace) -> tuple[str, Path, Path, object]:
    """Resolve dataset-dependent defaults while preserving explicit path overrides."""
    dataset = getattr(args, "dataset", "tfinance")
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"unsupported dataset: {dataset}")
    config = DATASET_CONFIGS[dataset]
    data_path = Path(args.data) if getattr(args, "data", None) else config["data"]
    output_dir = (
        Path(args.output_dir)
        if getattr(args, "output_dir", None)
        else config["output_dir"]
    )
    return dataset, data_path, output_dir, config["loader"]


def collect_minority_incident_edges(
    graph: dgl.DGLGraph,
    minority_ids: torch.Tensor,
    chunk_size: int = 100_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect exactly the edges needed for minority paths of length at most two.

    Any such path has an edge incident to a minority endpoint, so unrelated graph
    edges can be omitted. This avoids materializing every edge of T-Social.
    """
    src_chunks: list[torch.Tensor] = []
    dst_chunks: list[torch.Tensor] = []
    minority_ids = minority_ids.detach().cpu().long().flatten()
    for start in range(0, minority_ids.numel(), chunk_size):
        nodes = minority_ids[start : start + chunk_size]
        in_src, in_dst = graph.in_edges(nodes)
        out_src, out_dst = graph.out_edges(nodes)
        src_chunks.extend((in_src.cpu(), out_src.cpu()))
        dst_chunks.extend((in_dst.cpu(), out_dst.cpu()))
    if not src_chunks:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    return torch.cat(src_chunks), torch.cat(dst_chunks)


def build_training_balls(
    data: DatasetBundle,
    split_quality_threshold: float,
    min_split_size: int,
) -> tuple:
    minority_ids = data.train_idx[data.labels[data.train_idx] == data.minority_class]
    print(
        f"构建两跳少数类候选图：训练少数类节点={minority_ids.numel()}",
        flush=True,
    )
    src, dst = collect_minority_incident_edges(data.graph, minority_ids)
    adjacency = build_two_hop_minority_hypergraph(
        (src, dst), data.graph.num_nodes(), minority_ids
    )
    del src, dst
    print(
        "候选图隐式构建完成："
        f"超边组={adjacency.edge_offsets.numel() - 1}，"
        f"端点关联={adjacency.endpoints.numel()}；开始粒球划分",
        flush=True,
    )
    local_features = data.features[minority_ids]
    builder = GranularBallBuilder(
        quality_threshold=split_quality_threshold,
        min_split_size=min_split_size,
    )
    balls = builder.build(local_features, adjacency)
    centered = [ball.center for ball in balls if ball.center is not None]
    centers = (
        torch.stack(centered)
        if centered
        else torch.empty((0, local_features.shape[1]), dtype=local_features.dtype)
    )
    return balls, centers, minority_ids


def summarize_ball_partition(balls) -> dict:
    """Return JSON-ready aggregate granular-ball partition statistics."""
    sizes = np.asarray([int(ball.member_ids.numel()) for ball in balls], dtype=np.int64)
    qualities = np.asarray(
        [
            1.0 if ball.member_ids.numel() == 1 else float(ball.quality)
            for ball in balls
            if ball.quality is not None
        ],
        dtype=np.float64,
    )
    if not sizes.size:
        values = {"minimum": 0, "mean": 0.0, "median": 0.0, "maximum": 0}
    else:
        values = {
            "minimum": int(sizes.min()),
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "maximum": int(sizes.max()),
        }

    size_statistics = {}
    for name, value in values.items():
        count = int(np.count_nonzero(np.isclose(sizes, value, rtol=0.0, atol=1e-12)))
        size_statistics[name] = {
            "value": value,
            "ball_count": count,
            "ball_percentage": 100.0 * count / sizes.size if sizes.size else 0.0,
        }
    return {
        "num_balls": int(sizes.size),
        "covered_nodes": int(sizes.sum()),
        "size_statistics": size_statistics,
        "average_quality": float(qualities.mean()) if qualities.size else 0.0,
    }


def print_ball_summary(balls) -> None:
    """Print aggregate statistics immediately after granular-ball partitioning."""
    summary = summarize_ball_partition(balls)
    stats = summary["size_statistics"]
    print("粒球划分完成：")
    print(f"  粒球总数：{summary['num_balls']}")
    print(f"  覆盖节点数：{summary['covered_nodes']}")
    print(
        "  粒球规模（最小值/均值/中位数/最大值）："
        f"{stats['minimum']['value']}/{stats['mean']['value']:.2f}/"
        f"{stats['median']['value']:.2f}/{stats['maximum']['value']}"
    )
    for label, key in (
        ("最小值", "minimum"),
        ("均值", "mean"),
        ("中位数", "median"),
        ("最大值", "maximum"),
    ):
        item = stats[key]
        print(
            f"  规模等于{label}的粒球：{item['ball_count']} "
            f"({item['ball_percentage']:.2f}%)"
        )
    print(f"  平均质量：{summary['average_quality']:.4f}")


def make_loader(
    graph: dgl.DGLGraph,
    indices: torch.Tensor,
    fanouts: list[int],
    batch_size: int,
    shuffle: bool,
) -> dgl.dataloading.DataLoader:
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    return dgl.dataloading.DataLoader(
        graph,
        indices,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: dgl.dataloading.DataLoader,
    features: torch.Tensor,
    device: torch.device,
    positive_class: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    node_chunks, probability_chunks = [], []
    for input_nodes, output_nodes, blocks in loader:
        blocks = [block.to(device) for block in blocks]
        logits = model(blocks, features)
        node_chunks.append(output_nodes.cpu())
        probability_chunks.append(logits.softmax(dim=1)[:, positive_class].cpu())
    return torch.cat(node_chunks), torch.cat(probability_chunks)


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, np.linspace(0.01, 0.99, 199), probability, 1.0])
    scores = [f1_score(y_true, probability >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def analyze_validation_thresholds(
    y_true: np.ndarray,
    probability: np.ndarray,
    target_precision: float = 0.90,
    target_recall: float = 0.85,
) -> dict:
    """Check whether a validation PR-curve operating point meets both targets."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    # The final precision/recall pair has no associated threshold.
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    feasible = (precision >= target_precision) & (recall >= target_recall)
    deficit = (
        np.maximum(0.0, target_precision - precision) ** 2
        + np.maximum(0.0, target_recall - recall) ** 2
    )

    def point(index: int) -> dict:
        return {
            "threshold": float(thresholds[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }

    closest_candidates = np.flatnonzero(deficit == deficit.min())
    closest_index = int(closest_candidates[np.argmax(f1[closest_candidates])])
    feasible_indices = np.flatnonzero(feasible)
    best_feasible = None
    if feasible_indices.size:
        best_index = int(feasible_indices[np.argmax(f1[feasible_indices])])
        best_feasible = point(best_index)

    recall_mask = recall >= target_recall
    precision_mask = precision >= target_precision
    best_precision_at_target_recall = None
    if recall_mask.any():
        candidates = np.flatnonzero(recall_mask)
        index = int(candidates[np.argmax(precision[candidates])])
        best_precision_at_target_recall = point(index)
    best_recall_at_target_precision = None
    if precision_mask.any():
        candidates = np.flatnonzero(precision_mask)
        index = int(candidates[np.argmax(recall[candidates])])
        best_recall_at_target_precision = point(index)

    return {
        "target_precision": float(target_precision),
        "target_recall": float(target_recall),
        "target_is_feasible": bool(feasible_indices.size),
        "num_feasible_thresholds": int(feasible_indices.size),
        "best_feasible_point": best_feasible,
        "closest_point": point(closest_index),
        "best_precision_at_target_recall": best_precision_at_target_recall,
        "best_recall_at_target_precision": best_recall_at_target_precision,
    }


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    minority_recall = recall_score(y_true, prediction, zero_division=0)
    majority_recall = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "minority_recall": float(minority_recall),
        "minority_precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "minority_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, prediction, average="macro", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "g_mean": float(np.sqrt(minority_recall * majority_recall)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def classification_loss_on_seeds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    seed_nodes: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute supervision only for the destination seed nodes in a batch."""
    if logits.shape[0] != seed_nodes.numel():
        raise ValueError("logits must contain exactly one row per seed node")
    return criterion(logits, labels.index_select(0, seed_nodes))


def dgl_cuda_is_available(device: torch.device | str = "cuda") -> bool:
    """Check CUDA support in both PyTorch and the installed DGL build.

    ``torch.cuda.is_available()`` alone is insufficient: a CUDA-enabled PyTorch
    installation can be used together with a CPU-only DGL wheel.
    """
    if not torch.cuda.is_available():
        return False
    try:
        empty = torch.empty(0, dtype=torch.int64)
        dgl.graph((empty, empty), num_nodes=1).to(torch.device(device))
    except (dgl.DGLError, RuntimeError):
        return False
    return True


def resolve_device(requested: str | None) -> torch.device:
    """Resolve ``auto`` safely, or validate an explicitly requested device."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if dgl_cuda_is_available():
            return torch.device("cuda")
        if torch.cuda.is_available():
            print("CUDA is available in PyTorch but not in DGL; using CPU.")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not dgl_cuda_is_available(device):
        raise RuntimeError(
            f"Cannot use {device}: CUDA must be available in both PyTorch and DGL. "
            "Use --device cpu or install a CUDA-enabled DGL build compatible "
            "with the installed PyTorch/CUDA version."
        )
    return device


def synchronize_device(device: torch.device) -> None:
    """Wait for asynchronous CUDA work so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args: argparse.Namespace) -> dict:
    if args.min_ball_size < 1:
        raise ValueError("min_ball_size must be at least 1")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset, data_path, output_dir, dataset_loader = resolve_dataset_config(args)
    print(f"device={device}")
    print(f"dataset={dataset} data={data_path} output_dir={output_dir}")
    data = dataset_loader(data_path, args.cache_dir, args.seed)
    supervision_metadata = apply_supervised_minority_ratio(
        data, getattr(args, "supervised_minority_ratio", "original"), args.seed
    )
    print(
        "supervised minority labels: "
        + json.dumps(supervision_metadata, ensure_ascii=False),
        flush=True,
    )
    granular_ball_construction_started = time.perf_counter()
    balls, centers, minority_ids = build_training_balls(
        data,
        args.quality_threshold,
        args.min_split_size,
    )
    granular_ball_construction_time = (
        time.perf_counter() - granular_ball_construction_started
    )
    print_ball_summary(balls)
    print(
        "粒球构建时间："
        f"{granular_ball_construction_time:.6f}s",
        flush=True,
    )
    features = data.features
    features_device = features.to(device)
    labels_device = data.labels.to(device)
    original_minority_ratio = float(
        (data.labels[data.train_idx] == data.minority_class).float().mean()
    )
    minority_ratio = (
        original_minority_ratio
        if args.minority_ratio == "original"
        else float(args.minority_ratio)
    )
    print(
        f"training minority ratio: target={minority_ratio:.8f} "
        f"original={original_minority_ratio:.8f}"
    )

    enhancement_mode = getattr(args, "enhancement_mode", "granular_ball")
    prototype_balls = [
        ball for ball in balls if ball.center is not None and ball.quality is not None
    ]
    sampling_ball_members = [minority_ids[ball.member_ids] for ball in prototype_balls]
    granular_ball_centers = (
        torch.stack([ball.center for ball in prototype_balls])
        if prototype_balls
        else features.new_empty((0, features.shape[1]))
    )
    granular_ball_quality = torch.tensor(
        [ball.quality for ball in prototype_balls], dtype=features.dtype
    )
    reliable_ball_count = sum(
        members.numel() >= args.min_ball_size for members in sampling_ball_members
    )

    if enhancement_mode == "granular_ball":
        enhancement_members = sampling_ball_members
        enhancement_centers = granular_ball_centers
        enhancement_quality = granular_ball_quality
        enhancement_min_size = args.min_ball_size
        enhancement_quality_threshold = 0.0
        allow_singleton_prototypes = args.min_ball_size == 1
        num_enhancement_candidates = reliable_ball_count
        matching_score = "granular_ball_quality * exp(-euclidean_distance)"
    elif enhancement_mode == "minority_node":
        # Point enhancement control: every supervised minority training node is
        # a one-member prototype.  Unit quality removes the granular-ball
        # quality factor, leaving exp(-distance) as the matching score.
        enhancement_members = [node.reshape(1) for node in minority_ids]
        enhancement_centers = features.index_select(0, minority_ids)
        enhancement_quality = features.new_ones(minority_ids.numel())
        enhancement_min_size = 1
        enhancement_quality_threshold = 0.0
        allow_singleton_prototypes = True
        num_enhancement_candidates = int(minority_ids.numel())
        matching_score = "exp(-euclidean_distance)"
    else:
        raise ValueError(f"unsupported enhancement_mode: {enhancement_mode}")
    print(
        f"enhancement_mode={enhancement_mode} "
        f"candidates={num_enhancement_candidates} "
        f"top_k={args.prototype_top_k} score={matching_score}",
        flush=True,
    )

    train_seed_sampler = BallBalancedSeedSampler(
        train_idx=data.train_idx,
        y=data.labels,
        ball_members=sampling_ball_members,
        seed_batch_size=args.batch_size,
        minority_ratio=minority_ratio,
        random_seed=args.seed,
    )
    train_neighbor_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanouts)
    val_loader = make_loader(data.graph, data.val_idx, args.eval_fanouts, args.batch_size, False)
    model = BallGraphSAGE(
        in_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        ball_members=enhancement_members,
        ball_centers=enhancement_centers,
        ball_quality=enhancement_quality,
        min_ball_size=enhancement_min_size,
        quality_threshold=enhancement_quality_threshold,
        prototype_top_k=args.prototype_top_k,
        prototype_layer_norm=args.prototype_layer_norm,
        allow_singleton_prototypes=allow_singleton_prototypes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_labels = data.labels[data.train_idx]
    counts = torch.bincount(train_labels, minlength=2).float()
    if args.class_weight_gamma < 0.0:
        raise ValueError("class_weight_gamma must be non-negative")
    base_class_weight = counts.sum() / (2.0 * counts.clamp_min(1))
    class_weight = base_class_weight.pow(args.class_weight_gamma).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    best_state, best_val_pr, patience_left = None, -1.0, args.patience
    training_times: list[float] = []

    for epoch in range(1, args.epochs + 1):
        train_seed_sampler.set_epoch(epoch - 1)
        synchronize_device(device)
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        epoch_seed_count = 0
        training_started = time.perf_counter()
        for seed_nodes in train_seed_sampler:
            input_nodes, output_nodes, blocks = train_neighbor_sampler.sample_blocks(
                data.graph, seed_nodes
            )
            blocks = [block.to(device) for block in blocks]
            logits = model(blocks, features_device)
            output_nodes_device = output_nodes.to(device)
            loss = classification_loss_on_seeds(
                logits, labels_device, output_nodes_device, criterion
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * output_nodes.numel()
            epoch_seed_count += output_nodes.numel()
        synchronize_device(device)
        training_time = time.perf_counter() - training_started
        training_times.append(training_time)

        val_nodes, val_probability = predict(
            model,
            val_loader,
            features_device,
            device,
            positive_class=data.minority_class,
        )
        synchronize_device(device)
        epoch_time = time.perf_counter() - epoch_started
        val_y = (data.labels[val_nodes] == data.minority_class).numpy().astype(np.int64)
        val_pr = average_precision_score(val_y, val_probability.numpy())
        print(
            f"epoch={epoch:03d} "
            f"loss={total_loss / epoch_seed_count:.6f} "
            f"val_pr_auc={val_pr:.6f} "
            f"train_time={training_time:.3f}s "
            f"epoch_time={epoch_time:.3f}s"
        )
        if val_pr > best_val_pr + 1e-6:
            best_val_pr = float(val_pr)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is None:
        raise RuntimeError("训练没有产生有效模型")
    model.load_state_dict(best_state)
    val_nodes, val_probability = predict(
        model,
        val_loader,
        features_device,
        device,
        positive_class=data.minority_class,
    )
    val_y = (data.labels[val_nodes] == data.minority_class).numpy().astype(np.int64)
    validation_threshold_analysis = analyze_validation_thresholds(
        val_y, val_probability.numpy()
    )
    threshold = best_f1_threshold(val_y, val_probability.numpy())
    test_loader = make_loader(data.graph, data.test_idx, args.eval_fanouts, args.batch_size, False)
    synchronize_device(device)
    inference_started = time.perf_counter()
    test_nodes, test_probability = predict(
        model,
        test_loader,
        features_device,
        device,
        positive_class=data.minority_class,
    )
    synchronize_device(device)
    test_inference_time = time.perf_counter() - inference_started
    inference_time_per_sample = test_inference_time / test_nodes.numel()
    test_y = (data.labels[test_nodes] == data.minority_class).numpy().astype(np.int64)
    result = metrics(test_y, test_probability.numpy(), threshold)
    result.update(
        {
            "num_balls": len(balls),
            "num_reliable_prototype_balls": reliable_ball_count,
            "prototype_enhancement": {
                "enabled": True,
                "mode": enhancement_mode,
                "candidate_type": (
                    "granular_ball" if enhancement_mode == "granular_ball"
                    else "minority_training_node"
                ),
                "num_candidates": num_enhancement_candidates,
                "matching_score": matching_score,
                "min_ball_size": enhancement_min_size,
                "eligibility": (
                    "minimum_size"
                    if enhancement_mode == "granular_ball"
                    else "all_supervised_minority_training_nodes"
                ),
                "split_quality_threshold": args.quality_threshold,
                "min_split_size": args.min_split_size,
                "top_k": args.prototype_top_k,
                "layer_norm": args.prototype_layer_norm,
            },
            "training_seed_sampling": {
                "strategy": "ball_balanced_class_aware",
                "minority_ratio": minority_ratio,
                "requested_minority_ratio": args.minority_ratio,
                "original_minority_ratio": original_minority_ratio,
                "seed_batch_size": args.batch_size,
                "batches_per_epoch": len(train_seed_sampler),
                "minority_class_from_training_labels": train_seed_sampler.minority_class,
            },
            "loss": {
                "name": "weighted_cross_entropy",
                "class_weight_gamma": args.class_weight_gamma,
                "class_weights": class_weight.detach().cpu().tolist(),
            },
            "num_minority_train": int(minority_ids.numel()),
            "minority_class": data.minority_class,
            "split_sizes": {
                "train": int(data.train_idx.numel()),
                "validation": int(data.val_idx.numel()),
                "test": int(data.test_idx.numel()),
            },
            "supervised_minority_subsampling": supervision_metadata,
            "best_validation_pr_auc": best_val_pr,
            "validation_threshold_analysis": validation_threshold_analysis,
            "granular_ball_construction_time_seconds": (
                granular_ball_construction_time
            ),
            "average_training_time_per_epoch_seconds": float(np.mean(training_times)),
            "test_inference_time_seconds": test_inference_time,
            "test_inference_time_per_sample_seconds": inference_time_per_sample,
            "test_inference_time_per_sample_ms": inference_time_per_sample * 1000.0,
            "granular_ball_summary": summarize_ball_partition(balls),
            "dataset": dataset,
        }
    )
    validate_test_output_fields(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "ball_centers": centers,
            "minority_scaler_mean": data.minority_mean,
            "minority_scaler_std": data.minority_std,
            "threshold": threshold,
            "args": {
                **vars(args),
                "dataset": dataset,
                "data": str(data_path),
                "output_dir": str(output_dir),
            },
        },
        output_dir / "best_model.pt",
    )
    torch.save(
        [
            {
                "global_member_ids": minority_ids[ball.member_ids],
                "center": ball.center,
                "quality": ball.quality,
                "depth": ball.depth,
            }
            for ball in balls
        ],
        output_dir / "granular_balls.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate GraphGBME for imbalanced node classification"
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset_name,
        choices=tuple(DATASET_CONFIGS),
        default="tfinance",
        help="amazon、tfinance",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "graphgbme")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--supervised-minority-ratio",
        default="original",
        help="少数类比例：(0,1) 或 original",
    )
    parser.add_argument("--quality-threshold", type=float, default=0.90)
    parser.add_argument("--min-split-size", type=int, default=4, )
    parser.add_argument("--min-ball-size", type=int, default=2,)
    parser._option_string_actions["--min-split-size"].help = (
        "Minimum parent-ball node count required to attempt a split"
    )
    parser.add_argument("--prototype-top-k", type=int, default=3)
    parser.add_argument(
        "--enhancement-mode",
        choices=("granular_ball", "minority_node"),
        default="granular_ball",
        help=(
            "granular_ball uses quality-weighted ball prototypes; minority_node "
            "uses supervised minority nodes and distance-only matching"
        ),
    )
    parser.add_argument("--prototype-layer-norm", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fanouts", type=int, nargs=2, default=[25, 10])
    parser.add_argument("--eval-fanouts", type=int, nargs=2, default=[25, 10])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--minority-ratio", type=parse_minority_ratio, default='original')
    parser.add_argument(
        "--class-weight-gamma",
        type=float,
        default=0.25,
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu、cuda 或 cuda:0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
