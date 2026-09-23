"""Train two dataset configurations from scratch with five random seeds.

No checkpoint or learned model parameters are read. Each seed starts a fresh
``train.py`` process, initializes a new model, trains it, and evaluates its own
best validation checkpoint on the test split.

Amazon uses split quality threshold 1.00, while T-Finance uses 0.95. Prototype
quality filtering is disabled. The five default seeds are 42, 43, 44, 45, and 46. Standard
deviation is the sample standard deviation (denominator n - 1), as normally
reported for multi-seed experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path("outputs/five-seed-gb-saign")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
SHARED_FANOUTS = [25, 10]
COMMON_TRAINING_CONFIGURATION: dict[str, Any] = {
    "min_ball_size": 2,
    "prototype_top_k": 3,
    "prototype_layer_norm": False,
    "hidden_dim": 128,
    "dropout": 0.3,
    "batch_size": 1024,
    "class_weight_gamma": 0.25,
    "epochs": 50,
    "patience": 15,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
DATASET_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "amazon": {
        "quality_threshold": 1.00,
        "min_split_size": 7,
        "minority_ratio": "original",
    },
    "tfinance": {
        "quality_threshold": 0.95,
        "min_split_size": 9,
        "minority_ratio": "original",
    },
}

# Metrics suitable for an aggregate multi-seed report. Nested values are
# addressed by tuples so the report can also include ball statistics.
REPORT_METRICS: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy",),
    "minority_recall": ("minority_recall",),
    "minority_precision": ("minority_precision",),
    "minority_f1": ("minority_f1",),
    "macro_f1": ("macro_f1",),
    "roc_auc": ("roc_auc",),
    "pr_auc": ("pr_auc",),
    "g_mean": ("g_mean",),
    "decision_threshold": ("threshold",),
    "best_validation_pr_auc": ("best_validation_pr_auc",),
    "num_balls": ("num_balls",),
    "num_reliable_prototype_balls": ("num_reliable_prototype_balls",),
    "average_granular_ball_quality": (
        "granular_ball_summary",
        "average_quality",
    ),
    "granular_ball_construction_time_seconds": (
        "granular_ball_construction_time_seconds",
    ),
    "average_training_time_per_epoch_seconds": (
        "average_training_time_per_epoch_seconds",
    ),
    "test_inference_time_seconds": ("test_inference_time_seconds",),
    "test_inference_time_per_sample_ms": (
        "test_inference_time_per_sample_ms",
    ),
}


def parse_seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("随机种子必须是逗号分隔的整数") from exc
    if len(seeds) != 5:
        raise argparse.ArgumentTypeError("必须提供恰好 5 个随机种子")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("5 个随机种子不能重复")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="不读取旧模型，以 5 个随机种子从头训练并报告 mean ± std。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="5 次训练及汇总报告的输出目录",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=DEFAULT_SEEDS,
        help="恰好 5 个逗号分隔的随机种子（默认：42,43,44,45,46）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练设备，例如 cuda、cuda:0 或 cpu",
    )
    parser.add_argument(
        "--amazon-min-split-size",
        type=int,
        default=DATASET_CONFIGURATIONS["amazon"]["min_split_size"],
        help="Amazon minimum node count required for granular-ball splitting",
    )
    parser.add_argument(
        "--tfinance-min-split-size",
        type=int,
        default=DATASET_CONFIGURATIONS["tfinance"]["min_split_size"],
        help="T-Finance minimum node count required for granular-ball splitting",
    )
    parser.add_argument(
        "--amazon-quality-threshold",
        type=float,
        default=DATASET_CONFIGURATIONS["amazon"]["quality_threshold"],
        help="Amazon granular-ball split quality threshold",
    )
    parser.add_argument(
        "--tfinance-quality-threshold",
        type=float,
        default=DATASET_CONFIGURATIONS["tfinance"]["quality_threshold"],
        help="T-Finance granular-ball split quality threshold",
    )
    parser.add_argument(
        "--min-ball-size",
        type=int,
        default=COMMON_TRAINING_CONFIGURATION["min_ball_size"],
        help="Minimum granular-ball size eligible as a prototype",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=COMMON_TRAINING_CONFIGURATION["hidden_dim"],
        help="Hidden feature dimension (default: 128)",
    )
    args = parser.parse_args()
    if args.amazon_min_split_size < 2:
        parser.error("--amazon-min-split-size must be at least 2")
    if args.tfinance_min_split_size < 2:
        parser.error("--tfinance-min-split-size must be at least 2")
    if not 0.0 <= args.amazon_quality_threshold <= 1.0:
        parser.error("--amazon-quality-threshold must be in [0, 1]")
    if not 0.0 <= args.tfinance_quality_threshold <= 1.0:
        parser.error("--tfinance-quality-threshold must be in [0, 1]")
    if args.min_ball_size < 1:
        parser.error("--min-ball-size must be at least 1")
    if args.hidden_dim < 1:
        parser.error("--hidden-dim must be at least 1")
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = mapping
    try:
        for key in path:
            value = value[key]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"metrics.json 缺少字段：{'.'.join(path)}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"指标 {'.'.join(path)} 不是数值：{value!r}")
    return float(value)


def build_fixed_train_args(
    dataset: str,
    dataset_configuration: dict[str, Any],
    device: str,
) -> tuple[list[str], dict[str, Any]]:
    """Build a complete configuration without loading any prior checkpoint."""
    scalar_options: tuple[tuple[str, str], ...] = (
        ("min_ball_size", "--min-ball-size"),
        ("min_split_size", "--min-split-size"),
        ("prototype_top_k", "--prototype-top-k"),
        ("hidden_dim", "--hidden-dim"),
        ("dropout", "--dropout"),
        ("batch_size", "--batch-size"),
        ("minority_ratio", "--minority-ratio"),
        ("class_weight_gamma", "--class-weight-gamma"),
        ("epochs", "--epochs"),
        ("patience", "--patience"),
        ("lr", "--lr"),
        ("weight_decay", "--weight-decay"),
    )
    resolved = {**COMMON_TRAINING_CONFIGURATION, **dataset_configuration}
    resolved: dict[str, Any] = {
        **resolved,
        "dataset": dataset,
        "fanouts": SHARED_FANOUTS.copy(),
        "eval_fanouts": SHARED_FANOUTS.copy(),
        "device": device,
    }
    command_args = [
        "--dataset",
        dataset,
        "--quality-threshold",
        str(resolved["quality_threshold"]),
    ]
    for name, option in scalar_options:
        command_args.extend((option, str(resolved[name])))
    command_args.extend(("--fanouts", *(str(value) for value in resolved["fanouts"])))
    command_args.extend(
        ("--eval-fanouts", *(str(value) for value in resolved["eval_fanouts"]))
    )
    if bool(resolved["prototype_layer_norm"]):
        command_args.append("--prototype-layer-norm")
    command_args.extend(("--device", str(resolved["device"])))
    return command_args, resolved


def run_training(command: list[str], log_path: Path, project_dir: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def make_report(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != 5:
        raise ValueError(f"mean ± std 需要 5 次结果，实际得到 {len(runs)} 次")
    report_metrics: dict[str, dict[str, Any]] = {}
    for report_name, metric_path in REPORT_METRICS.items():
        values = [get_nested(run["metrics"], metric_path) for run in runs]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        report_metrics[report_name] = {
            "mean": mean,
            "std": std,
            "mean_plus_minus_std": f"{mean:.6f} ± {std:.6f}",
            "values": values,
        }
    return {
        "num_runs": len(runs),
        "seeds": [run["seed"] for run in runs],
        "std_definition": "sample standard deviation (ddof=1)",
        "metrics": report_metrics,
    }


def write_report_csv(path: Path, reports: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "metric",
                "mean",
                "std",
                "mean_plus_minus_std",
            ),
        )
        writer.writeheader()
        for dataset, report in reports.items():
            for metric, values in report["metrics"].items():
                writer.writerow(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "mean": values["mean"],
                        "std": values["std"],
                        "mean_plus_minus_std": values["mean_plus_minus_std"],
                    }
                )


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else project_dir / args.output_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_args_by_dataset: dict[str, list[str]] = {}
    resolved_configurations: dict[str, dict[str, Any]] = {}
    for dataset, configuration in DATASET_CONFIGURATIONS.items():
        configuration = {
            **configuration,
            "min_ball_size": args.min_ball_size,
            "hidden_dim": args.hidden_dim,
            "min_split_size": getattr(args, f"{dataset}_min_split_size"),
            "quality_threshold": getattr(args, f"{dataset}_quality_threshold"),
        }
        fixed_args, resolved = build_fixed_train_args(
            dataset,
            configuration,
            args.device,
        )
        fixed_args_by_dataset[dataset] = fixed_args
        resolved_configurations[dataset] = resolved
    experiment = {
        "training_mode": "from_scratch_no_checkpoint_loading",
        "seeds": list(args.seeds),
        "resolved_training_configurations": resolved_configurations,
    }
    write_json(output_dir / "experiment_config.json", experiment)
    print("从头训练配置（不读取任何旧检查点）：")
    print(json.dumps(experiment, ensure_ascii=False, indent=2, default=str))

    all_runs: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    train_script = project_dir / "train.py"
    for dataset in DATASET_CONFIGURATIONS:
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_runs: list[dict[str, Any]] = []
        for index, seed in enumerate(args.seeds, start=1):
            run_dir = dataset_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(train_script),
                *fixed_args_by_dataset[dataset],
                "--seed",
                str(seed),
                "--output-dir",
                str(run_dir),
            ]
            print(
                f"\n=== dataset={dataset} run={index}/5 seed={seed} ===",
                flush=True,
            )
            run_training(command, run_dir / "train.log", project_dir)
            metrics_path = run_dir / "metrics.json"
            metrics = read_json(metrics_path)
            # Validate every report field immediately after the run.
            for metric_path in REPORT_METRICS.values():
                get_nested(metrics, metric_path)
            record = {
                "dataset": dataset,
                "run": index,
                "seed": seed,
                "output_dir": str(run_dir),
                "metrics": metrics,
            }
            dataset_runs.append(record)
            all_runs.append(record)
            write_json(dataset_dir / "all_runs.json", dataset_runs)
            write_json(output_dir / "all_runs.json", all_runs)

        report = make_report(dataset_runs)
        report["dataset"] = dataset
        report["training_mode"] = "from_scratch_no_checkpoint_loading"
        report["resolved_training_configuration"] = resolved_configurations[dataset]
        reports[dataset] = report
        write_json(dataset_dir / "mean_std_metrics.json", report)
        write_report_csv(dataset_dir / "mean_std_metrics.csv", {dataset: report})

        print(f"\n{dataset} 的 5 个随机种子 mean ± std：")
        for metric, values in report["metrics"].items():
            print(f"  {metric}: {values['mean_plus_minus_std']}")

    combined_report = {
        "training_mode": "from_scratch_no_checkpoint_loading",
        "seeds": list(args.seeds),
        "datasets": reports,
    }
    write_json(output_dir / "mean_std_metrics.json", combined_report)
    write_report_csv(output_dir / "mean_std_metrics.csv", reports)
    print(f"\n完整报告：{output_dir / 'mean_std_metrics.json'}")


if __name__ == "__main__":
    main()
