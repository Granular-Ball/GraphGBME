"""Train the full GraphGBME model and three component ablations with five seeds each."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train


VARIANTS = (
    "full",
    "without_prototype_enhancement",
    "without_graph_constraint",
    "without_class_weighting",
)


def main() -> None:
    args = train.parse_experiment_args(__doc__)
    output_dir = args.output_dir or Path("outputs") / "ablation" / args.dataset
    train.start_experiment(args, output_dir, variants=VARIANTS)

    summary = {}
    for variant in VARIANTS:
        print(f"{args.dataset} | variant={variant}", flush=True)
        variant_args = argparse.Namespace(**vars(args), ablation_variant=variant)
        if variant == "without_class_weighting":
            variant_args.class_weight_gamma = 0.0
        summary[variant] = train.run_seeds(
            variant_args, output_dir / variant,
            graph_constrained=variant != "without_graph_constraint",
            use_prototypes=variant != "without_prototype_enhancement",
        )
    (output_dir / "mean_std_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
