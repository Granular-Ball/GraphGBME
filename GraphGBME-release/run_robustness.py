"""Train GraphGBME at 1%, 2%, 3%, and 4% minority supervision with five seeds each."""

from __future__ import annotations

import json
from pathlib import Path

import train
from data import apply_supervised_minority_ratio


RATIOS = ("0.01", "0.02", "0.03", "0.04")


def main() -> None:
    args = train.parse_experiment_args(__doc__)
    output_dir = args.output_dir or Path("outputs") / "robustness" / args.dataset
    train.start_experiment(args, output_dir, ratios=RATIOS)

    summary = {}
    for ratio in RATIOS:
        print(f"{args.dataset} | supervised minority ratio={ratio}", flush=True)
        summary[ratio] = train.run_seeds(
            args, output_dir / f"ratio_{ratio}",
            prepare_data=lambda data, seed: apply_supervised_minority_ratio(data, ratio, seed),
        )
    (output_dir / "mean_std_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
