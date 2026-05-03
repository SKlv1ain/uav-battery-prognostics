"""
CLI entry-point for the auto-routing inference pipeline.

Usage (from the ml/ directory):
    python -m src.evaluation.run_pipeline \
        --data-dir data/raw/BatteryAgingARC-FY08Q4 \
        --multi-dir saved_models/multi_cycle \
        --single-dir saved_models/single_cycle \
        --batteries 5 6 7 18

Prints a per-battery, per-model metric table and a champion summary.
"""

from __future__ import annotations

import argparse
import json
from typing import List

from .metrics import regression_metrics
from .pipeline import load_bundle, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--multi-dir", default="saved_models/multi_cycle")
    parser.add_argument("--single-dir", default="saved_models/single_cycle")
    parser.add_argument(
        "--batteries", type=int, nargs="+", default=[5, 6, 7, 18]
    )
    parser.add_argument("--window-size", type=int, default=15)
    args = parser.parse_args()

    multi_bundle = load_bundle(args.multi_dir)
    single_bundle = load_bundle(args.single_dir)

    print("Bundle status:")
    print(f"  multi_cycle  : {'OK' if multi_bundle else 'MISSING'}")
    print(f"  single_cycle : {'OK' if single_bundle else 'MISSING'}\n")

    rows: List[dict] = []
    for b_id in args.batteries:
        res = run_pipeline(
            b_id, args.data_dir, multi_bundle, single_bundle, args.window_size
        )
        if res is None:
            print(f"B00{b_id:02d}: skipped (bundle missing)")
            continue
        for model_name, preds in res["preds"].items():
            m = regression_metrics(res["actual"], preds)
            rows.append(
                {
                    "battery": res["battery"],
                    "pipeline": res["pipeline"],
                    "model": model_name,
                    **{k: round(v, 5) for k, v in m.items()},
                }
            )

    if not rows:
        print("No results — train the models first.")
        return

    rows.sort(key=lambda r: r["rmse"])
    print(json.dumps(rows, indent=2))
    champ = rows[0]
    print(
        f"\n>>> Champion: [{champ['pipeline']}] {champ['model']} "
        f"({champ['battery']})  RMSE={champ['rmse']}  R2={champ['r2']}"
    )


if __name__ == "__main__":
    main()
