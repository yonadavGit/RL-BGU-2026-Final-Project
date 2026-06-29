#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bayes_adachain.reward_emulator import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ThroughputEmulator,
    load_adachain_trace_data,
    save_emulator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a throughput emulator from AdaChain CSV traces.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "external" / "AdaChain" / "src" / "learning" / "data",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "adachain_throughput_emulator.joblib")
    parser.add_argument("--metrics-output", type=Path, default=ROOT / "models" / "adachain_throughput_emulator_metrics.json")
    parser.add_argument("--model", choices=["random_forest", "extra_trees"], default="extra_trees")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_model(kind: str, seed: int):
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.data_dir.glob("**/*.csv"))
    data = load_adachain_trace_data(csv_paths)
    x = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = data[TARGET_COLUMN].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    model = build_model(args.model, args.seed)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    residual_std = float(np.std(y_train - train_pred))
    metrics = {
        "model": args.model,
        "num_csv_files_found": len(csv_paths),
        "num_csv_files_used": int(data["source_file"].nunique()),
        "num_rows": int(len(data)),
        "num_train_rows": int(len(x_train)),
        "num_test_rows": int(len(x_test)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "train_rmse": float(mean_squared_error(y_train, train_pred, squared=False)),
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
        "residual_std": residual_std,
    }

    emulator = ThroughputEmulator(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        residual_std=residual_std,
    )
    save_emulator(args.output, emulator, metrics)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.metrics_output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
