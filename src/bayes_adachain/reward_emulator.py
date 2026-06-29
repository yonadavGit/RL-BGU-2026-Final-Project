from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

from .actions import Action, make_feature_vector


FEATURE_COLUMNS = [
    "write_ratio",
    "hot_key_ratio",
    "trans_arrival_rate",
    "execution_delay",
    "blocksize",
    "early_execution",
    "reorder",
    "blocksize * early_execution",
]
TARGET_COLUMN = "throughput"


def _normalize_column(name: str) -> str:
    normalized = name.strip()
    if normalized == "execution_delay (us)":
        return "execution_delay"
    return normalized


def load_adachain_trace_data(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            continue
        df = df.rename(columns={col: _normalize_column(col) for col in df.columns})
        missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
        if missing:
            continue

        trimmed = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            trimmed[col] = pd.to_numeric(trimmed[col], errors="coerce")
        trimmed = trimmed.dropna()
        trimmed["source_file"] = str(path)
        frames.append(trimmed)

    if not frames:
        raise ValueError("no valid AdaChain trace CSV files were provided")
    return pd.concat(frames, ignore_index=True)


@dataclass
class ThroughputEmulator:
    model: object
    feature_columns: List[str]
    residual_std: float

    def predict_expected(self, context: np.ndarray, action: Action) -> float:
        row = make_feature_vector(context, action).reshape(1, -1)
        return float(self.model.predict(row)[0])

    def sample_reward(self, context: np.ndarray, action: Action, rng: np.random.Generator) -> float:
        expected = self.predict_expected(context, action)
        return max(0.0, expected + rng.normal(0.0, self.residual_std))


def save_emulator(path: Path, emulator: ThroughputEmulator, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"emulator": emulator, "metadata": metadata}, path)


def load_emulator(path: Path) -> ThroughputEmulator:
    payload = joblib.load(path)
    return payload["emulator"]
