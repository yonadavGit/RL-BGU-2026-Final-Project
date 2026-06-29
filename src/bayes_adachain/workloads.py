from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = ["write_ratio", "hot_key_ratio", "arrival_rate", "execution_delay"]
FEATURE_LOW = np.array([0.01, 0.01, 100.0, 0.0], dtype=float)
FEATURE_HIGH = np.array([0.99, 0.99, 6000.0, 15000.0], dtype=float)


WORKLOAD_PHASES = np.array(
    [
        [0.20, 0.95, 3000.0, 5000.0],
        [0.50, 0.99, 1000.0, 1000.0],
        [0.50, 0.10, 3000.0, 10000.0],
        [0.90, 0.95, 1000.0, 0.0],
    ],
    dtype=float,
)


@dataclass
class WorkloadGenerator:
    rng: np.random.Generator
    phase_length: int = 40
    drift_scale: float = 0.035

    def sample(self, episode: int) -> np.ndarray:
        phase = (episode // self.phase_length) % len(WORKLOAD_PHASES)
        center = WORKLOAD_PHASES[phase]
        feature_scale = np.array([0.08, 0.08, 550.0, 900.0], dtype=float)
        drift = self.rng.normal(0.0, self.drift_scale, size=4) * feature_scale
        state = center + drift
        return np.clip(state, FEATURE_LOW, FEATURE_HIGH)

