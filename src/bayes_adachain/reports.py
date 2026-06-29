from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .workloads import FEATURE_HIGH, FEATURE_LOW

ReportScenario = Literal[
    "biased_outliers",
    "persistent_bias",
    "heterogeneous_noise",
    "reliability_drift",
    "stale_reports",
    "outlier_bursts",
    "mixed_unreliable",
]


@dataclass
class ReportGenerator:
    rng: np.random.Generator
    num_nodes: int = 9
    unreliable_fraction: float = 0.25
    scenario: ReportScenario = "biased_outliers"
    honest_noise_scale: np.ndarray = None
    unreliable_bias_scale: np.ndarray = None
    outlier_probability: float = 0.10
    stale_lag: int = 4

    def __post_init__(self) -> None:
        if self.honest_noise_scale is None:
            self.honest_noise_scale = np.array([0.035, 0.035, 250.0, 350.0], dtype=float)
        if self.unreliable_bias_scale is None:
            self.unreliable_bias_scale = np.array([0.18, 0.18, 900.0, 2200.0], dtype=float)
        self.node_bias = self.rng.normal(0.0, self.honest_noise_scale * 0.5, size=(self.num_nodes, 4))
        self.noise_multiplier = np.ones((self.num_nodes, 4), dtype=float)
        num_unreliable = int(math.ceil(self.num_nodes * self.unreliable_fraction))
        self.unreliable_nodes = set(self.rng.choice(self.num_nodes, size=num_unreliable, replace=False).tolist())
        self.bias_direction = self.rng.choice([-1.0, 1.0], size=(self.num_nodes, 4))
        self.history = []
        self.episode = 0

        for node in self.unreliable_nodes:
            if self.scenario in {"persistent_bias", "mixed_unreliable"}:
                self.node_bias[node] += self.bias_direction[node] * self.unreliable_bias_scale
            if self.scenario in {"heterogeneous_noise", "mixed_unreliable"}:
                self.noise_multiplier[node] = np.array([3.0, 3.0, 4.0, 4.0])

    def sample(self, true_context: np.ndarray) -> np.ndarray:
        self.history.append(np.asarray(true_context, dtype=float).copy())
        reports = []
        for node in range(self.num_nodes):
            report_context = self._reported_context_for_node(node, true_context)
            noise_scale = self.honest_noise_scale * self.noise_multiplier[node]
            noise = self.rng.standard_t(df=5, size=4) * noise_scale
            report = report_context + self.node_bias[node] + noise

            if node in self.unreliable_nodes and self.scenario in {"outlier_bursts", "mixed_unreliable"}:
                if self.rng.random() < self.outlier_probability:
                    burst = self.bias_direction[node] * self.unreliable_bias_scale * self.rng.uniform(1.0, 2.5)
                    report = report + burst

            if node in self.unreliable_nodes and self.scenario == "biased_outliers":
                bias = self.bias_direction[node] * self.unreliable_bias_scale * 1.8
                heavy_tail_noise = self.rng.standard_t(df=3, size=4) * self.honest_noise_scale * 2.0
                report = true_context + bias + heavy_tail_noise

            reports.append(np.clip(report, FEATURE_LOW, FEATURE_HIGH))
        self.episode += 1
        return np.vstack(reports)

    def _reported_context_for_node(self, node: int, true_context: np.ndarray) -> np.ndarray:
        if node not in self.unreliable_nodes:
            return true_context

        if self.scenario in {"reliability_drift", "mixed_unreliable"}:
            phase = (self.episode // 50) % 3
            if phase == 1:
                self.noise_multiplier[node] = np.array([4.0, 4.0, 5.0, 5.0])
                drift = self.bias_direction[node] * self.unreliable_bias_scale * 0.7
                return true_context + drift
            if phase == 2:
                self.noise_multiplier[node] = np.array([1.5, 1.5, 2.0, 2.0])
            else:
                self.noise_multiplier[node] = np.ones(4)

        if self.scenario in {"stale_reports", "mixed_unreliable"} and len(self.history) > self.stale_lag:
            return self.history[-self.stale_lag - 1]

        return true_context
