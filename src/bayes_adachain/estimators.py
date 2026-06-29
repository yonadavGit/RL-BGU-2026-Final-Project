from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .workloads import FEATURE_HIGH, FEATURE_LOW


class ContextEstimator:
    def estimate(self, reports: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanEstimator(ContextEstimator):
    def estimate(self, reports: np.ndarray) -> np.ndarray:
        return np.mean(reports, axis=0)


class MedianEstimator(ContextEstimator):
    def estimate(self, reports: np.ndarray) -> np.ndarray:
        return np.median(reports, axis=0)


@dataclass
class RobustBayesianEstimator:
    """Approximate sequential Bayesian filter for noisy node reports.

    This is intentionally lightweight: feature-wise robust Student-t weighting,
    mean-reverting node bias/noise estimates, and Gaussian posterior samples
    around the robust latent context estimate.
    """

    num_nodes: int
    rng: np.random.Generator
    nu: float = 4.0
    rho_b: float = 0.95
    rho_sigma: float = 0.95
    tau_floor: np.ndarray = None

    def __post_init__(self) -> None:
        if self.tau_floor is None:
            self.tau_floor = np.array([0.025, 0.025, 180.0, 250.0], dtype=float)
        self.bias = np.zeros((self.num_nodes, 4), dtype=float)
        self.sigma = np.tile(self.tau_floor * 1.5, (self.num_nodes, 1))
        self.posterior_mean = np.zeros(4, dtype=float)
        self.posterior_std = self.tau_floor.copy()

    def update(self, reports: np.ndarray) -> None:
        debiased = reports - self.rho_b * self.bias
        mean = np.median(debiased, axis=0)

        for _ in range(4):
            residual = debiased - mean
            scale = np.maximum(self.sigma, self.tau_floor)
            weights = (self.nu + 1.0) / (self.nu + (residual / scale) ** 2)
            precision = weights / (scale**2)
            mean = np.sum(precision * debiased, axis=0) / np.sum(precision, axis=0)

        residual = reports - mean
        raw_bias = residual - np.mean(residual, axis=0, keepdims=True)
        self.bias = self.rho_b * self.bias + (1.0 - self.rho_b) * raw_bias
        self.bias -= np.mean(self.bias, axis=0, keepdims=True)

        abs_resid = np.abs(reports - mean - self.bias)
        target_sigma = np.maximum(self.tau_floor, 1.253 * abs_resid)
        self.sigma = self.rho_sigma * self.sigma + (1.0 - self.rho_sigma) * target_sigma

        scale = np.maximum(self.sigma, self.tau_floor)
        residual = reports - self.bias - mean
        weights = (self.nu + 1.0) / (self.nu + (residual / scale) ** 2)
        precision = weights / (scale**2)
        post_var = 1.0 / np.sum(precision, axis=0)
        self.posterior_mean = np.clip(mean, FEATURE_LOW, FEATURE_HIGH)
        self.posterior_std = np.maximum(np.sqrt(post_var), self.tau_floor * 0.35)

    def draw_posterior(self, num_samples: int) -> np.ndarray:
        samples = self.rng.normal(self.posterior_mean, self.posterior_std, size=(num_samples, 4))
        return np.clip(samples, FEATURE_LOW, FEATURE_HIGH)

    def sample(self, reports: np.ndarray, num_samples: int) -> np.ndarray:
        self.update(reports)
        return self.draw_posterior(num_samples)

    def estimate(self, reports: np.ndarray) -> np.ndarray:
        self.update(reports)
        return self.posterior_mean.copy()
