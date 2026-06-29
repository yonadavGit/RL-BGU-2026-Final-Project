from __future__ import annotations

import numpy as np

from .actions import Action


def expected_throughput(context: np.ndarray, action: Action) -> float:
    """Synthetic reward surface inspired by AdaChain's architectural tradeoffs."""
    write, hot, arrival_rate, execution_delay = np.asarray(context, dtype=float)
    arrival = arrival_rate / 3000.0
    delay = execution_delay / 5000.0
    score = 420.0

    if action.name == "OX":
        score += 650.0 * write * hot
        score += 150.0 * (1.0 - delay)
        score -= 160.0 * arrival
    elif action.name == "OXII":
        score += 300.0 * delay
        score += 180.0 * hot
        score += 110.0 * arrival
        score -= 170.0 * write * hot
    elif action.name == "XOV":
        score += 580.0 * (1.0 - hot)
        score += 250.0 * delay
        score -= 420.0 * write * hot
    elif action.name == "XOV++":
        score += 570.0 * delay
        score += 210.0 * (1.0 - write)
        score += 130.0 * hot
        score -= 520.0 * write * hot
    else:
        raise ValueError(f"unknown action family: {action.name}")

    block_penalty = abs(action.blocksize - 80.0) * 0.18
    if action.blocksize >= 300 and hot > 0.9:
        block_penalty += 90.0
    return max(80.0, score - block_penalty)


def sample_reward(context: np.ndarray, action: Action, rng: np.random.Generator, noise_sd: float = 45.0) -> float:
    return max(0.0, expected_throughput(context, action) + rng.normal(0.0, noise_sd))

