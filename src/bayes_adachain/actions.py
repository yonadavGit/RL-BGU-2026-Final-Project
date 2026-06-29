from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Action:
    name: str
    blocksize: int
    early_execution: bool
    reorder: bool

    @property
    def features(self) -> np.ndarray:
        early = float(self.early_execution)
        reorder = float(self.reorder)
        product = float(self.blocksize if self.early_execution else -self.blocksize)
        return np.array([float(self.blocksize), early, reorder, product], dtype=float)

    @property
    def label(self) -> str:
        return f"{self.name}(block={self.blocksize})"


def default_action_space() -> List[Action]:
    """Compact AdaChain-style action space for simulation.

    The original artifact enumerates many block sizes. This reduced set keeps
    experiments fast while preserving the architectural knobs AdaChain uses.
    """
    return [
        Action("OX", 50, False, False),
        Action("OX", 200, False, False),
        Action("OXII", 100, False, True),
        Action("OXII", 400, False, True),
        Action("XOV", 1, True, False),
        Action("XOV", 50, True, False),
        Action("XOV++", 50, True, True),
        Action("XOV++", 400, True, True),
    ]


def make_feature_vector(context: np.ndarray, action: Action) -> np.ndarray:
    return np.concatenate([np.asarray(context, dtype=float), action.features])

