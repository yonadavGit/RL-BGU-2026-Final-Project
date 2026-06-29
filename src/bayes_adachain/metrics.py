from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .actions import Action
from .rewards import expected_throughput


def oracle_best(context: np.ndarray, actions: List[Action]) -> Action:
    return max(actions, key=lambda action: expected_throughput(context, action))


def result_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["cumulative_reward"] = df.groupby("agent")["reward"].cumsum()
    df["cumulative_regret"] = df.groupby("agent")["regret"].cumsum()
    return df

