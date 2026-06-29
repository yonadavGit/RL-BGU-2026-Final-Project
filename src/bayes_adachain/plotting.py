from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _rolling(df: pd.DataFrame, column: str, window: int = 8) -> pd.DataFrame:
    ordered = df.sort_values(["agent", "episode"]).copy()
    ordered[f"{column}_rolling"] = (
        ordered.groupby("agent")[column]
        .transform(lambda values: values.rolling(window, min_periods=1).mean())
    )
    return ordered


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_cumulative(df, "cumulative_reward", "Cumulative reward", output_dir / "cumulative_reward.png")
    _plot_cumulative(df, "cumulative_regret", "Cumulative regret", output_dir / "cumulative_regret.png")
    _plot_rolling(df, "reward", "Rolling reward", "reward_rolling", output_dir / "rolling_reward.png")
    _plot_rolling(df, "context_error_l2", "Context estimation error", "context_error_l2_rolling", output_dir / "context_error.png")
    _plot_accuracy(df, output_dir / "architecture_accuracy.png")


def _plot_cumulative(df: pd.DataFrame, column: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent, part in df.groupby("agent"):
        ax.plot(part["episode"], part[column], label=agent)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(column.replace("_", " "))
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_rolling(df: pd.DataFrame, column: str, title: str, rolling_column: str, path: Path) -> None:
    rolled = _rolling(df, column)
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent, part in rolled.groupby("agent"):
        ax.plot(part["episode"], part[rolling_column], label=agent)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(column.replace("_", " "))
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_accuracy(df: pd.DataFrame, path: Path) -> None:
    rolled = _rolling(df.assign(architecture_correct=df["architecture_correct"].astype(float)), "architecture_correct")
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent, part in rolled.groupby("agent"):
        ax.plot(part["episode"], part["architecture_correct_rolling"], label=agent)
    ax.set_title("Rolling architecture selection accuracy")
    ax.set_xlabel("Episode")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
