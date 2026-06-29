#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bayes_adachain.actions import default_action_space
from bayes_adachain.agents import BayesianImputationAgent, BayesianUncertaintyAgent, ForestBanditAgent
from bayes_adachain.estimators import MeanEstimator, MedianEstimator, RobustBayesianEstimator
from bayes_adachain.experiment import ExperimentConfig, run_experiment
from bayes_adachain.plotting import plot_results
from bayes_adachain.reward_emulator import load_emulator


def build_agents(config: ExperimentConfig, actions):
    return [
        ForestBanditAgent(
            name="oracle",
            actions=actions,
            rng=np.random.default_rng(config.seed + 10),
            sees_true_context=True,
        ),
        ForestBanditAgent(
            name="mean_reports",
            actions=actions,
            rng=np.random.default_rng(config.seed + 20),
            estimator=MeanEstimator(),
        ),
        ForestBanditAgent(
            name="median_reports",
            actions=actions,
            rng=np.random.default_rng(config.seed + 30),
            estimator=MedianEstimator(),
        ),
        BayesianImputationAgent(
            name="bayesian_imputation",
            actions=actions,
            rng=np.random.default_rng(config.seed + 40),
            estimator=RobustBayesianEstimator(
                num_nodes=config.num_nodes,
                rng=np.random.default_rng(config.seed + 41),
            ),
            num_imputations=5,
            decision_samples=32,
        ),
        BayesianUncertaintyAgent(
            name="bayesian_uncertainty",
            actions=actions,
            rng=np.random.default_rng(config.seed + 50),
            estimator=RobustBayesianEstimator(
                num_nodes=config.num_nodes,
                rng=np.random.default_rng(config.seed + 51),
            ),
            decision_samples=32,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian AdaChain simulation.")
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nodes", type=int, default=9)
    parser.add_argument("--unreliable-fraction", type=float, default=0.25)
    parser.add_argument(
        "--report-scenario",
        choices=[
            "biased_outliers",
            "persistent_bias",
            "heterogeneous_noise",
            "reliability_drift",
            "stale_reports",
            "outlier_bursts",
            "mixed_unreliable",
        ],
        default="biased_outliers",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "smoke")
    parser.add_argument("--reward-source", choices=["synthetic", "emulator"], default="synthetic")
    parser.add_argument(
        "--emulator-path",
        type=Path,
        default=ROOT / "models" / "adachain_throughput_emulator.joblib",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        episodes=args.episodes,
        seed=args.seed,
        num_nodes=args.nodes,
        unreliable_fraction=args.unreliable_fraction,
        report_scenario=args.report_scenario,
        show_progress=not args.no_progress,
    )
    actions = default_action_space()
    agents = build_agents(config, actions)
    throughput_emulator = None
    if args.reward_source == "emulator":
        throughput_emulator = load_emulator(args.emulator_path)
    df = run_experiment(config, actions, agents, throughput_emulator=throughput_emulator)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "episode_results.csv"
    df.to_csv(csv_path, index=False)
    summary = (
        df.groupby("agent")
        .agg(
            total_reward=("reward", "sum"),
            total_regret=("regret", "sum"),
            mean_context_error=("context_error_l2", "mean"),
            architecture_accuracy=("architecture_correct", "mean"),
        )
        .sort_values("total_reward", ascending=False)
    )
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path)

    if not args.no_plots:
        plot_results(df, args.output_dir)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(summary.round(3))


if __name__ == "__main__":
    main()
