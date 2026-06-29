from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .agents import Agent
from .actions import Action
from .metrics import result_frame
from .reports import ReportGenerator
from .reward_emulator import ThroughputEmulator
from .rewards import expected_throughput, sample_reward
from .workloads import WorkloadGenerator


@dataclass
class ExperimentConfig:
    episodes: int = 160
    seed: int = 7
    num_nodes: int = 9
    unreliable_fraction: float = 0.25
    report_scenario: str = "biased_outliers"
    phase_length: int = 40
    reward_noise_sd: float = 45.0
    show_progress: bool = True


def run_experiment(
    config: ExperimentConfig,
    actions: List[Action],
    agents: List[Agent],
    throughput_emulator: Optional[ThroughputEmulator] = None,
) -> pd.DataFrame:
    env_rng = np.random.default_rng(config.seed)
    workload_gen = WorkloadGenerator(env_rng, phase_length=config.phase_length)
    report_gen = ReportGenerator(
        np.random.default_rng(config.seed + 1),
        num_nodes=config.num_nodes,
        unreliable_fraction=config.unreliable_fraction,
        scenario=config.report_scenario,
    )
    reward_rngs: Dict[str, np.random.Generator] = {
        agent.name: np.random.default_rng(config.seed + 100 + idx)
        for idx, agent in enumerate(agents)
    }

    def expected_reward(context: np.ndarray, action: Action) -> float:
        if throughput_emulator is not None:
            return throughput_emulator.predict_expected(context, action)
        return expected_throughput(context, action)

    def observed_reward(context: np.ndarray, action: Action, rng: np.random.Generator) -> float:
        if throughput_emulator is not None:
            return throughput_emulator.sample_reward(context, action, rng)
        return sample_reward(context, action, rng, noise_sd=config.reward_noise_sd)

    rows = []
    episode_iter = tqdm(
        range(config.episodes),
        desc="Simulating episodes",
        unit="episode",
        disable=not config.show_progress,
    )
    for episode in episode_iter:
        true_context = workload_gen.sample(episode)
        reports = report_gen.sample(true_context)
        best_action = max(actions, key=lambda action: expected_reward(true_context, action))
        best_expected = expected_reward(true_context, best_action)

        for agent in agents:
            decision = agent.select_action(reports, true_context=true_context)
            reward = observed_reward(true_context, decision.action, reward_rngs[agent.name])
            agent.observe_reward(reports, decision.action, reward)
            estimated_error = float(np.linalg.norm(decision.estimated_context - true_context))
            decision_expected = expected_reward(true_context, decision.action)
            rows.append(
                {
                    "episode": episode + 1,
                    "agent": agent.name,
                    "action": decision.action.label,
                    "oracle_action": best_action.label,
                    "architecture_correct": decision.action == best_action,
                    "reward": reward,
                    "expected_reward": decision_expected,
                    "oracle_expected_reward": best_expected,
                    "regret": max(0.0, best_expected - decision_expected),
                    "context_error_l2": estimated_error,
                    "true_write_ratio": true_context[0],
                    "true_hot_key_ratio": true_context[1],
                    "true_arrival_rate": true_context[2],
                    "true_execution_delay": true_context[3],
                    "estimated_write_ratio": decision.estimated_context[0],
                    "estimated_hot_key_ratio": decision.estimated_context[1],
                    "estimated_arrival_rate": decision.estimated_context[2],
                    "estimated_execution_delay": decision.estimated_context[3],
                }
            )

    return result_frame(rows)
