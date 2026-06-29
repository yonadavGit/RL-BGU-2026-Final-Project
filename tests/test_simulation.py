from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bayes_adachain.actions import Action, default_action_space, make_feature_vector
from bayes_adachain.agents import BayesianImputationAgent, BayesianUncertaintyAgent, ForestBanditAgent
from bayes_adachain.estimators import MeanEstimator, MedianEstimator, RobustBayesianEstimator
from bayes_adachain.experiment import ExperimentConfig, run_experiment
from bayes_adachain.reports import ReportGenerator
from bayes_adachain.reward_emulator import (
    FEATURE_COLUMNS,
    ThroughputEmulator,
    load_adachain_trace_data,
    load_emulator,
    save_emulator,
)
from bayes_adachain.workloads import FEATURE_HIGH, FEATURE_LOW


class ActionTests(unittest.TestCase):
    def test_action_features_follow_adachain_convention(self) -> None:
        xov = Action("XOV", 50, True, False)
        ox = Action("OX", 200, False, False)

        np.testing.assert_allclose(xov.features, np.array([50.0, 1.0, 0.0, 50.0]))
        np.testing.assert_allclose(ox.features, np.array([200.0, 0.0, 0.0, -200.0]))

    def test_feature_vector_concatenates_context_and_action(self) -> None:
        context = np.array([0.2, 0.9, 3000.0, 5000.0])
        action = Action("XOV++", 50, True, True)

        feature_vector = make_feature_vector(context, action)

        self.assertEqual(feature_vector.shape, (8,))
        np.testing.assert_allclose(feature_vector[:4], context)
        np.testing.assert_allclose(feature_vector[4:], action.features)


class ReportGeneratorTests(unittest.TestCase):
    def test_reports_have_expected_shape_and_bounds(self) -> None:
        generator = ReportGenerator(np.random.default_rng(1), num_nodes=9, unreliable_fraction=0.25)
        true_context = np.array([0.4, 0.7, 2500.0, 3000.0])

        reports = generator.sample(true_context)

        self.assertEqual(reports.shape, (9, 4))
        self.assertTrue(np.all(reports >= FEATURE_LOW))
        self.assertTrue(np.all(reports <= FEATURE_HIGH))

    def test_unreliable_node_count_uses_ceiling(self) -> None:
        generator_40 = ReportGenerator(np.random.default_rng(2), num_nodes=9, unreliable_fraction=0.40)
        generator_50 = ReportGenerator(np.random.default_rng(2), num_nodes=9, unreliable_fraction=0.50)

        self.assertEqual(len(generator_40.unreliable_nodes), 4)
        self.assertEqual(len(generator_50.unreliable_nodes), 5)

    def test_biased_outlier_reports_are_more_deviant_than_reliable_reports(self) -> None:
        generator = ReportGenerator(np.random.default_rng(3), num_nodes=10, unreliable_fraction=0.3)
        true_context = np.array([0.4, 0.5, 2500.0, 3000.0])
        reports = generator.sample(true_context)

        unreliable = sorted(generator.unreliable_nodes)
        honest = [idx for idx in range(generator.num_nodes) if idx not in generator.unreliable_nodes]
        unreliable_error = np.linalg.norm(reports[unreliable] - true_context, axis=1).mean()
        honest_error = np.linalg.norm(reports[honest] - true_context, axis=1).mean()

        self.assertGreater(unreliable_error, honest_error)

    def test_persistent_bias_scenario_offsets_unreliable_nodes(self) -> None:
        generator = ReportGenerator(
            np.random.default_rng(16),
            num_nodes=8,
            unreliable_fraction=0.5,
            scenario="persistent_bias",
        )

        unreliable_bias_norm = np.linalg.norm(generator.node_bias[list(generator.unreliable_nodes)], axis=1).mean()
        honest = [idx for idx in range(generator.num_nodes) if idx not in generator.unreliable_nodes]
        honest_bias_norm = np.linalg.norm(generator.node_bias[honest], axis=1).mean()

        self.assertGreater(unreliable_bias_norm, honest_bias_norm)

    def test_stale_report_scenario_uses_history_after_lag(self) -> None:
        generator = ReportGenerator(
            np.random.default_rng(17),
            num_nodes=4,
            unreliable_fraction=1.0,
            scenario="stale_reports",
            honest_noise_scale=np.zeros(4),
            stale_lag=1,
        )
        first = np.array([0.2, 0.3, 1000.0, 1000.0])
        second = np.array([0.8, 0.9, 5000.0, 9000.0])
        generator.sample(first)
        reports = generator.sample(second)

        np.testing.assert_allclose(reports, np.tile(first, (4, 1)))


class EstimatorTests(unittest.TestCase):
    def test_mean_and_median_estimators(self) -> None:
        reports = np.array(
            [
                [0.2, 0.8, 1000.0, 2000.0],
                [0.4, 0.6, 2000.0, 4000.0],
                [0.8, 0.2, 3000.0, 6000.0],
            ]
        )

        np.testing.assert_allclose(MeanEstimator().estimate(reports), np.mean(reports, axis=0))
        np.testing.assert_allclose(MedianEstimator().estimate(reports), np.median(reports, axis=0))

    def test_robust_estimator_sampling_shape_bounds_and_bias_constraint(self) -> None:
        reports = np.array(
            [
                [0.30, 0.70, 2100.0, 3100.0],
                [0.31, 0.69, 2150.0, 3050.0],
                [0.29, 0.72, 2050.0, 3150.0],
                [0.95, 0.05, 6000.0, 15000.0],
                [0.28, 0.71, 2080.0, 3120.0],
            ]
        )
        estimator = RobustBayesianEstimator(num_nodes=5, rng=np.random.default_rng(4))

        samples = estimator.sample(reports, 64)

        self.assertEqual(samples.shape, (64, 4))
        self.assertTrue(np.all(samples >= FEATURE_LOW))
        self.assertTrue(np.all(samples <= FEATURE_HIGH))
        np.testing.assert_allclose(estimator.bias.sum(axis=0), np.zeros(4), atol=1e-8)

    def test_robust_estimator_downweights_outlier_cluster(self) -> None:
        honest_center = np.array([0.30, 0.70, 2100.0, 3100.0])
        reports = np.vstack(
            [
                honest_center,
                honest_center + np.array([0.01, -0.01, 40.0, -50.0]),
                honest_center + np.array([-0.01, 0.02, -35.0, 45.0]),
                np.array([0.95, 0.05, 6000.0, 15000.0]),
                honest_center + np.array([0.005, 0.0, 10.0, 0.0]),
            ]
        )
        estimator = RobustBayesianEstimator(num_nodes=5, rng=np.random.default_rng(5))

        estimate = estimator.estimate(reports)

        robust_error = np.linalg.norm(estimate - honest_center)
        mean_error = np.linalg.norm(reports.mean(axis=0) - honest_center)
        self.assertLess(robust_error, mean_error)


class AgentTests(unittest.TestCase):
    def test_forest_agent_uses_true_context_for_oracle(self) -> None:
        actions = default_action_space()
        agent = ForestBanditAgent(
            name="oracle",
            actions=actions,
            rng=np.random.default_rng(6),
            sees_true_context=True,
        )
        reports = np.zeros((3, 4))
        true_context = np.array([0.2, 0.9, 3000.0, 5000.0])

        decision = agent.select_action(reports, true_context=true_context)

        np.testing.assert_allclose(decision.estimated_context, true_context)
        self.assertIn(decision.action, actions)

    def test_forest_agent_window_size_is_enforced(self) -> None:
        actions = default_action_space()
        agent = ForestBanditAgent(
            name="mean",
            actions=actions,
            rng=np.random.default_rng(7),
            estimator=MeanEstimator(),
            window_size=3,
            min_train_samples=2,
        )
        reports = np.tile(np.array([0.2, 0.9, 3000.0, 5000.0]), (5, 1))

        for idx in range(5):
            decision = agent.select_action(reports)
            agent.observe_reward(reports, decision.action, reward=100.0 + idx)

        self.assertEqual(len(agent.experiences), 3)
        self.assertTrue(agent.is_trained)

    def test_bayesian_agent_buffers_and_sampling(self) -> None:
        actions = default_action_space()
        estimator = RobustBayesianEstimator(num_nodes=5, rng=np.random.default_rng(8))
        agent = BayesianImputationAgent(
            name="bayes",
            actions=actions,
            rng=np.random.default_rng(9),
            estimator=estimator,
            num_imputations=3,
            decision_samples=10,
            window_size=4,
            min_train_samples=2,
        )
        reports = np.tile(np.array([0.2, 0.9, 3000.0, 5000.0]), (5, 1))

        decision = agent.select_action(reports)
        self.assertEqual(agent.last_samples.shape, (10, 4))
        for idx in range(5):
            agent.observe_reward(reports, decision.action, reward=100.0 + idx)

        self.assertEqual(len(agent.buffers), 3)
        self.assertTrue(all(len(buffer) == 4 for buffer in agent.buffers))
        self.assertTrue(agent.is_trained)

    def test_bayesian_uncertainty_agent_trains_on_mean_std_action_features(self) -> None:
        actions = default_action_space()
        estimator = RobustBayesianEstimator(num_nodes=5, rng=np.random.default_rng(18))
        agent = BayesianUncertaintyAgent(
            name="bayes_uncertainty",
            actions=actions,
            rng=np.random.default_rng(19),
            estimator=estimator,
            decision_samples=10,
            window_size=4,
            min_train_samples=2,
        )
        reports = np.tile(np.array([0.2, 0.9, 3000.0, 5000.0]), (5, 1))

        for idx in range(5):
            decision = agent.select_action(reports)
            agent.observe_reward(reports, decision.action, reward=100.0 + idx)

        self.assertEqual(len(agent.experiences), 4)
        self.assertEqual(agent.experiences[-1][0].shape, (12,))
        self.assertTrue(agent.is_trained)


class ExperimentTests(unittest.TestCase):
    def test_experiment_outputs_one_row_per_episode_per_agent(self) -> None:
        actions = default_action_space()
        agents = [
            ForestBanditAgent(
                name="oracle",
                actions=actions,
                rng=np.random.default_rng(10),
                sees_true_context=True,
                min_train_samples=2,
            ),
            ForestBanditAgent(
                name="mean_reports",
                actions=actions,
                rng=np.random.default_rng(11),
                estimator=MeanEstimator(),
                min_train_samples=2,
            ),
        ]
        config = ExperimentConfig(
            episodes=5,
            seed=12,
            num_nodes=5,
            unreliable_fraction=0.2,
            show_progress=False,
        )

        df = run_experiment(config, actions, agents)

        self.assertEqual(len(df), 10)
        self.assertEqual(set(df["agent"]), {"oracle", "mean_reports"})
        self.assertTrue((df[df["agent"] == "oracle"]["context_error_l2"] == 0.0).all())
        for column in ["reward", "regret", "cumulative_reward", "cumulative_regret"]:
            self.assertIn(column, df.columns)

    def test_experiment_can_use_throughput_emulator(self) -> None:
        model = DummyRegressor(strategy="constant", constant=500.0)
        model.fit(np.zeros((2, len(FEATURE_COLUMNS))), np.array([500.0, 500.0]))
        emulator = ThroughputEmulator(
            model=model,
            feature_columns=FEATURE_COLUMNS,
            residual_std=0.0,
        )
        actions = default_action_space()
        agents = [
            ForestBanditAgent(
                name="oracle",
                actions=actions,
                rng=np.random.default_rng(14),
                sees_true_context=True,
                min_train_samples=2,
            )
        ]
        config = ExperimentConfig(
            episodes=3,
            seed=15,
            num_nodes=5,
            unreliable_fraction=0.2,
            show_progress=False,
        )

        df = run_experiment(config, actions, agents, throughput_emulator=emulator)

        self.assertEqual(len(df), 3)
        np.testing.assert_allclose(df["reward"].to_numpy(), np.full(3, 500.0))
        np.testing.assert_allclose(df["expected_reward"].to_numpy(), np.full(3, 500.0))


class RewardEmulatorTests(unittest.TestCase):
    def test_loader_skips_non_trace_csvs_and_normalizes_columns(self) -> None:
        data_dir = ROOT / "external" / "AdaChain" / "src" / "learning" / "data"
        paths = [
            data_dir / "eval1" / "workload_a_ada.csv",
            data_dir / "eval2" / "eval2_ox.csv",
        ]

        df = load_adachain_trace_data(paths)

        self.assertGreater(len(df), 0)
        self.assertEqual(set(FEATURE_COLUMNS).issubset(df.columns), True)
        self.assertIn("throughput", df.columns)
        self.assertEqual(df["source_file"].nunique(), 1)

    def test_emulator_can_be_saved_loaded_and_sample_reward(self) -> None:
        model = DummyRegressor(strategy="constant", constant=123.0)
        model.fit(np.zeros((2, len(FEATURE_COLUMNS))), np.array([123.0, 123.0]))
        emulator = ThroughputEmulator(
            model=model,
            feature_columns=FEATURE_COLUMNS,
            residual_std=0.0,
        )
        path = ROOT / "results" / "test_emulator.joblib"
        save_emulator(path, emulator, {"test": True})

        loaded = load_emulator(path)
        context = np.array([0.2, 0.9, 3000.0, 5000.0])
        action = Action("XOV", 50, True, False)

        self.assertEqual(loaded.predict_expected(context, action), 123.0)
        self.assertEqual(loaded.sample_reward(context, action, np.random.default_rng(13)), 123.0)


if __name__ == "__main__":
    unittest.main()
