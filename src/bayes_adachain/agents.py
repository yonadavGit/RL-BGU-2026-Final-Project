from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .actions import Action, make_feature_vector
from .estimators import ContextEstimator, RobustBayesianEstimator


Experience = Tuple[np.ndarray, float]


def _make_forest(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=2,
        random_state=seed,
    )


def _make_uncertainty_feature_vector(mean: np.ndarray, std: np.ndarray, action: Action) -> np.ndarray:
    return np.concatenate([np.asarray(mean, dtype=float), np.asarray(std, dtype=float), action.features])


@dataclass
class AgentDecision:
    action: Action
    estimated_context: np.ndarray
    predicted_values: np.ndarray


class Agent:
    name: str

    def select_action(self, reports: np.ndarray, true_context: Optional[np.ndarray] = None) -> AgentDecision:
        raise NotImplementedError

    def observe_reward(self, reports: np.ndarray, action: Action, reward: float) -> None:
        raise NotImplementedError


@dataclass
class ForestBanditAgent(Agent):
    name: str
    actions: List[Action]
    rng: np.random.Generator
    estimator: Optional[ContextEstimator] = None
    window_size: int = 80
    min_train_samples: int = 8
    sees_true_context: bool = False
    experiences: Deque[Experience] = field(default_factory=deque)
    forest: RandomForestRegressor = field(init=False)
    last_context: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.forest = _make_forest(int(self.rng.integers(0, 2**31 - 1)))

    @property
    def is_trained(self) -> bool:
        return len(self.experiences) >= self.min_train_samples

    def _context(self, reports: np.ndarray, true_context: Optional[np.ndarray]) -> np.ndarray:
        if self.sees_true_context:
            if true_context is None:
                raise ValueError("oracle agent requires true_context")
            return np.asarray(true_context, dtype=float)
        if self.estimator is None:
            raise ValueError("non-oracle agent requires an estimator")
        return self.estimator.estimate(reports)

    def _fit(self) -> None:
        if not self.is_trained:
            return
        x_train = np.vstack([row[0] for row in self.experiences])
        y_train = np.array([row[1] for row in self.experiences], dtype=float)
        self.forest.fit(x_train, y_train)

    def select_action(self, reports: np.ndarray, true_context: Optional[np.ndarray] = None) -> AgentDecision:
        context = self._context(reports, true_context)
        self.last_context = context

        if not self.is_trained:
            action = self.rng.choice(self.actions)
            values = np.zeros(len(self.actions), dtype=float)
            return AgentDecision(action=action, estimated_context=context, predicted_values=values)

        candidates = np.vstack([make_feature_vector(context, action) for action in self.actions])
        values = self.forest.predict(candidates)
        best = int(np.argmax(values))
        return AgentDecision(action=self.actions[best], estimated_context=context, predicted_values=values)

    def observe_reward(self, reports: np.ndarray, action: Action, reward: float) -> None:
        if self.last_context is None:
            context = self.estimator.estimate(reports) if self.estimator is not None else np.mean(reports, axis=0)
        else:
            context = self.last_context
        self.experiences.append((make_feature_vector(context, action), float(reward)))
        while len(self.experiences) > self.window_size:
            self.experiences.popleft()
        self._fit()


@dataclass
class BayesianImputationAgent(Agent):
    name: str
    actions: List[Action]
    rng: np.random.Generator
    estimator: RobustBayesianEstimator
    num_imputations: int = 5
    decision_samples: int = 32
    window_size: int = 80
    min_train_samples: int = 8
    buffers: List[Deque[Experience]] = field(init=False)
    forests: List[RandomForestRegressor] = field(init=False)
    last_samples: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.buffers = [deque() for _ in range(self.num_imputations)]
        self.forests = [_make_forest(int(self.rng.integers(0, 2**31 - 1))) for _ in range(self.num_imputations)]

    @property
    def is_trained(self) -> bool:
        return all(len(buffer) >= self.min_train_samples for buffer in self.buffers)

    def _fit(self) -> None:
        if not self.is_trained:
            return
        for forest, buffer in zip(self.forests, self.buffers):
            x_train = np.vstack([row[0] for row in buffer])
            y_train = np.array([row[1] for row in buffer], dtype=float)
            forest.fit(x_train, y_train)

    def select_action(self, reports: np.ndarray, true_context: Optional[np.ndarray] = None) -> AgentDecision:
        samples = self.estimator.sample(reports, self.decision_samples)
        self.last_samples = samples
        estimated_context = np.mean(samples, axis=0)

        if not self.is_trained:
            action = self.rng.choice(self.actions)
            values = np.zeros(len(self.actions), dtype=float)
            return AgentDecision(action=action, estimated_context=estimated_context, predicted_values=values)

        values = []
        for action in self.actions:
            action_rows = np.vstack([make_feature_vector(sample, action) for sample in samples])
            per_forest = [forest.predict(action_rows).mean() for forest in self.forests]
            values.append(float(np.mean(per_forest)))
        values_arr = np.array(values, dtype=float)
        return AgentDecision(
            action=self.actions[int(np.argmax(values_arr))],
            estimated_context=estimated_context,
            predicted_values=values_arr,
        )

    def observe_reward(self, reports: np.ndarray, action: Action, reward: float) -> None:
        samples = self.estimator.draw_posterior(self.num_imputations)
        for sample, buffer in zip(samples, self.buffers):
            buffer.append((make_feature_vector(sample, action), float(reward)))
            while len(buffer) > self.window_size:
                buffer.popleft()
        self._fit()


@dataclass
class BayesianUncertaintyAgent(Agent):
    name: str
    actions: List[Action]
    rng: np.random.Generator
    estimator: RobustBayesianEstimator
    decision_samples: int = 32
    window_size: int = 80
    min_train_samples: int = 8
    experiences: Deque[Experience] = field(default_factory=deque)
    forest: RandomForestRegressor = field(init=False)
    last_mean: Optional[np.ndarray] = None
    last_std: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.forest = _make_forest(int(self.rng.integers(0, 2**31 - 1)))

    @property
    def is_trained(self) -> bool:
        return len(self.experiences) >= self.min_train_samples

    def _fit(self) -> None:
        if not self.is_trained:
            return
        x_train = np.vstack([row[0] for row in self.experiences])
        y_train = np.array([row[1] for row in self.experiences], dtype=float)
        self.forest.fit(x_train, y_train)

    def select_action(self, reports: np.ndarray, true_context: Optional[np.ndarray] = None) -> AgentDecision:
        samples = self.estimator.sample(reports, self.decision_samples)
        posterior_mean = self.estimator.posterior_mean.copy()
        posterior_std = self.estimator.posterior_std.copy()
        self.last_mean = posterior_mean
        self.last_std = posterior_std

        if not self.is_trained:
            action = self.rng.choice(self.actions)
            values = np.zeros(len(self.actions), dtype=float)
            return AgentDecision(action=action, estimated_context=posterior_mean, predicted_values=values)

        values = []
        for action in self.actions:
            action_rows = np.vstack(
                [_make_uncertainty_feature_vector(sample, posterior_std, action) for sample in samples]
            )
            values.append(float(self.forest.predict(action_rows).mean()))
        values_arr = np.array(values, dtype=float)
        return AgentDecision(
            action=self.actions[int(np.argmax(values_arr))],
            estimated_context=posterior_mean,
            predicted_values=values_arr,
        )

    def observe_reward(self, reports: np.ndarray, action: Action, reward: float) -> None:
        if self.last_mean is None or self.last_std is None:
            self.estimator.estimate(reports)
            posterior_mean = self.estimator.posterior_mean.copy()
            posterior_std = self.estimator.posterior_std.copy()
        else:
            posterior_mean = self.last_mean
            posterior_std = self.last_std
        features = _make_uncertainty_feature_vector(posterior_mean, posterior_std, action)
        self.experiences.append((features, float(reward)))
        while len(self.experiences) > self.window_size:
            self.experiences.popleft()
        self._fit()
