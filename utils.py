from abc import ABC, abstractmethod

import numpy as np


class MultiArmedBandit:
    def __init__(self, n_arms):
        self.mean_rewards = np.random.randn(n_arms)

    def take_action(self, k):
        return np.random.normal(self.mean_rewards[k], 1)


class Policy(ABC):
    EPS = 1e-6

    def __init__(self, n_arms, size):
        self.n_arms = n_arms
        self.size = size

    @abstractmethod
    def choose_action(self):
        ...

    @abstractmethod
    def update(self, reward, action):
        ...

    @classmethod
    def random_argmax(cls, a, **kwargs):
        random_mat = cls.EPS + np.random.random(a.shape)
        multiple_maxes = np.isclose(a, a.max(**kwargs, keepdims=True))
        return np.argmax(random_mat * multiple_maxes, **kwargs)

    @staticmethod
    def select_from_rows(arr, idx):
        return np.take_along_axis(
            arr=arr,
            indices=idx.reshape(-1, 1),
            axis=1,
        ).flatten()

    @staticmethod
    def put_to_rows(arr, idx, values):
        np.put_along_axis(
            arr=arr,
            indices=idx.reshape(-1, 1),
            values=values.reshape(-1, 1),
            axis=1,
        )


class UCBPolicy(Policy):
    def __init__(self, c, n_arms, size):
        super().__init__(n_arms=n_arms, size=size)
        self.c = c

        self.counts = np.zeros((size, n_arms))
        self.sample_mean_rewards = np.zeros((size, n_arms))
        self.uncertainties = np.zeros((size, n_arms))

    def choose_action(self):
        upper_confidence_bounds = self.sample_mean_rewards + self.uncertainties
        return self.random_argmax(upper_confidence_bounds, axis=1)

    def update(self, reward, action):
        # update counts
        selected_counts = self.select_from_rows(self.counts, action) + 1
        self.put_to_rows(self.counts, action, selected_counts)

        # update rewards
        selected_rewards = self.select_from_rows(self.sample_mean_rewards, action)
        deviation = reward - selected_rewards
        new_rewards = selected_rewards + deviation / selected_counts
        self.put_to_rows(self.sample_mean_rewards, action, new_rewards)

        # update uncertainties
        time_factor = np.log(np.sum(self.counts, axis=1, keepdims=True))
        self.uncertainties = self.c * np.sqrt(time_factor / (self.counts + self.EPS))


class GradientPolicy(Policy):
    def __init__(self, alpha, n_arms, size):
        super().__init__(n_arms=n_arms, size=size)
        self.alpha = alpha

        self.rng = np.random.default_rng()
        self.scores = np.zeros((size, n_arms))
        self.count = 0
        self.sample_mean_rewards = np.zeros(size)

    def choose_action(self):
        probs = self.softmax_rows(self.scores)
        return self.rng.multinomial(1, probs).argmax(axis=1)

    def update(self, reward, action):
        # update rewards
        self.count += 1
        deviation = reward - self.sample_mean_rewards
        self.sample_mean_rewards += deviation / self.count

        # update scores
        probs = self.softmax_rows(self.scores)
        baseline_deviation = (reward - self.sample_mean_rewards).reshape(-1, 1)
        grad = np.eye(self.n_arms)[action] - probs
        self.scores += self.alpha * baseline_deviation * grad

    @staticmethod
    def softmax_rows(a):
        shifted_exp = np.exp(a - np.max(a, axis=1, keepdims=True))
        return shifted_exp / np.sum(shifted_exp, axis=1, keepdims=True)


class Simulator:
    def __init__(
        self,
        bandit: MultiArmedBandit,
        policy: Policy,
        time_horizon,
    ):
        assert bandit.mean_rewards.shape[0] == policy.n_arms

        self.bandit = bandit
        self.policy = policy
        self.time_horizon = time_horizon

        self.max_reward = np.max(self.bandit.mean_rewards)
        self.mean_regrets = np.zeros(time_horizon)
        self.se_regrets = np.zeros(time_horizon)

    def simulate(self):
        for i in range(self.time_horizon):
            k = self.policy.choose_action()
            reward = self.bandit.take_action(k=k)

            self.policy.update(reward=reward, action=k)
            self.mean_regrets[i] = self.max_reward - np.mean(reward)
            self.se_regrets[i] = np.std(reward) / np.sqrt(reward.shape[0])
