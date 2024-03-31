import numpy as np

from utils import (
    tabular_prediction,
    linear_prediction,
    plot,
)

if __name__ == "__main__":
    np.random.seed(42)

    n_episodes = 5000

    # tabular case
    v_true = tabular_prediction(alpha=0.9, n_episodes=n_episodes, n_runs=1000)

    # state aggregation
    n = 10
    bins = np.repeat(np.arange(n), 1000 / n)

    def state_aggregation(s):
        features = np.zeros(n)
        features[bins[s - 1]] = 1.0
        return features

    v_agg = linear_prediction(
        dim=n,
        transform_func=state_aggregation,
        alpha=2e-4,
        n_episodes=n_episodes,
        name="State Aggregation",
    )
    plot(v_true=v_true, v_pred=v_agg, label="State Aggregation")

    # polynomial basis
    n = 5

    def polynomial_transform(s):
        s = (s - 500.5) / 1000
        features = np.ones(n + 1, dtype=float)
        for i in range(1, n + 1):
            features[i] = s**i
        return features

    v_poly = linear_prediction(
        dim=n + 1,
        transform_func=polynomial_transform,
        alpha=1e-4,
        n_episodes=n_episodes,
        name="Polynomial",
    )
    plot(v_true=v_true, v_pred=v_poly, label="Polynomial Basis")

    # fourier basis
    n = 5

    def fourier_transform(s):
        s = (s - 1) / 999
        features = np.ones(n + 1, dtype=float)
        for i in range(1, n + 1):
            features[i] = np.cos(np.pi * s * i)
        return features

    v_fourier = linear_prediction(
        dim=n + 1,
        transform_func=fourier_transform,
        alpha=5e-5,
        n_episodes=n_episodes,
        name="Fourier",
    )
    plot(v_true=v_true, v_pred=v_fourier, label="Fourier Basis")
