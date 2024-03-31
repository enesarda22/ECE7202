import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def tabular_prediction(alpha, n_episodes, n_runs):
    v = np.random.randn(n_runs, 1001)
    v[:, 0] = 0.0  # terminal state

    for _ in tqdm(range(n_episodes), desc="Tabular Prediction"):
        s = 500 * np.ones(n_runs, dtype=int)
        not_ended = s != 0
        while np.any(not_ended):
            r, s_hat = _take_action(s)

            diff = (r + v[not_ended, s_hat]) - v[not_ended, s]
            v[not_ended, s] += alpha * diff

            not_ended[not_ended] = s_hat != 0
            s = s_hat[s_hat != 0]

    v = np.mean(v, axis=0)
    return v


def linear_prediction(dim, transform_func, alpha, n_episodes, name):
    w = np.random.randn(dim)
    for _ in tqdm(range(n_episodes), desc=f"{name} Prediction"):
        return_ = 0

        s = 500
        visited_states = []
        while s != 0:
            r, s_hat = _take_action(s)
            return_ += r[0]
            s = s_hat[0]

            if s != 0:
                visited_states.append(s)

        for state in visited_states:
            x = transform_func(state)
            diff = return_ - (x @ w)
            w += alpha * diff * x

    v = np.array([transform_func(i) @ w for i in range(1, 1001)])
    return v


def _take_action(s):
    if isinstance(s, (int, np.integer)):
        s = np.array([s])

    n = len(s)
    choose_right = np.random.random(n) > 0.5

    s_hat = np.empty(n, dtype=int)
    s_hat[choose_right] = np.random.randint(s[choose_right] + 1, s[choose_right] + 101)
    s_hat[~choose_right] = np.random.randint(s[~choose_right] - 100, s[~choose_right])

    right_end_reached = s_hat > 1000
    left_end_reached = s_hat < 1

    r = np.zeros(n)
    r[right_end_reached] = 1.0
    r[left_end_reached] = -1.0

    s_hat[right_end_reached | left_end_reached] = 0
    return r, s_hat


def plot(v_true, v_pred, label):
    states = np.arange(1, 1001)
    plt.plot(states, v_true[1:], label="True Value Function")
    plt.plot(states, v_pred, label=label)

    plt.xlabel("State")
    plt.ylabel("Value")
    plt.title("State Value Function")

    plt.legend()
    plt.grid()
    plt.show()

    mse = np.mean((v_true[1:] - v_pred) ** 2)
    print(f"MSE = {mse:.2e}")
