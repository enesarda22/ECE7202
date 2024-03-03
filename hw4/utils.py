import numpy as np


def q_learning(
    state_transition_mat: np.ndarray,
    reward_mat: np.ndarray,
    alpha: float,
    eps: float,
    n_episodes: int,
):
    n_states, n_actions = state_transition_mat.shape

    q = np.random.randn(n_states + 1, n_actions)
    q[-1, :] = 0.0  # terminal state

    accumulated_rewards = np.zeros(n_episodes)
    for i in range(n_episodes):
        s = 0
        while s != -1:
            a = _choose_eps_greedy(values=q[s, :], eps=eps)
            s_hat = state_transition_mat[s, a]
            r = reward_mat[s, a]

            target = r + np.max(q[s_hat, :])
            q[s, a] += alpha * (target - q[s, a])

            s = s_hat
            accumulated_rewards[i] += r

    return accumulated_rewards


def _choose_eps_greedy(values, eps):
    choose_random = np.random.random() < eps
    if choose_random:
        return np.random.randint(len(values))
    else:
        return np.argmax(values)
