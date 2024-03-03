import numpy as np
from tqdm import tqdm


def q_learning(
    state_transition_mat: np.ndarray,
    reward_mat: np.ndarray,
    alpha: float,
    eps: float,
    n_episodes: int,
    n_experiments: int,
):
    n_states, n_actions = state_transition_mat.shape

    q = np.random.randn(n_experiments, n_states + 1, n_actions)
    q[:, -1, :] = 0.0  # terminal state

    accumulated_rewards = np.zeros((n_experiments, n_episodes))
    for i in tqdm(range(n_episodes), "Q-Learning Episodes"):
        s = np.zeros(n_experiments, dtype=int)
        ongoing_exp = s != -1
        while np.any(ongoing_exp):
            s_trun = s[ongoing_exp]
            a = _choose_eps_greedy(values=q[ongoing_exp, s_trun, :], eps=eps)

            s_hat = state_transition_mat[s_trun, a]
            r = reward_mat[s_trun, a]

            target = r + np.max(q[ongoing_exp, s_hat, :], axis=1)
            q[ongoing_exp, s_trun, a] += alpha * (target - q[ongoing_exp, s_trun, a])

            s[ongoing_exp] = s_hat
            accumulated_rewards[ongoing_exp, i] += r

            ongoing_exp = s != -1

    return accumulated_rewards


def sarsa(
    state_transition_mat: np.ndarray,
    reward_mat: np.ndarray,
    alpha: float,
    eps: float,
    n_episodes: int,
    n_experiments: int,
):
    n_states, n_actions = state_transition_mat.shape

    q = np.random.randn(n_experiments, n_states + 1, n_actions)
    q[:, -1, :] = 0.0  # terminal state

    accumulated_rewards = np.zeros((n_experiments, n_episodes))
    for i in tqdm(range(n_episodes), "SARSA Episodes"):
        s = np.zeros(n_experiments, dtype=int)
        ongoing_exp = s != -1
        a = _choose_eps_greedy(values=q[ongoing_exp, s, :], eps=eps)
        while np.any(ongoing_exp):
            s_trun = s[ongoing_exp]

            s_hat = state_transition_mat[s_trun, a]
            a_hat = _choose_eps_greedy(values=q[ongoing_exp, s_hat, :], eps=eps)
            r = reward_mat[s_trun, a]

            target = r + q[ongoing_exp, s_hat, a_hat]
            q[ongoing_exp, s_trun, a] += alpha * (target - q[ongoing_exp, s_trun, a])

            s[ongoing_exp] = s_hat
            a = a_hat[s_hat != -1]
            accumulated_rewards[ongoing_exp, i] += r

            ongoing_exp = s != -1

    return accumulated_rewards


def _choose_eps_greedy(values, eps):
    n_experiments, n_actions = values.shape
    actions = np.empty(n_experiments, dtype=int)

    choose_random = np.random.random(n_experiments) < eps
    actions[choose_random] = np.random.randint(n_actions, size=np.sum(choose_random))
    actions[~choose_random] = np.argmax(values[~choose_random, :], axis=1)

    return actions
