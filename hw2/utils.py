import numpy as np


def iterative_policy_evaluation(
    policy_mat,
    reward_mat,
    state_transition_mat,
    state_order,
    th,
):
    i = 0
    v = np.random.randn(policy_mat.shape[0] + 1)
    v[-1] = 0.0

    value_functions = [np.copy(v)]
    while True:
        i += 1
        diff = 0.0
        for s in state_order:
            old_v = v[s]
            v[s] = policy_mat[s, :] @ (reward_mat[s, :] + v[state_transition_mat[s, :]])
            diff = max(diff, abs(old_v - v[s]))

        value_functions.append(np.copy(v))
        if diff < th:
            return value_functions, i


def order_independent_iterative_policy_evaluation(
    policy_mat,
    reward_mat,
    state_transition_mat,
    state_order,
    th,
):
    i = 0
    v = np.random.randn(policy_mat.shape[0] + 1)
    v[-1] = 0.0

    value_functions = [np.copy(v)]
    while True:
        i += 1
        old_v = np.copy(v)
        for s in state_order:
            v[s] = policy_mat[s, :] @ (
                reward_mat[s, :] + old_v[state_transition_mat[s, :]]
            )

        diff = np.max(np.abs(old_v - v))
        value_functions.append(np.copy(v))
        if diff < th:
            return value_functions, i
