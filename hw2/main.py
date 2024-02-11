import numpy as np

from utils import (
    iterative_policy_evaluation,
    order_independent_iterative_policy_evaluation,
)

if __name__ == "__main__":
    np.random.seed(42)
    n_states = 9  # not including the terminal state
    n_actions = 4  # 0-up, 1-down, 2-left, 3-right

    reward_mat = -np.ones((n_states, n_actions))  # -1 for every move
    reward_mat[7, 3] = 10.0  # reaching goal
    reward_mat[6, 3] = -10.0  # reaching trap
    reward_mat[8, 0] = -10.0  # reaching trap

    state_transition_mat = np.vstack([np.arange(n_states)] * n_actions).T
    state_transition_mat[0, 0] = 1
    state_transition_mat[0, 3] = 3

    state_transition_mat[1, 0] = 2
    state_transition_mat[1, 1] = 0

    state_transition_mat[2, 1] = 1
    state_transition_mat[2, 3] = 4

    state_transition_mat[3, 2] = 0
    state_transition_mat[3, 3] = 5

    state_transition_mat[4, 2] = 2
    state_transition_mat[4, 3] = 7

    state_transition_mat[5, 0] = 6
    state_transition_mat[5, 2] = 3
    state_transition_mat[5, 3] = 8

    state_transition_mat[6, 0] = 7
    state_transition_mat[6, 1] = 5
    state_transition_mat[6, 3] = 9

    state_transition_mat[7, 1] = 6
    state_transition_mat[7, 2] = 4
    state_transition_mat[7, 3] = 9

    state_transition_mat[8, 0] = 9
    state_transition_mat[8, 2] = 5

    # random policy
    policy_mat = np.ones((n_states, n_actions)) * 0.25

    # iterative policy evaluation
    th = 1e-6

    state_order = np.arange(n_states)
    v, i = iterative_policy_evaluation(
        policy_mat=policy_mat,
        reward_mat=reward_mat,
        state_transition_mat=state_transition_mat,
        state_order=state_order,
        th=th,
    )
    print(f"It took {i} steps to converge in order {state_order}")
    print(f"Value function = {np.array2string(v[-1], precision=2)}")

    v, i = order_independent_iterative_policy_evaluation(
        policy_mat=policy_mat,
        reward_mat=reward_mat,
        state_transition_mat=state_transition_mat,
        state_order=state_order,
        th=th,
    )
    print(f"It took {i} steps with order-independent algorithm.")
    print(f"Value function = {np.array2string(v[-1], precision=2)}\n")

    state_order = np.flip(state_order)
    v, i = iterative_policy_evaluation(
        policy_mat=policy_mat,
        reward_mat=reward_mat,
        state_transition_mat=state_transition_mat,
        state_order=state_order,
        th=th,
    )
    print(f"It took {i} steps to converge in order {state_order}")
    print(f"Value function = {np.array2string(v[-1], precision=2)}")

    v, i = order_independent_iterative_policy_evaluation(
        policy_mat=policy_mat,
        reward_mat=reward_mat,
        state_transition_mat=state_transition_mat,
        state_order=state_order,
        th=th,
    )
    print(f"It took {i} steps with order-independent algorithm.")
    print(f"Value function = {np.array2string(v[-1], precision=2)}\n")

    state_order = np.random.choice(state_order, n_states, replace=False)
    v, i = iterative_policy_evaluation(
        policy_mat=policy_mat,
        reward_mat=reward_mat,
        state_transition_mat=state_transition_mat,
        state_order=state_order,
        th=th,
    )
    print(f"It took {i} steps to converge in order {state_order}")
    print(f"Value function = {np.array2string(v[-1], precision=2)}\n")

    print("Finding greedy policies for different value functions:")
    indices = np.linspace(1, len(v) - 1, 5)
    for i in indices:
        v_f = v[int(i)]
        greedy_policy = np.argmax(v_f[state_transition_mat] + reward_mat, axis=1)

        print(f"At iteration {int(i)}:")
        print(f"Value function = {np.array2string(v_f, precision=2)}")
        print(f"Greedy policy = {greedy_policy}\n")
