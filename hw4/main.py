import numpy as np
from matplotlib import pyplot as plt

from utils import q_learning

if __name__ == "__main__":
    alpha = 0.1
    eps = 0.1
    n_episodes = 500

    n_states = 37  # 0-start, -1-terminal
    n_actions = 4  # 0-up, 1-down, 2-left, 3-right

    state_transition_mat = np.vstack([np.arange(n_states)] * n_actions).T
    state_transition_mat[:, 0] += 1
    state_transition_mat[:, 1] -= 1
    state_transition_mat[:, 2] -= 3
    state_transition_mat[:, 3] += 3

    top_row = np.arange(3, 37, 3)
    state_transition_mat[top_row, 0] = top_row

    left_column = np.arange(4)
    state_transition_mat[left_column, 2] = left_column

    right_column = np.arange(34, 37)
    state_transition_mat[right_column, 3] = right_column

    bottom_row = np.arange(4, 34, 3)
    state_transition_mat[bottom_row, 1] = 0

    state_transition_mat[0, 1] = 0
    state_transition_mat[0, 3] = 0
    state_transition_mat[34, 1] = -1

    reward_mat = -np.ones((n_states, n_actions))
    reward_mat[bottom_row, 1] = -100.0
    reward_mat[0, 3] = -100.0

    accumulated_rewards = q_learning(
        state_transition_mat=state_transition_mat,
        reward_mat=reward_mat,
        alpha=alpha,
        eps=eps,
        n_episodes=n_episodes,
    )

    plt.plot(np.arange(1, n_episodes + 1), accumulated_rewards)

    plt.xlabel("Episodes")
    plt.ylabel("Accumulated Reward")

    plt.grid()
    plt.show()
