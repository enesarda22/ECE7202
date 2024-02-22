import numpy as np
from matplotlib import pyplot as plt


def value_iteration(goal: int, p_h: float, th: float):
    v = np.random.randn(goal)
    v[0] = 0.0  # terminal state

    def argmax(x):
        # breaks ties by choosing the first argument
        return np.argmax(np.isclose(x, np.max(x), atol=th, rtol=0))

    def action_values(state):
        a = np.arange(1, min(state, goal - state) + 1)

        win_states = state + a
        win_states[win_states == goal] = 0  # send to the terminal state

        win_return = (state + a == goal).astype(int) + v[win_states]
        loss_return = v[state - a]

        return p_h * win_return + (1 - p_h) * loss_return

    while True:
        diff = 0.0
        for s in range(1, goal):
            old_v = v[s]
            v[s] = np.max(action_values(s))

            diff = max(diff, abs(old_v - v[s]))

        if diff < th:
            policy = np.empty(goal - 1, dtype=int)
            for s in range(1, goal):
                policy[s - 1] = argmax(action_values(s)) + 1

            return v, policy


def every_visit_mc_prediction(policy: np.array, goal: int, p_h: float, th: float):
    v = np.random.randn(goal - 1)
    state_counts = np.zeros(goal - 1, dtype=int)
    v_var = np.ones(goal - 1)  # to track the convergence
    non_converged_count = 0  # to track the non-converged states

    def did_win():
        return np.random.random() < p_h

    def generate_episode(initial_states: np.ndarray):
        s_arr = np.empty(0, dtype=int)
        s = np.random.choice(initial_states)

        while True:
            s_arr = np.append(s_arr, s)
            a = policy[s - 1]
            s = (s + a) if did_win() else (s - a)

            if s == goal:
                return s_arr, 1

            if s == 0:
                return s_arr, 0

    while True:
        # check convergence
        variance_not_reliable = state_counts < 10000
        variance_not_low = np.sqrt(v_var / (state_counts + 1e-8)) > th
        non_converged_idx = variance_not_reliable | variance_not_low
        if ~np.any(non_converged_idx):
            return v

        if non_converged_count != np.sum(non_converged_idx):
            non_converged_count = np.sum(non_converged_idx)
            print(f"{non_converged_count} states left to converge.")

        # generate an episode for non-converged states
        non_converged_states = np.where(non_converged_idx)[0] + 1
        states, return_ = generate_episode(initial_states=non_converged_states)

        # update counts
        unique_states, counts = np.unique(states, return_counts=True)
        state_counts[unique_states - 1] += counts

        # update value estimates
        diff = return_ - v[unique_states - 1]
        step_size = counts / state_counts[unique_states - 1]
        update = step_size * diff
        v[unique_states - 1] += update

        # update the sample variance of value estimates
        v_var[unique_states - 1] += step_size * (
            np.mean((return_ - v[unique_states - 1]) ** 2)
            - v_var[unique_states - 1]
            + (state_counts[unique_states - 1] / counts - 1) * update**2
        )


def plot(goal: int, policy: np.array, p_h: float):
    s = np.arange(1, goal)
    plt.plot(s, policy)

    plt.title(f"Optimal Policy When p_h = {p_h}")
    plt.xlabel("Current Capital")
    plt.ylabel("Optimal Stake")

    if p_h < 0.5:
        yticks = np.linspace(0, np.max(policy), 9)
        yticks[0] = 1
        plt.yticks(yticks)

    plt.xticks(np.linspace(0, goal, 9))
    plt.grid()
    plt.show()
