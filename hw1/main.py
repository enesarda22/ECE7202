import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import (
    MultiArmedBandit,
    Simulator,
    UCBPolicy,
    GradientPolicy,
    EpsGreedyPolicy,
    ExploreFirstGreedyPolicy,
    plot,
)


if __name__ == "__main__":
    n_arms = 10
    time_horizon = 1000
    n_runs = 5000

    bandit = MultiArmedBandit(n_arms=n_arms, size=n_runs)

    # explore first greedy
    n_grid = np.arange(1, 101)
    regrets = np.empty(0)
    regrets_se = np.empty(0)
    for n in tqdm(n_grid, "Explore First Greedy"):
        policy = ExploreFirstGreedyPolicy(n=n, n_arms=n_arms, size=n_runs)
        simulator = Simulator(
            bandit=bandit,
            policy=policy,
            time_horizon=time_horizon,
        )
        simulator.simulate()
        regrets = np.append(regrets, np.sum(simulator.mean_regrets))
        regrets_se = np.append(regrets_se, np.sum(simulator.se_regrets))

    plot(n_grid, regrets, regrets_se, "Explore First Greedy Algorithm", "N")

    # epsilon greedy
    eps_grid = np.linspace(1e-4, 0.5, 100)
    regrets = np.empty(0)
    regrets_se = np.empty(0)
    for eps in tqdm(eps_grid, "Eps Greedy"):
        policy = EpsGreedyPolicy(eps=eps, n_arms=n_arms, size=n_runs)
        simulator = Simulator(
            bandit=bandit,
            policy=policy,
            time_horizon=time_horizon,
        )
        simulator.simulate()
        regrets = np.append(regrets, np.sum(simulator.mean_regrets))
        regrets_se = np.append(regrets_se, np.sum(simulator.se_regrets))

    plot(eps_grid, regrets, regrets_se, "Epsilon Greedy Algorithm", "eps")
    best_eps = eps_grid[np.argmin(regrets)]

    # UCB
    c_grid = np.linspace(1e-2, 3, 100)
    regrets = np.empty(0)
    regrets_se = np.empty(0)
    for c in tqdm(c_grid, "UCB"):
        policy = UCBPolicy(c=c, n_arms=n_arms, size=n_runs)
        simulator = Simulator(
            bandit=bandit,
            policy=policy,
            time_horizon=time_horizon,
        )
        simulator.simulate()
        regrets = np.append(regrets, np.sum(simulator.mean_regrets))
        regrets_se = np.append(regrets_se, np.sum(simulator.se_regrets))

    plot(c_grid, regrets, regrets_se, "UCB Algorithm", "c")
    best_c = c_grid[np.argmin(regrets)]

    # gradient bandit
    alpha_grid = np.linspace(1e-2, 4, 100)
    regrets = np.empty(0)
    regrets_se = np.empty(0)
    for alpha in tqdm(alpha_grid, "Gradient Bandit"):
        policy = GradientPolicy(alpha=alpha, n_arms=n_arms, size=n_runs)
        simulator = Simulator(
            bandit=bandit,
            policy=policy,
            time_horizon=time_horizon,
        )
        simulator.simulate()
        regrets = np.append(regrets, np.sum(simulator.mean_regrets))
        regrets_se = np.append(regrets_se, np.sum(simulator.se_regrets))

    plot(alpha_grid, regrets, regrets_se, "Gradient Bandit", "alpha")

    # compare eps-greedy and ucb
    time_horizon = 10000

    policy = EpsGreedyPolicy(eps=best_eps, n_arms=n_arms, size=n_runs)
    simulator = Simulator(
        bandit=bandit,
        policy=policy,
        time_horizon=time_horizon,
    )
    simulator.simulate()
    eps_greedy_regrets = np.cumsum(simulator.mean_regrets)
    eps_greedy_regrets_se = np.cumsum(simulator.se_regrets)

    policy = UCBPolicy(c=best_c, n_arms=n_arms, size=n_runs)
    simulator = Simulator(
        bandit=bandit,
        policy=policy,
        time_horizon=time_horizon,
    )
    simulator.simulate()
    ucb_regrets = np.cumsum(simulator.mean_regrets)
    ucb_regrets_se = np.cumsum(simulator.se_regrets)

    time_horizon_grid = np.arange(1, time_horizon + 1)
    plt.plot(
        time_horizon_grid,
        eps_greedy_regrets,
        "r",
        label=f"Eps-Greedy, eps={best_eps:.2f}",
    )
    plt.fill_between(
        time_horizon_grid,
        eps_greedy_regrets - eps_greedy_regrets_se,
        eps_greedy_regrets + eps_greedy_regrets_se,
        color="r",
        alpha=0.2,
    )

    plt.plot(time_horizon_grid, ucb_regrets, "b", label=f"UCB, c={best_c:.2f}")
    plt.fill_between(
        time_horizon_grid,
        ucb_regrets - ucb_regrets_se,
        ucb_regrets + ucb_regrets_se,
        color="b",
        alpha=0.2,
    )

    plt.title("Regret Comparison of UCB and Eps-Greedy Methods")
    plt.xlabel("Time Horizon")
    plt.ylabel("Regret")

    plt.legend()
    plt.grid()
    plt.show()
