import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import (
    MultiArmedBandit,
    Simulator,
    UCBPolicy,
    GradientPolicy,
    EpsGreedyPolicy,
)

if __name__ == "__main__":
    n_arms = 10
    time_horizon = 1000
    n_runs = 5000

    bandit = MultiArmedBandit(n_arms=n_arms, size=n_runs)

    # epsilon greedy
    eps_grid = np.logspace(-4, 0, 100)
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

    plt.plot(eps_grid, regrets)
    plt.fill_between(
        eps_grid, regrets - regrets_se, regrets + regrets_se, color="b", alpha=0.2
    )

    plt.title("Performance of Epsilon Greedy Algorithm")
    plt.xlabel("eps")
    plt.ylabel("Regret")

    plt.grid()
    plt.show()

    # UCB
    c_grid = np.logspace(-2, 0.6, 100)
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

    plt.plot(c_grid, regrets)
    plt.fill_between(
        c_grid, regrets - regrets_se, regrets + regrets_se, color="b", alpha=0.2
    )

    plt.title("Performance of UCB Algorithm")
    plt.xlabel("c")
    plt.ylabel("Regret")

    plt.grid()
    plt.show()

    # gradient bandit
    alpha_grid = np.logspace(-2, 0.6, 100)
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

    plt.plot(alpha_grid, regrets)
    plt.fill_between(
        alpha_grid, regrets - regrets_se, regrets + regrets_se, color="b", alpha=0.2
    )

    plt.title("Performance of Gradient Bandit Algorithm")
    plt.xlabel("alpha")
    plt.ylabel("Regret")

    plt.grid()
    plt.show()
