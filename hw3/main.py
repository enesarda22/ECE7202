import numpy as np
from matplotlib import pyplot as plt

from hw3.utils import value_iteration, plot, every_visit_mc_prediction

if __name__ == "__main__":
    np.random.seed(42)
    th = 1e-6  # threshold to stop the value iteration
    goal = 100

    p_h = 0.25
    _, policy = value_iteration(goal=goal, p_h=p_h, th=th)
    plot(goal=goal, policy=policy, p_h=p_h)

    p_h = 0.55
    v1, policy = value_iteration(goal=goal, p_h=p_h, th=th)
    plot(goal=goal, policy=policy, p_h=p_h)

    th = 1e-2
    v2 = every_visit_mc_prediction(policy=policy, goal=goal, p_h=p_h, th=th)

    states = np.arange(1, 100)
    plt.loglog(states, v1[1:], "x-", label="VI evaluation")
    plt.loglog(states, v2, ".-", label="MC evaluation")

    plt.xlabel("States (Capital")
    plt.ylabel("Value (Probability of Winning)")
    plt.title("Evaluated Value Functions With Different Methods")

    plt.legend()
    plt.grid(which="both")
    plt.tight_layout()
    plt.show()

    print(f"Value Iteration Output: {np.array2string(v1, precision=2)}")
    print(f"MC Prediction Output: {np.array2string(v2, precision=2)}")
