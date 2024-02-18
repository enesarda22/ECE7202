import numpy as np

from hw3.utils import value_iteration, plot, every_visit_mc_prediction

if __name__ == "__main__":
    th = 1e-6  # threshold to stop the value iteration
    goal = 100

    p_h = 0.25
    _, policy = value_iteration(goal=goal, p_h=p_h, th=th)
    plot(goal=goal, policy=policy, p_h=p_h)

    p_h = 0.55
    v1, policy = value_iteration(goal=goal, p_h=p_h, th=th)
    plot(goal=goal, policy=policy, p_h=p_h)

    v2 = every_visit_mc_prediction(policy=policy, goal=goal, p_h=p_h, th=th)
    v2 = np.insert(v2, 0, 0.0)
