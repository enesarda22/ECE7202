# Cartpole playing using GA
import time
import gymnasium as gym
import numpy as np


class Policy:
    def __init__(self, rules=None):
        # Ensure rules are a numpy array for dot product operation
        self.rules = np.array(rules if rules is not None else np.random.rand(4) * 2 - 1)

    def decide_action(self, observation):
        # Convert observation to numpy array to ensure dot product works correctly
        observation = np.array(observation)
        return 0 if np.dot(self.rules, observation) < 0 else 1


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")

    best_policy = Policy([0.10693759, -0.27635204, 0.79183118, 0.59610431])

    for episode_i in range(100):
        observation, info = env.reset()

        for time_i in range(5000):
            env.render()  # render the environment at the current state
            action = best_policy.decide_action(observation)
            observation, reward, done, truncated, info = env.step(action)

            time.sleep(0.01)  # sleep to slow down the loop
            if done or truncated:  # break the loop if the episode is over
                break

    env.close()  # close the environment after all episodes are done
