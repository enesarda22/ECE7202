

#Cartpole playing using GA
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

    best_policy = Policy()  # This should be replaced with your actual best policy from GA
    
    for episode_i in range(100):
        observation, info = env.reset()  # Use env.reset() correctly based on gymnasium API

        for time_i in range(5000):
            env.render()  # Render the environment at the current state
            action = best_policy.decide_action(observation)  # Get action from the policy
            observation, reward, done, truncated, info = env.step(action)  # Execute the action

            time.sleep(0.01)  # Sleep to slow down the loop for visual inspection
            if done or truncated:  # Break the loop if the episode is over
                break

    env.close()  # Close the environment after all episodes are done
