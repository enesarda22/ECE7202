import gymnasium as gym
import numpy as np
from tqdm import tqdm
import time


class Policy:
    def __init__(self, rules=None):
        self.rules = rules if rules is not None else np.random.rand(4) * 2 - 1

    def decide_action(self, observation):
        return 0 if np.dot(self.rules, observation) < 0 else 1

    def mutate(self, mutation_rate=0.1):
        mutation = (np.random.rand(4) * 2 - 1) * mutation_rate
        self.rules += mutation


def calculate_fitness(policy, env):
    reset_return = env.reset()
    # Check if the environment's reset method returns a tuple and extract the observation
    if isinstance(reset_return, tuple):
        observation = reset_return[0]  # Assuming the observation is the first element
    else:
        observation = reset_return  # Direct assignment if not a tuple

    total_reward = 0
    for _ in range(1000):
        action = policy.decide_action(observation)
        step_return = env.step(action)
        if isinstance(step_return, tuple) and len(step_return) == 5:
            (
                observation,
                reward,
                done,
                truncated,
                _,
            ) = step_return  # Adjusted for environments returning a tuple with 5 elements
        elif isinstance(step_return, tuple) and len(step_return) == 4:
            (
                observation,
                reward,
                done,
                _,
            ) = step_return  # Standard Gym environment return
        else:
            observation = step_return  # Direct assignment if not a tuple, uncommon case

        total_reward += reward
        if (
            done or truncated
        ):  # Added 'truncated' to handle environments where an episode can be truncated
            break
    return total_reward


def genetic_algorithm(env, generations=50, population_size=10):
    population = [Policy() for _ in range(population_size)]
    fitness_scores = []  # Initialize fitness_scores outside the loop

    for _ in tqdm(range(generations), desc="Generation"):
        fitness_scores = [calculate_fitness(policy, env) for policy in population]

        sorted_population = sorted(
            zip(fitness_scores, population), key=lambda x: x[0], reverse=True
        )
        survivors = [policy for _, policy in sorted_population[: population_size // 2]]

        children = []
        while len(children) < population_size - len(survivors):
            parent1, parent2 = np.random.choice(survivors, 2, replace=False)
            child1_rules = (parent1.rules + parent2.rules) / 2
            child1 = Policy(child1_rules)
            child1.mutate()
            children.append(child1)

        population = survivors + children

    if population and fitness_scores:
        best_fitness = max(fitness_scores)
        best_policy_index = fitness_scores.index(best_fitness)
        return population[best_policy_index]
    else:
        return Policy()


def play_cartpole_with_policy(env, policy, episodes=5):
    for episode in range(episodes):
        reset_return = env.reset()
        # Check if the environment's reset method returns a tuple and extract the observation
        if isinstance(reset_return, tuple):
            observation = reset_return[
                0
            ]  # Assuming the observation is the first element
        else:
            observation = reset_return  # Direct assignment if not a tuple

        total_reward = 0
        for t in range(1000):
            env.render()
            action = policy.decide_action(observation)
            step_return = env.step(action)
            if isinstance(step_return, tuple) and len(step_return) == 5:
                (
                    observation,
                    reward,
                    done,
                    truncated,
                    _,
                ) = step_return  # Adjusted for environments returning a tuple with 5 elements
            elif isinstance(step_return, tuple) and len(step_return) == 4:
                (
                    observation,
                    reward,
                    done,
                    _,
                ) = step_return  # Standard Gym environment return
            else:
                observation = (
                    step_return  # Direct assignment if not a tuple, uncommon case
                )

            total_reward += reward
            if (
                done or truncated
            ):  # Added 'truncated' to handle environments where an episode can be truncated
                print(
                    f"Episode {episode+1} finished after {t+1} timesteps with reward {total_reward}."
                )
                break
    env.close()


# Initialize the environment and run the genetic algorithm
env = gym.make("CartPole-v1")

start_time = time.time()  # Start time
best_policy = genetic_algorithm(env)
end_time = time.time()  # End time
print(f"Execution time: {end_time - start_time} seconds")

if best_policy:
    print("Best policy rules:", best_policy.rules)
    play_cartpole_with_policy(env, best_policy)
else:
    print("Failed to evolve a policy.")
