import gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from gym.wrappers import AtariPreprocessing

Gens = 50
Pops = 10
NumEpisodes=10

MutRate = 0.01
MutStr = 0.1
RemSurvivors = 2

SleepForGame = 0.0005
RendMode = None

actionNum = 6

class Policy:
    def __init__(self, rules=None):
        self.rules = rules if rules is not None else np.random.rand(84*84, actionNum) - 0.5

    def decide_action(self, observation, epsilon=0):
        # Flatten the observation correctly assuming it is an 84x84 image
        observation_flat = observation.reshape(-1)  # Reshape to ensure it is flattened correctly
        
        # Calculate the action scores
        action_scores = np.dot(observation_flat, self.rules)
        
        # Implement epsilon-greedy action selection
        if np.random.rand() < epsilon:
            # Return a random action
            return np.random.randint(0, len(action_scores))
        else:
            # Return the action with the highest score
            return np.argmax(action_scores)

    def mutate(self, mutation_rate=MutRate, mutation_strength=MutStr):
        mutation_mask = np.random.rand(*self.rules.shape) < mutation_rate
        mutations = np.random.randn(*self.rules.shape) * mutation_strength
        self.rules += mutation_mask * mutations

def play_game_with_policy(env, policy, render_mode=RendMode, sleep_time=SleepForGame):
    observation, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        if render_mode == 'human':
            env.render()
        action = policy.decide_action(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        time.sleep(sleep_time)  # Slow down the game a bit so you can see what's happening
    
    env.close()
    print(f"Total Reward: {total_reward}")

def calculate_fitness(policy, env):
    Episode_reward = []

    for episodes in range(NumEpisodes):
        observation, _ = env.reset()
        done = False
        reward = 0
        total_reward=0 
        while not done:
            action = policy.decide_action(observation)
            observation, reward, done, _, info = env.step(action)  # Adjust this line based on the output of your env.step()

            total_reward += reward
        Episode_reward.append(total_reward)

    return np.mean(Episode_reward)

def genetic_algorithm(env, generations=Gens, population_size=Pops):
    population = [Policy() for _ in range(population_size)]
    best_fitness_scores = []

    for _ in tqdm(range(generations), desc="Generation"):
        fitness_scores = [calculate_fitness(policy, env) for policy in population]
        print("Fitness scores:", fitness_scores)
        best_fitness_scores.append(max(fitness_scores))
        sorted_population = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
        survivors = [policy for _, policy in sorted_population[: population_size // RemSurvivors]]

        children = []
        for _ in range(len(survivors), population_size):
            parent1, parent2 = np.random.choice(survivors, 2, replace=False)
            child1_rules = (parent1.rules + parent2.rules) / 2
            child1 = Policy(child1_rules)
            child1.mutate()
            children.append(child1)

        population = survivors + children

    best_fitness = max(fitness_scores)
    best_policy_index = fitness_scores.index(best_fitness)
    return population[best_policy_index], best_fitness_scores

env = gym.make('PongNoFrameskip-v4')
env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)

start_time = time.time()
best_policy, best_fitness_scores = genetic_algorithm(env)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

if best_policy:
    print("Best policy rules:", best_policy.rules)
    plt.plot(np.arange(1, len(best_fitness_scores) + 1), best_fitness_scores)
    plt.title("Genetic Algorithm Result")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid()
    plt.show()
else:
    print("Failed to evolve a policy.")

best_policy = Policy()  # This would be the policy you obtained from your genetic algorithm
play_game_with_policy(env, best_policy)