from collections import OrderedDict

import gym
import numpy as np
import torch
from tqdm import tqdm
import time
from gym.wrappers import AtariPreprocessing

from project.utils import CNN, choose_eps_greedy, plot_rewards

Gens = 500
Pops = 10
NumEpisodes = 10

MutRate = 0.01
MutStr = 0.1
RemSurvivors = 5

SleepForGame = 0.0005
RendMode = None


def play_game_with_policy(env, policy, render_mode=RendMode, sleep_time=SleepForGame):
    observation, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if render_mode == "human":
            env.render()
        action = policy.decide_action(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        time.sleep(
            sleep_time
        )  # Slow down the game a bit so you can see what's happening

    env.close()
    print(f"Total Reward: {total_reward}")


def calculate_fitness(q_net, env):
    episode_reward = []

    for episodes in range(NumEpisodes):
        observation, _ = env.reset()
        observation = torch.tensor(observation, dtype=torch.float, device=device)
        observation = observation[None, None, :, :]

        done = False

        total_reward = 0
        while not done:
            action = choose_eps_greedy(
                env=env,
                q_net=q_net,
                state=observation,
                eps=1e-3,
            )
            observation, reward, done, _, info = env.step(
                action
            )  # Adjust this line based on the output of your env.step()
            observation = torch.tensor(observation, dtype=torch.float, device=device)
            observation = observation[None, None, :, :]

            total_reward += reward

        episode_reward.append(total_reward)

    return np.mean(episode_reward)


def genetic_algorithm(env, generations=Gens, population_size=Pops):
    population = [CNN(n_actions) for _ in range(population_size)]
    best_fitness_scores = []

    for _ in tqdm(range(generations), desc="Generation"):
        fitness_scores = [calculate_fitness(q_net, env) for q_net in population]
        print("Fitness scores:", fitness_scores)

        best_fitness_scores.append(max(fitness_scores))
        sorted_population = sorted(
            zip(fitness_scores, population), key=lambda x: x[0], reverse=True
        )
        survivors = [q_net for _, q_net in sorted_population[:RemSurvivors]]

        children = []
        for _ in range(len(survivors), population_size):
            parent1, parent2 = np.random.choice(survivors, 2, replace=False)
            child_dict = OrderedDict()
            sd1 = parent1.state_dict()
            sd2 = parent2.state_dict()

            for key in sd1:
                child_dict[key] = (sd1[key] + sd2[key]) / 2

                mutation_mask = torch.randn(sd1[key].shape) < MutRate
                mutations = torch.randn(sd1[key].shape) * MutStr

                child_dict[key] += mutation_mask * mutations

            child = CNN(n_actions)
            child.load_state_dict(child_dict)

            children.append(child)

        population = survivors + children

    best_fitness = max(best_fitness_scores)
    best_policy_index = best_fitness_scores.index(best_fitness)
    return population[best_policy_index], best_fitness_scores


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("PongNoFrameskip-v4")
env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
n_actions = env.action_space.n

start_time = time.time()
best_q_net, best_fitness_scores = genetic_algorithm(env)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

plot_rewards(best_fitness_scores, w=50)
torch.save(best_q_net.state_dict(), "policy_net_ga_pong.pt")

# best_policy = (
#     Policy()
# )  # This would be the policy you obtained from your genetic algorithm
# play_game_with_policy(env, best_policy)
