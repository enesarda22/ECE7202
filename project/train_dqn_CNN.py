import time

import gym
from gym.wrappers import AtariPreprocessing

import math
from itertools import count
import torch
import torch.optim as optim
from tqdm import tqdm

from utils import (
    CNN,
    ReplayMemory,
    choose_eps_greedy,
    calculate_loss,
    plot_durations,
    plot_rewards
)

if __name__ == "__main__":
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.99
    EPS_END = 0.01
    EPS_DECAY = 10000  # controls the decay rate
    LR = 2.5e-4
    N_MEMORY = 100000
    GAME = "BoxingNoFrameskip-v4"  # ALl games should be with NoFrameskip to compatible with preprocessor!
    NUM_EPISODES = 250
    UPDATE_C = 1000

    step = 0  # to keep track of the eps decay
    update_step = 0  # to keep track of the target net update, HARD UPDATE NOT SOFT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(GAME, repeat_action_probability=0.0)  # buttons will not be stuck
    env = AtariPreprocessing(env, frame_skip=4)

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = CNN(n_actions).to(device)
    target_net = CNN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(N_MEMORY)

    episode_durations = []
    start_time = time.time()
    rewards_list = []

    for i_episode in tqdm(range(NUM_EPISODES)):
        # initialize the environment and get its state
        state, info = env.reset()
        state = (torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)).unsqueeze(0)
        total_reward = 0
        for t in count():
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)
            action = choose_eps_greedy(env, policy_net, state, eps)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += pow(GAMMA, t) * reward
            if terminated:
                next_state = None
            else:
                next_state = (torch.tensor(
                    observation, device=device, dtype=torch.float32
                ).unsqueeze(0)).unsqueeze(0)

            # store the transition in memory
            memory.push(state, action, next_state, reward)

            # move to the next state
            state = next_state

            # perform one step of the optimization (on the policy network)
            loss = calculate_loss(
                memory=memory,
                policy_net=policy_net,
                target_net=target_net,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                device=device,
            )

            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            if update_step > UPDATE_C:
                target_net.load_state_dict(policy_net.state_dict())

            step += 1
            update_step += 1

            if terminated or truncated:
                episode_durations.append(t + 1)
                rewards_list.append(total_reward)
                break

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_durations(episode_durations, w=50)
    plot_rewards(rewards_list, w=50)
    torch.save(target_net.state_dict(), f"policy_net_{GAME}.pt")

