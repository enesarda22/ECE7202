import time

import gym
import numpy as np
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
    plot_rewards,
    evaluate_cnn,
)

if __name__ == "__main__":
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.99
    EPS_END = 0.1
    EPS_DECAY = 1000000  # controls the decay rate
    LR = 2.5e-4
    N_MEMORY = 100000
    GAME = "BoxingNoFrameskip-v4"  # should be with NoFrameskip
    NUM_EPISODES = 250
    UPDATE_C = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(GAME)
    env = AtariPreprocessing(env)

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = CNN(n_actions).to(device)
    target_net = CNN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(N_MEMORY)

    step = 0  # to keep track of the eps decay
    mean_rewards_list = []
    start_time = time.time()

    for i_episode in tqdm(range(NUM_EPISODES)):
        # initialize the environment and get its state
        state, _ = env.reset()
        state = torch.FloatTensor(state, device=device)
        state = state[None, None, ...]

        for t in count():
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)
            action = choose_eps_greedy(env, policy_net, state, eps)

            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                next_state = None
            else:
                next_state = torch.FloatTensor(next_state, device=device)
                next_state = next_state[None, None, ...]

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
                optimizer.step()

            if step > UPDATE_C:
                target_net.load_state_dict(policy_net.state_dict())

            step += 1

            if terminated or truncated:
                rewards_list = evaluate_cnn(
                    env=env,
                    policy_net=policy_net,
                    gamma=GAMMA,
                    device=device,
                )
                mean_rewards_list.append(np.mean(rewards_list))
                break

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_rewards(mean_rewards_list, w=50)
    torch.save(target_net.state_dict(), f"policy_net_{GAME}.pt")
