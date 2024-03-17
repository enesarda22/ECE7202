import time

import gymnasium as gym
import math
from itertools import count

import torch
import torch.optim as optim
from tqdm import tqdm

from utils import (
    DQN,
    ReplayMemory,
    choose_eps_greedy,
    calculate_loss,
    plot_durations,
)

if __name__ == "__main__":
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.001
    EPS_DECAY = 1000  # controls the decay rate
    TAU = 0.01
    LR = 2.65e-3
    EARLY_STOP = 20

    step = 0  # to keep track of the eps decay
    num_episodes = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    memory = ReplayMemory(10000)

    episode_durations = []

    start_time = time.time()
    early_stop_count = 0
    for i_episode in tqdm(range(num_episodes)):
        # initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        for t in count():
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)

            action = choose_eps_greedy(env, policy_net, state, eps)
            observation, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, device=device, dtype=torch.float32
                ).unsqueeze(0)

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
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

            # soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                diff = policy_net_state_dict[key] - target_net_state_dict[key]
                target_net_state_dict[key] += diff * TAU
            target_net.load_state_dict(target_net_state_dict)

            step += 1
            if terminated or truncated:
                if t == 499:
                    early_stop_count += 1
                else:
                    early_stop_count = 0

                episode_durations.append(t + 1)
                break
        scheduler.step()
        if early_stop_count > EARLY_STOP:
            break
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_durations(episode_durations, w=EARLY_STOP)
    torch.save(target_net.state_dict(), "policy_net.pt")
