from collections import deque, namedtuple
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):  # returns the action value function approximation
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def choose_eps_greedy(env, q_net, state, eps):
    if random.random() > eps:
        with torch.no_grad():
            return q_net(state).argmax(axis=1).item()
    else:
        return env.action_space.sample()


def calculate_loss(memory, policy_net, target_net, batch_size, gamma, device):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.LongTensor(batch.action).unsqueeze(1)
    reward_batch = torch.FloatTensor(batch.reward)

    # compute Q(s_t, a) according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # compute V(s_{t+1}) greedily wrt target_net
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )

    expected_state_action_values = reward_batch + gamma * next_state_values
    loss = F.smooth_l1_loss(state_action_values.flatten(), expected_state_action_values)
    return loss


def plot_durations(episode_durations, w=100):
    durations_t = np.array(episode_durations)
    padded_durations_t = np.concatenate([np.zeros(w - 1), durations_t])
    moving_avg = np.convolve(padded_durations_t, np.ones(w), "valid") / w

    plt.plot(durations_t)
    plt.plot(moving_avg)

    plt.title("Result")
    plt.xlabel("Episode")
    plt.ylabel("Duration")

    plt.grid()
    plt.show()
