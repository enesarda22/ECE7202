from collections import deque, namedtuple
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class CNN(nn.Module):
    def __init__(self, n_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.PReLU()

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu4 = nn.PReLU()

        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.contiguous().view(-1, 64 * 7 * 7)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.prelu_1 = nn.PReLU()
        self.layer2 = nn.Linear(128, 128)
        self.prelu_2 = nn.PReLU()
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):  # returns the action value function approximation
        x = self.prelu_1(self.layer1(x))
        x = self.prelu_2(self.layer2(x))
        return self.layer3(x)


def clip_reward(reward):
    if reward > 0.0:
        return min(1.0, reward)
    elif reward < 0.0:
        return max(-1.0, reward)
    return reward


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
    action_batch = torch.tensor(
        batch.action, device=device, dtype=torch.int64
    ).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.int64)

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


def plot_durations(episode_durations, w=25):
    durations_t = np.array(episode_durations)
    padded_durations_t = np.concatenate([np.zeros(w - 1), durations_t])
    moving_avg = np.convolve(padded_durations_t, np.ones(w), "valid") / w

    plt.plot(durations_t, label="Behavior Policy Result")
    plt.plot(moving_avg, label="Moving Average")

    plt.title("Gradient Based Method Result")
    plt.xlabel("Episode")
    plt.ylabel("Duration")

    plt.legend()
    plt.grid()
    plt.show()
