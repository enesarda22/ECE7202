from collections import deque, namedtuple
import random
from itertools import count

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
RANDOM_STATE = 42


class CNN(nn.Module):
    def __init__(self, n_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.prelu4(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


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
    return np.sign(reward)


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
    action_batch = torch.tensor(batch.action, device=device, dtype=torch.int64).unsqueeze(1)
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


def evaluate_cnn(env, policy_net, gamma, device, num_episodes=30, eps=0.05):
    rewards_list = []

    for _ in range(num_episodes):
        # initialize the environment and get its state
        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32)
        state = state[None, None, ...]

        total_reward = 0
        for t in count():
            action = choose_eps_greedy(env, policy_net, state, eps)

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += pow(gamma, t) * reward

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
                next_state = next_state[None, None, ...]

            # move to the next state
            state = next_state

            if terminated or truncated:
                rewards_list.append(total_reward)
                break

    return rewards_list


def plot_durations(episode_durations, w=25):
    durations_t = np.array(episode_durations)
    pre_padding = np.ones(int((w - 1) / 2)) * durations_t[0]
    post_padding = np.ones(int((w - 1) / 2)) * durations_t[-1]

    padded_durations_t = np.concatenate([pre_padding, durations_t, post_padding])
    moving_avg = np.convolve(padded_durations_t, np.ones(w), "valid") / w
    moving_avg = moving_avg[: len(durations_t)]

    plt.plot(durations_t, label="Behavior Policy Result")
    plt.plot(moving_avg, label="Moving Average")

    plt.title("Gradient Based Method Result")
    plt.xlabel("Episode")
    plt.ylabel("Duration")

    plt.legend()
    plt.grid()
    plt.show()


def plot_rewards(episode_rewards, w=25):
    rewards_t = np.array(episode_rewards)
    pre_padding = np.ones(int((w - 1) / 2)) * rewards_t[0]
    post_padding = np.ones(int((w - 1) / 2)) * rewards_t[-1]

    padded_rewards_t = np.concatenate([pre_padding, rewards_t, post_padding])
    moving_avg = np.convolve(padded_rewards_t, np.ones(w), "valid") / w
    moving_avg = moving_avg[: len(rewards_t)]

    plt.plot(rewards_t, label="Behavior Policy Result")
    plt.plot(moving_avg, label="Moving Average")

    plt.title("Gradient Based Method Result")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Rewards")

    plt.legend()
    plt.grid()
    plt.show()

def set_seed():
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False
    torch.autograd.set_detect_anomaly(True)