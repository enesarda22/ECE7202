import time

import gym
from gym.wrappers import AtariPreprocessing

import torch
from train_dqn_frostbite import CNN
from utils import choose_eps_greedy

if __name__ == "__main__":
    GAME = "BreakoutNoFrameskip-v4"  # should be with NoFrameskip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(GAME, render_mode="human", repeat_action_probability=0.0)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True)

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = CNN(n_actions).to(device)
    policy_net.load_state_dict(torch.load(f"policy_net_{GAME}.pt", map_location=device))

    for episode_i in range(10000):
        state, _ = env.reset()
        env.render()

        for time_i in range(10000):
            state = torch.tensor(state, device=device, dtype=torch.float32)
            state = state[None, None, ...]
            action = policy_net(state).argmax().item()

            state, reward, terminated, truncated, info = env.step(action)

            time.sleep(0.01)

            if terminated:
                time.sleep(0.5)
                break

    env.close()
