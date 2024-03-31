import time

import gym
from gym.wrappers import AtariPreprocessing, FrameStack

import torch
from train_dqn_frostbite import CNN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("AsteroidsNoFrameskip-v0", render_mode="human")
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True)

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = CNN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load("policy_net_asteroids.pt", map_location=device))

    for episode_i in range(10000):
        state, _ = env.reset()
        env.render()

        for time_i in range(10000):
            action = policy_net((torch.tensor(
                    state, device=device, dtype=torch.float32
                ).unsqueeze(0)).unsqueeze(0)).argmax().item()
            state, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.01)

            if terminated:
                time.sleep(0.5)
                break

    env.close()
