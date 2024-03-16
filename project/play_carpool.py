import time

import gymnasium as gym
import torch

from project.train_dqn import DQN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1", render_mode="human")
    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load("policy_net.pt", map_location=device))

    for episode_i in range(10000):
        state, info = env.reset()
        env.render()

        for time_i in range(500):
            action = policy_net(torch.FloatTensor(state, device=device)).argmax().item()
            state, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.01)

            if terminated:
                time.sleep(0.5)
                break

    env.close()
