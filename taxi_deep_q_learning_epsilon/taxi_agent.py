import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from plot_utils import plot_rewards


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TaxiAgent:
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.model = DQN(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def train(self, num_episodes=5000, epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.001):
        gamma = 0.99
        epsilon = epsilon_start
        rewards = []

        for ep in tqdm(range(num_episodes), desc="Training Progress"):
            state, _ = self.env.reset()
            state_tensor = torch.eye(self.state_size, device=device)[state].to(device)
            done = False
            total_reward = 0

            while not done:
                with torch.no_grad():
                    q_values = self.model(state_tensor)

                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.action_size)
                else:
                    action = torch.argmax(q_values).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_tensor = torch.eye(self.state_size, device=device)[next_state]

                with torch.no_grad():
                    target = reward + gamma * torch.max(self.model(next_tensor))

                output = self.model(state_tensor)[action]
                loss = self.loss_fn(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state_tensor = next_tensor
                total_reward += reward

            # Decay epsilon
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * ep)
            rewards.append(total_reward)

        plot_rewards(rewards, "Deep Q-Learning (Epsilon-Greedy) - Taxi-v3")

    def play(self, episodes=1):
        self.env = gym.make("Taxi-v3", render_mode="ansi")
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                state_tensor = torch.eye(self.state_size, device=device)[state].to(device)
                with torch.no_grad():
                    action = torch.argmax(self.model(state_tensor)).item()
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                print(self.env.render())
                time.sleep(0.5)
            print("Episode Reward:", total_reward)
