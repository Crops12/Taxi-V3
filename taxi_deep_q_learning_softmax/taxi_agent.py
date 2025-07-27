import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt


def plot_rewards(rewards, title="Rewards over Episodes"):
    print("Training finished.")
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    moving_avg = np.convolve(rewards, np.ones(100) / 100, mode='valid')
    plt.plot(moving_avg, label='Moving Average (100 episodes)')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TaxiAgent:
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.TAU = 0.01
        self.LR = 5e-4
        self.SOFTMAX_TEMP = 1.0

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayBuffer(10000)

    def select_action(self, state):
        with torch.no_grad():
            q_values = self.model(state).squeeze(0)
            probs = F.softmax(q_values / self.SOFTMAX_TEMP, dim=0).cpu()
            action = torch.multinomial(probs, 1).item()
        return torch.tensor([[action]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        experiences = self.memory.sample(self.BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes=3000):
        all_rewards = []
        for ep in tqdm(range(num_episodes), desc="Training Progress"):
            state, _ = self.env.reset()
            state = F.one_hot(torch.tensor(state), num_classes=self.state_size).float().unsqueeze(0).to(device)

            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state)
                next_state_idx, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                if reward == 20:
                    reward += 5
                elif reward == -10:
                    reward = -20
                reward = np.clip(reward, -20, 25)
                reward = torch.tensor([reward], device=device)
                total_reward += reward.item()

                if done:
                    next_state = None
                else:
                    next_state = F.one_hot(torch.tensor(next_state_idx), num_classes=self.state_size).float().unsqueeze(
                        0).to(device)

                self.memory.push(state, action, reward, next_state, done)
                state = next_state

                self.optimize_model()

                target_net_state_dict = self.target_model.state_dict()
                policy_net_state_dict = self.model.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                                1 - self.TAU)
                self.target_model.load_state_dict(target_net_state_dict)

            all_rewards.append(total_reward)

        plot_rewards(all_rewards, "Softmax DQN - Taxi-v3")

    def play(self, episodes=3):
        eval_env = gym.make("Taxi-v3", render_mode="ansi")
        for i in range(episodes):
            state, _ = eval_env.reset(seed=42 + i)
            done = False
            total_reward = 0
            print(f"--- OYNANI\u015e B\u00d6L\u00dcM\u00dc {i + 1} ---")
            print(eval_env.render())
            time.sleep(1)

            while not done:
                state_tensor = F.one_hot(torch.tensor(state), num_classes=self.state_size).float().unsqueeze(0).to(
                    device)
                with torch.no_grad():
                    q_values = self.model(state_tensor).squeeze(0)
                    probs = F.softmax(q_values / self.SOFTMAX_TEMP, dim=0).cpu()
                    action = torch.multinomial(probs, 1).item()

                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                print("\n" * 20)
                print(eval_env.render())
                print(f"Aksiyon: {action}, Anl\u0131k \u00d6d\u00fcl: {reward}, Toplam \u00d6d\u00fcl: {total_reward}")
                time.sleep(0.5)

            print(f"B\u00f6l\u00fcm {i + 1} bitti! Toplam \u00d6d\u00fcl: {total_reward}")
        eval_env.close()


if __name__ == '__main__':
    agent = TaxiAgent()
    agent.train(num_episodes=3000)
    agent.play()
