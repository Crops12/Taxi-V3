import gymnasium as gym
import numpy as np
from tqdm import tqdm
from plot_utils import plot_rewards
import time

class TaxiAgent:
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.q_table = np.zeros((self.state_size, self.action_size))  # ← Q-table

    def train(self, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, decay_rate=0.001):
        epsilon = epsilon_start
        rewards = []
        steps_per_episode = []
        epsilons = []

        for ep in tqdm(range(num_episodes), desc="Training Progress"):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            step_counter = 0  # ← ADIM SAYACINI SIFIRLA

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.action_size)
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                best_next = np.max(self.q_table[next_state])
                td_target = reward + gamma * best_next
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += alpha * td_error

                state = next_state
                total_reward += reward
                step_counter += 1  # ← BURADA 1 ARTIR

            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * ep)
            rewards.append(total_reward)
            steps_per_episode.append(step_counter)  # ← ARTIK GERÇEK ADIM SAYISI
            epsilons.append(epsilon)

        plot_rewards(rewards, "Q-Learning (Epsilon-Greedy) - Taxi-v3")
        self.plot_steps_and_epsilon(steps_per_episode, epsilons)

    def play(self, episodes=1):
        self.env = gym.make("Taxi-v3", render_mode="ansi")
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                print(self.env.render())
                time.sleep(0.5)
                total_reward += reward
            print("Episode Reward:", total_reward)

    def plot_steps_and_epsilon(self, steps, epsilons):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(6, 10))

        axs[0].plot(steps, color='orange', label="Steps per Episode")
        axs[0].set_title("Steps Taken to Reach Goal")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Steps")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(epsilons, color='green', label="Epsilon Value Over Episodes")
        axs[1].set_title("Epsilon Value Over Episodes")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Epsilon")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
