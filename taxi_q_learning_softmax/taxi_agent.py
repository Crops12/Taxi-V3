import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from plot_utils import plot_rewards
from utils import get_epsilon_greedy_action, get_softmax_action

class TaxiAgent:
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def train(self, num_episodes=1000, alpha=0.1, gamma=0.6, temperature=1.0):
        rewards = []
        for ep in range(num_episodes):
            state, _ = self.env.reset(seed=31)
            done = False
            total_reward = 0

            while not done:
                action = get_softmax_action(state, self.q_table, temperature)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                best_next = np.max(self.q_table[next_state])
                self.q_table[state, action] += alpha * (reward + gamma * best_next - self.q_table[state, action])
                state = next_state
                total_reward += reward

            rewards.append(total_reward)
        plot_rewards(rewards, "Q-Learning (Softmax) - Taxi-v3")

    def play(self, episodes=1):
        self.env = gym.make("Taxi-v3", render_mode="human")
        state, _ = self.env.reset(seed=31)
        self.env.render()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, done, truncated, _ = self.env.step(action)
            if done:
                self.env.render()
                break
            total_reward += reward

        print("Episode Reward:", total_reward)