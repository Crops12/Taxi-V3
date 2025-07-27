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

    def train(self, num_episodes=1000, alpha=0.1, gamma=0.6, epsilon=0.1):
        rewards = []
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            action = get_epsilon_greedy_action(state, self.q_table, self.n_actions, epsilon)
            done = False
            total_reward = 0

            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_action = get_epsilon_greedy_action(next_state, self.q_table, self.n_actions, epsilon)
                self.q_table[state, action] += alpha * (
                    reward + gamma * self.q_table[next_state, next_action] - self.q_table[state, action]
                )
                state = next_state
                action = next_action
                total_reward += reward

            rewards.append(total_reward)
        plot_rewards(rewards, "SARSA (Epsilon) - Taxi-v3")

    def play(self, episodes=1):
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
            print("Episode Reward:", total_reward)