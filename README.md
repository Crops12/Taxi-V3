# 🚕 Taxi-v3 Reinforcement Learning Agents

A comparative study of classical and deep reinforcement learning algorithms on the [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment. Six agents are implemented, combining three learning algorithms with two exploration strategies.

---

## 📋 Table of Contents

- [Environment Overview](#-environment-overview)
- [Implemented Algorithms](#-implemented-algorithms)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Algorithm Details](#-algorithm-details)
- [Hyperparameters](#-hyperparameters)
- [Results & Visualizations](#-results--visualizations)
- [Dependencies](#-dependencies)

---

## 🌍 Environment Overview

**Taxi-v3** is a discrete grid-world environment where a taxi agent must:
1. Navigate to a passenger's location
2. Pick up the passenger
3. Drive to the destination
4. Drop off the passenger

| Property | Value |
|---|---|
| State Space | 500 discrete states |
| Action Space | 6 discrete actions (N, S, E, W, Pick-up, Drop-off) |
| Reward | +20 for successful drop-off, −10 for illegal pick-up/drop-off, −1 per step |
| Episode End | Successful drop-off or step limit reached |

---

## 🤖 Implemented Algorithms

| Module | Algorithm | Exploration |
|---|---|---|
| `taxi_q_learning_epsilon` | Q-Learning | ε-Greedy |
| `taxi_q_learning_softmax` | Q-Learning | Softmax |
| `taxi_sarsa_epsilon` | SARSA | ε-Greedy |
| `taxi_sarsa_softmax` | SARSA | Softmax |
| `taxi_deep_q_learning_epsilon` | Deep Q-Network (DQN) | ε-Greedy |
| `taxi_deep_q_learning_softmax` | Deep Q-Network (DQN) | Softmax + Replay Buffer + Target Network |

---

## 📁 Project Structure

```
Taxi-V3-main/
│
├── main.py                          # Entry point — runs all agents sequentially
├── utils.py                         # Shared action-selection utilities
├── plot_utils.py                    # Reward plotting helper
│
├── taxi_q_learning_epsilon/
│   └── taxi_agent.py                # Q-Learning with ε-greedy exploration
│
├── taxi_q_learning_softmax/
│   └── taxi_agent.py                # Q-Learning with Softmax exploration
│
├── taxi_sarsa_epsilon/
│   └── taxi_agent.py                # SARSA with ε-greedy exploration
│
├── taxi_sarsa_softmax/
│   └── taxi_agent.py                # SARSA with Softmax exploration
│
├── taxi_deep_q_learning_epsilon/
│   └── taxi_agent.py                # DQN with ε-greedy (one-hot state, MLP)
│
└── taxi_deep_q_learning_softmax/
    └── taxi_agent.py                # DQN with Softmax, Replay Buffer & Target Network
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Taxi-V3.git
cd Taxi-V3
```

### 2. Install Dependencies

```bash
pip install gymnasium torch numpy matplotlib tqdm
```

> **Note:** For GPU acceleration, install the appropriate CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Run All Agents

```bash
python main.py
```

This trains and evaluates agents in the following order:
1. Q-Learning (Softmax)
2. Q-Learning (ε-Greedy)
3. Deep Q-Learning (ε-Greedy)

> To run specific agents, import and instantiate them directly from their respective modules.

---

## 🧠 Algorithm Details

### Q-Learning (Off-Policy TD)

Updates the Q-table using the **greedy** next-action estimate, regardless of the actual policy used:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### SARSA (On-Policy TD)

Updates the Q-table using the **actual** next action chosen by the policy (on-policy):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

### Deep Q-Network (DQN)

Replaces the Q-table with a neural network. Two variants are implemented:

- **ε-Greedy DQN** — Lightweight MLP (`input → 128 → output`), one-hot encoded states, no replay buffer.
- **Softmax DQN** — Deeper MLP (`input → 256 → 256 → output`) with:
  - **Experience Replay Buffer** (capacity 10,000)
  - **Target Network** with soft updates (τ = 0.01)
  - **Reward Shaping** (+5 bonus on success, −20 on illegal actions)
  - **Gradient Clipping** for stable training
  - **AdamW optimizer** with AMSGrad

### Exploration Strategies

**ε-Greedy** — Takes a random action with probability ε, otherwise exploits the best known action. ε decays exponentially during training:

$$\varepsilon_t = \varepsilon_{\text{end}} + (\varepsilon_{\text{start}} - \varepsilon_{\text{end}}) \cdot e^{-\lambda t}$$

**Softmax (Boltzmann)** — Converts Q-values into a probability distribution using temperature τ. Higher τ = more exploration:

$$P(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$

---

## ⚙️ Hyperparameters

### Tabular Methods (Q-Learning & SARSA)

| Parameter | ε-Greedy | Softmax |
|---|---|---|
| Episodes | 5,000 / 1,000 | 1,000 |
| Learning Rate (α) | 0.1 | 0.1 |
| Discount Factor (γ) | 0.99 / 0.6 | 0.6 |
| ε start / end | 1.0 → 0.1 | — |
| Decay Rate | 0.001 | — |
| Temperature (τ) | — | 1.0 |

### Deep Q-Network

| Parameter | ε-Greedy DQN | Softmax DQN |
|---|---|---|
| Episodes | 5,000 | 3,000 |
| Learning Rate | 0.001 (Adam) | 5e-4 (AdamW) |
| Discount Factor (γ) | 0.99 | 0.99 |
| Batch Size | — | 128 |
| Replay Buffer | — | 10,000 |
| Target Net Update (τ) | — | 0.01 (soft) |
| Temperature (τ) | — | 1.0 |
| Hidden Layers | 128 | 256 → 256 |

---

## 📊 Results & Visualizations

Each agent automatically generates the following plots after training:

- **Reward per Episode** — total cumulative reward over training
- **Steps per Episode** *(Q-Learning ε-Greedy)* — how quickly the agent solves each episode
- **Epsilon Decay** *(ε-Greedy agents)* — exploration rate over time
- **Moving Average** *(Softmax DQN)* — smoothed reward curve (window = 100 episodes)

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `gymnasium` | Taxi-v3 environment |
| `numpy` | Q-table operations and numerical utilities |
| `torch` | Neural network models for DQN agents |
| `matplotlib` | Training curve visualization |
| `tqdm` | Training progress bar |

Install all at once:

```bash
pip install gymnasium numpy torch matplotlib tqdm
```

---

## 📄 License

This project is intended for educational purposes. Feel free to fork, modify, and experiment.
