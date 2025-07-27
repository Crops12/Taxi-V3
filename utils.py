import numpy as np

def get_epsilon_greedy_action(state, q_table, nA, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(nA)
    else:
        return np.argmax(q_table[state])

def get_softmax_action(state, q_table, temperature=1.0):
    preferences = q_table[state] / temperature
    max_pref = np.max(preferences)
    exp_preferences = np.exp(preferences - max_pref)
    probs = exp_preferences / np.sum(exp_preferences)
    return np.random.choice(len(probs), p=probs)