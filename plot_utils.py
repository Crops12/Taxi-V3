import matplotlib.pyplot as plt

def plot_rewards(rewards, title):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()