import numpy as np
import matplotlib.pyplot as plt

# Variables for the restaurant multi-arm bandit problem
num_dishes = 5
true_satisfaction_levels = np.array([10,8,12,6,15])
num_customers = 1

# Number of rounds or pulls
num_rounds = num_customers

def explore_only():
    rewards = np.zeros(num_dishes)
    for t in range(num_rounds):
        chosen_dish = np.random.choice(num_dishes)
        reward = np.random.random() < true_satisfaction_levels[chosen_dish]
        rewards[chosen_dish] += reward
    return rewards

def greedy():
    estimated_values = np.zeros(num_dishes)
    rewards = np.zeros(num_dishes)
    for t in range(num_customers):
        chosen_dish = np.argmax(estimated_values)
        reward = np.random.random() < true_satisfaction_levels[chosen_dish]
        rewards[chosen_dish] += reward
        estimated_values[chosen_dish] = rewards[chosen_dish] / (t + 1)
    return rewards

# Run the exploration-only algorithm
explore_only_rewards = explore_only()
greedy_rewards = greedy()

# Calculate cumulative rewards for plotting
cumulative_explore_only_rewards = np.cumsum(explore_only_rewards)
cumulative_greedy_rewards = np.cumsum(greedy_rewards)
print("Exploration Rewards:", explore_only_rewards)
print("Greedy Rewards:", greedy_rewards)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(cumulative_explore_only_rewards, label='Explore Only')
plt.plot(cumulative_greedy_rewards, label='Greedy')
plt.xlabel('Customers')
plt.ylabel('Cumulative Satisfaction')
plt.title('Explore Only Strategy')
plt.legend()
plt.show()




