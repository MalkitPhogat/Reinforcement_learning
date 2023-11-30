import numpy as np
import matplotlib.pyplot as plt

num_dishes = 5
true_satisfaction_levels = np.array([10, 8 , 12, 6, 15])
num_customers = 1000

def optimistic_greedy(epsilon, initial_value):
    estimated_values = np.full(num_dishes, initial_value)
    rewards = np.zeros(num_dishes)
    for t in range(num_customers):
        chosen_arm = np.argmax(estimated_values)
        reward = np.random.random() < true_satisfaction_levels[chosen_arm]
        rewards[chosen_arm] += reward
        estimated_values[chosen_arm] = rewards[chosen_arm] / (t + 1)
    return rewards



def epsilon_greedy(epsilon):
    estimated_values = np.zeros(num_dishes)
    rewards = np.zeros(num_dishes)
    for t in range(num_customers):
        if np.random.random() < epsilon:
            chosen_arm = np.random.choice(num_dishes)
        else:
            chosen_arm = np.argmax(estimated_values)
        reward = np.random.random() < true_satisfaction_levels[chosen_arm]
        rewards[chosen_arm] += reward
        estimated_values[chosen_arm] = rewards[chosen_arm] / (t + 1)
    return rewards



def greedy():
    estimated_values = np.zeros(num_dishes)
    rewards = np.zeros(num_dishes)
    for t in range(num_customers):
        chosen_arm = np.argmax(estimated_values)
        reward = np.random.random() < true_satisfaction_levels[chosen_arm]
        rewards[chosen_arm] += reward
        estimated_values[chosen_arm] = rewards[chosen_arm] / (t + 1)
    return rewards




opt_greedy_rewards = optimistic_greedy(1, 5)
epsilon_greedy_rewards = epsilon_greedy(0.1)
greedy_rewards = greedy()

print("Optimistic Greedy Rewards:", opt_greedy_rewards)
print("Epsilon-Greedy Rewards:", epsilon_greedy_rewards)
print("Greedy Rewards:", greedy_rewards)


cumulative_opt_greedy_rewards = np.cumsum(opt_greedy_rewards)
cumulative_epsilon_greedy_rewards = np.cumsum(epsilon_greedy_rewards)
cumulative_greedy_rewards = np.cumsum(greedy_rewards)


plt.figure(figsize=(10, 6))
plt.plot(cumulative_opt_greedy_rewards, label='Optimistic Greedy')
plt.plot(cumulative_epsilon_greedy_rewards, label='Epsilon-Greedy')
plt.plot(cumulative_greedy_rewards, label='Greedy')
plt.xlabel('Customers')
plt.ylabel('Cumulative Satisfaction')
plt.title('Restaurant Dish Selection Comparison')
plt.legend()
plt.show()








def explore_only():
    rewards = np.zeros(num_dishes)
    for t in range(num_customers):
        chosen_dish = np.random.choice(num_dishes)
        reward = np.random.random() < true_satisfaction_levels[chosen_dish]
        rewards[chosen_dish] += reward
    return rewards
explore_only_rewards = explore_only()
cumulative_explore_only_rewards = np.cumsum(explore_only_rewards)
print("Exploration Rewards:", greedy_rewards)
