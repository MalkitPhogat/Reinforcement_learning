import numpy as np
R = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 100],
              [-1, -1, -1, 0, -1, -1],
              [-1,  0,  0, -1, 0, -1],
              [ 0, -1, -1,  0, -1, 100],
              [-1, 0, -1, -1, 0, 100]])

# Q matrix
Q = np.zeros_like(R, dtype=float)

# Hyperparameters
gamma = 0.8
learning_rate = 0.9
num_episodes = 3

# Q-learning update function
def update_q_matrix(state, action, reward, next_state):
    max_q_next = np.max(Q[next_state, :])
    Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + gamma * max_q_next)

# Training
for episode in range(num_episodes):
    current_state = np.random.randint(0, Q.shape[0])
    while current_state != 5:
        available_actions = np.where(R[current_state, :] >= 0)[0]
        action = np.random.choice(available_actions)
        reward = R[current_state, action]
        next_state = action
        update_q_matrix(current_state, action, reward, next_state)
        current_state = next_state

# Display trained Q matrix
print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

# Testing
current_state = 1
steps = [current_state]

while current_state != 5:
    next_step_index = np.argmax(Q[current_state, :])
    steps.append(next_step_index)
    current_state = next_step_index

# Display selected steps and final Q matrix after testing
print("Selected steps during testing:")
print(steps)
print("Final Q matrix after testing:")
print(Q / np.max(Q) * 100)
