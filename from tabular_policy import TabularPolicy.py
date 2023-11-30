import numpy as np

def policy_iteration(env, gamma=0.9, max_iterations=1000, tol=1e-6):
    num_states = env.nS
    num_actions = env.nA
    policy = np.ones((num_states, num_actions)) / num_actions  # Initialize with a uniform policy
    value_function = np.zeros(num_states)
    
    for i in range(max_iterations):
        # Policy Evaluation
        while True:
            delta = 0
            for state in range(num_states):
                v = value_function[state]
                new_v = 0
                for action, action_prob in enumerate(policy[state]):
                    for prob, next_state, reward, done in env.P[state][action]:
                        new_v += action_prob * prob * (reward + gamma * value_function[next_state])
                value_function[state] = new_v
                delta = max(delta, abs(v - new_v))
            if delta < tol:
                break
        
        # Policy Improvement
        policy_stable = True
        for state in range(num_states):
            old_action = np.argmax(policy[state])
            action_values = np.zeros(num_actions)
            for action in range(num_actions):
                for prob, next_state, reward, done in env.P[state][action]:
                    action_values[action] += prob * (reward + gamma * value_function[next_state])
            new_action = np.argmax(action_values)
            if old_action != new_action:
                policy_stable = False
            policy[state] = np.eye(num_actions)[new_action]
        
        if policy_stable:
            break

    return policy, value_function

# Example usage:
# You would need to replace `env` with your specific environment (e.g., OpenAI Gym environment).
# policy, value_function = policy_iteration(env)
