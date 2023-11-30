import numpy as np

def value_iteration(transitions, rewards, discount_factor=0.9, epsilon=1e-6, max_iterations=1000):
    states = list(transitions.keys())
    actions = {state: list(transitions[state].keys()) for state in states}

    V = {state: 0 for state in states}

    for _ in range(max_iterations):
        delta = 0

        for state in states:
            v = V[state]
            max_action_value = float('-inf')

            for action in actions[state]:
                action_value = sum(
                    prob * (rewards[state][action] + discount_factor * V[next_state]) for next_state, prob in transitions[state][action]
                )

                max_action_value = max(max_action_value, action_value)

            V[state] = max_action_value
            delta = max(delta, abs(v - V[state]))

        if delta < epsilon:
            break

    policy = {state: max(actions[state], key=lambda a: sum(prob * (rewards[state][a] + discount_factor * V[next_state]) for next_state, prob in transitions[state][a])) for state in states}

    return policy, V

if __name__ == "__main__":
  
    transitions = {
        's0': {'a0': [('s0', 0.5), ('s1', 0.5)], 'a1': [('s1', 1.0)]},
        's1': {'a0': [('s0', 0.7), ('s1', 0.3)], 'a1': [('s0', 0.1), ('s1', 0.9)]}
    }

    rewards = {
        's0': {'a0': 5, 'a1': 2},
        's1': {'a0': 1, 'a1': 3}
    }
    optimal_policy, optimal_values = value_iteration(transitions, rewards)

    print("Optimal Policy:")
    print(optimal_policy)
    print("\nOptimal Values:")
    print(optimal_values)
