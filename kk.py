import numpy as np

def policy_iteration(env, gamma=0.9):
  """
  Policy iteration algorithm for finding a policy that maximizes the expected return in a Markov decision process (MDP).

  Args:
    env: The MDP environment.
    gamma: The discount factor.

  Returns:
    A policy that maximizes the expected return in the MDP.
  """

  # Initialize the policy.
  policy = np.zeros(env.nA)

  # Iterate until the policy converges.
  while True:
    # Evaluate the policy.
    value_function = value_iteration(env, policy, gamma)

    # Improve the policy.
    for state in range(env.nS):
      action_values = np.zeros(env.nA)
      for action in range(env.nA):
        action_values[action] = np.sum([env.P[state, action, next_state] * (env.R[state, action, next_state] + gamma * value_function[next_state]) for next_state in range(env.nS)])
      policy[state] = np.argmax(action_values)

    # Check if the policy has converged.
    if np.all(policy == np.roll(policy, 1)):
      break

  return policy