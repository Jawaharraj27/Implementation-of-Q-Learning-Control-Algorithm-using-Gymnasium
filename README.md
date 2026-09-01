# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The agent learns an optimal action-value function using the Q-Learning update rule and an epsilon-greedy strategy to reach the goal while avoiding holes.


## Software Requirements
Google Colab
Python 3.x
Gymnasium
NumPy
Matplotlib


## Environment Description

FrozenLake-v1 is a grid-world environment where:

S = Starting state
F = Frozen/safe surface
H = Hole
G = Goal
The agent receives a reward of 1 for reaching the goal.
The agent receives 0 for other transitions.
The objective is to reach the goal while avoiding holes.

## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

Initialize the Q-table with zeros.
Start an episode from the initial state.
Select an action using epsilon-greedy policy.
Execute the action and observe reward and next state.
Update the Q-value using the Q-Learning update rule.
Reduce epsilon gradually to shift from exploration to exploitation.
Repeat for many episodes.
Extract the learned policy from the Q-table.
Evaluate the trained agent.

## Python Program

```python
# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
# Write your code here

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((env.observation_space.n, env.action_space.n))

episode_rewards = []

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
for episode in range(episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        action = choose_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        best_next_action = np.max(Q[next_state])

        Q[state, action] = Q[state, action] + learning_rate * (
            reward
            + discount_factor * best_next_action * (not done)
            - Q[state, action]
        )

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    # Reduce exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------
def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Calculate State Values and Policy
# -------------------------------------------------
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

# -------------------------------------------------
# Output
# -------------------------------------------------
print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:", average_reward)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
````








## Output

Final Q-table:

<img width="417" height="382" alt="image" src="https://github.com/user-attachments/assets/0bc8ca5d-6c9e-4d8b-a04b-7aabf46e92c3" />




Estimated State-Value Function:


<img width="418" height="131" alt="image" src="https://github.com/user-attachments/assets/1e4f5c2b-5d12-4bc7-a2a0-301d7e17eb49" />




Learned Policy:


<img width="586" height="134" alt="image" src="https://github.com/user-attachments/assets/b91a52a3-c20d-45c7-ace4-2f67d0d2c54d" />


Average reward over last 1000 episodes:

<img width="506" height="55" alt="image" src="https://github.com/user-attachments/assets/677fdf6f-1ace-47d5-84c0-844a3a677ba0" />



## Result
The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned the optimal action-value function through repeated interaction with the environment and successfully learned a policy for reaching the goal while avoiding the holes.



## Inference

Q-Learning is an off-policy reinforcement learning algorithm that learns the optimal Q-values without requiring a model of the environment. Using epsilon-greedy exploration, the agent initially explores different actions and gradually exploits the learned Q-values. The final Q-table can be used to select the best action in each state.

