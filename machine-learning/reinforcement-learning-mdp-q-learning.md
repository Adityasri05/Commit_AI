# Reinforcement Learning: MDPs and Q-Learning

Reinforcement learning (RL) studies how an agent learns to act through interaction with an environment. The agent observes a state, chooses an action, receives a reward, and observes the next state. Unlike supervised learning, the correct action is not usually provided for every state.

## Markov decision process

A finite Markov decision process (MDP) is commonly described by:

- S: possible states;
- A: available actions;
- P(s' | s, a): transition probability;
- R(s, a, s'): reward for a transition;
- gamma: discount factor between 0 and 1.

The Markov assumption says the current state contains the information needed to predict the next state and reward distribution. If important history is missing, the problem may be partially observable instead.

The objective is to maximize expected discounted return:

    G_t = r_(t+1) + gamma*r_(t+2) + gamma^2*r_(t+3) + ...

A large gamma values future rewards more strongly. A small gamma emphasizes immediate reward.

## Q-learning

Q-learning estimates the value of taking action a in state s and then following a good policy. Its tabular update is:

    Q(s,a) <- Q(s,a) + alpha * [
        r + gamma * max_a' Q(s',a') - Q(s,a)
    ]

Here alpha is the learning rate. The bracketed term is the temporal-difference error. For a terminal next state, omit the future-value term.

Q-learning is **off-policy**: the update targets the greedy action even when behavior uses exploration. It is a foundational method for small discrete state/action spaces; large or continuous spaces need function approximation or other algorithms.

## Epsilon-greedy exploration

With probability epsilon, choose a random valid action; otherwise choose the action with the largest estimated Q-value. Decay epsilon gradually so the agent explores early and increasingly exploits what it has learned. Do not decay it so quickly that important states are never explored.

~~~python
import numpy as np

q = np.zeros((n_states, n_actions))
alpha, gamma = 0.1, 0.99
epsilon = 1.0

for episode in range(num_episodes):
    state, info = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q[state]))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        target = reward if done else reward + gamma * np.max(q[next_state])
        q[state, action] += alpha * (target - q[state, action])
        state = next_state

    epsilon = max(0.05, epsilon * 0.995)
~~~

This is tabular pseudocode for a discrete environment whose observations can be used as integer state indices. Gymnasium environments may return structured observations; map them to states before indexing the Q-table. Treat true termination and time-limit truncation carefully when deciding whether bootstrapping is valid for the task.

## Evaluation

- Compare against a random and a simple heuristic policy.
- Plot episode return over multiple random seeds.
- Report mean and variation, not only the best run.
- Evaluate with exploration disabled or greatly reduced.
- Inspect failure states and episode length.
- Separate training environments from evaluation seeds where possible.

## Common pitfalls

- Reward hacking: the agent optimizes the reward definition, not necessarily the intended goal.
- Sparse rewards: useful feedback may be rare.
- Incorrect terminal handling: bootstrapping past terminal states changes the target.
- Unstable learning: learning rate and exploration schedule matter.
- State aliasing: a state representation that hides relevant history breaks the Markov assumption.

## Practice

Train Q-learning on a small grid world with obstacles and a goal. Change the reward for each step, compare learning curves, and inspect whether the learned path reflects the intended objective.

## References

- [Sutton and Barto: Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)
- [Gymnasium documentation](https://gymnasium.farama.org/)

**Key takeaway:** RL is a sequential decision problem. Define states, rewards, termination, and evaluation carefully before tuning an algorithm.