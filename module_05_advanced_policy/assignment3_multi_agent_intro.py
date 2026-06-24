# %% [markdown]
# # Module 05, Assignment 3: Multi-Agent RL Introduction
#
# ## Prerequisites
# - Module 02 A1: TabularQAgent (for inspiration)
# - Lecture notes section 7
#
# ## Learning Objectives
# 1. Understand the non-stationarity challenge in MARL
# 2. Implement independent Q-learning for cooperative agents
# 3. Observe emergent coordination failure and success
# 4. Understand centralized training / decentralized execution

# %% [markdown]
# ## Part 0: Theory Recap
#
# In single-agent RL, the environment is **stationary** — the transition function $P(s'|s,a)$
# doesn't change during training. In MARL, each agent's "environment" includes other agents
# whose policies are also changing. This **non-stationarity** breaks the convergence guarantees
# of standard Q-learning.
#
# **Independent Q-learning (IQL):** Each agent trains its own Q-function independently,
# treating other agents as part of the environment. Simple to implement, but non-stationarity
# can cause instability — the Q-values of agent $i$ shift as agent $j$ updates.
#
# **Centralized Training / Decentralized Execution (CTDE):** During training, a centralized
# critic sees the full global state and all agents' actions. At execution, each agent uses
# only its own observation. This is the key insight of QMIX, MADDPG, and similar methods.

# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random

# %% [markdown]
# ## Part 1: Implement the Cooperative Gridworld
#
# Implement `CoopGridWorld` — a simple 2D gridworld where two agents must both reach
# their respective goals *simultaneously* to get a positive reward.
#
# **Rules:**
# - Grid is `grid_size x grid_size`.
# - Agent 0 starts at (0, 0), goal at (grid_size-1, grid_size-1).
# - Agent 1 starts at (grid_size-1, 0), goal at (0, grid_size-1).
# - Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT (clipped at walls).
# - Episode ends when both agents are at their goals simultaneously, or max_steps=50.
# - **Cooperative reward:** Both agents get +1.0 only when *both* are at their goals.
# - Each agent's observation is its own (row, col) position flattened to a single integer
#   (state index = row * grid_size + col).
#
# *(Why simultaneous goal requirement? This creates a coordination challenge:
# an agent at its goal must "wait" (or leave and return) for the other to arrive.
# Without coordination, agents may reach goals at different times and get no reward.)*

# %%
class CoopGridWorld:
    """Two-agent cooperative gridworld. Both must reach their goals simultaneously.

    Observation space: per-agent integer state index (row * grid_size + col).
    Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
    Reward: +1.0 to both agents only when both are at their goals simultaneously.
    Episode ends when both at goals, or after max_steps.
    """

    ACTIONS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    def __init__(self, grid_size: int = 5, n_agents: int = 2, max_steps: int = 50):
        """
        Args:
            grid_size: side length of the square grid
            n_agents: number of agents (use 2 for this assignment)
            max_steps: maximum steps per episode before forced termination
        """
        raise NotImplementedError

    def reset(self) -> List[np.ndarray]:
        """Reset environment. Return list of per-agent observations (integer state indices).

        Agent 0 starts at (0, 0), goal at (grid_size-1, grid_size-1).
        Agent 1 starts at (grid_size-1, 0), goal at (0, grid_size-1).

        Return: [obs_agent_0, obs_agent_1] where each obs is np.array([state_idx]).
        """
        raise NotImplementedError

    def step(self, actions: List[int]) -> Tuple[List, List[float], List[bool], dict]:
        """Execute joint action.

        Args:
            actions: list of integer actions, one per agent

        Returns:
            obs_list: [obs_agent_0, obs_agent_1] — updated positions as state indices
            reward_list: [r0, r1] — both get +1 if both at goals, else 0
            done_list: [done, done] — True if both at goals or max_steps reached
            info: dict with 'both_at_goal', 'step'

        *(Why shared reward? In cooperative settings, individual rewards can lead to
        agents pursuing their own goals without coordinating — the classic tragedy of the commons.
        Shared reward aligns incentives: the only way to get reward is collective success.)*
        """
        raise NotImplementedError

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) to integer state index."""
        row, col = pos
        return row * self.grid_size + col

    def _clip_pos(self, row: int, col: int) -> Tuple[int, int]:
        """Clip position to stay within grid bounds."""
        return (max(0, min(self.grid_size - 1, row)),
                max(0, min(self.grid_size - 1, col)))


# %% [markdown]
# ## Part 2: Implement Independent Q-Learner
#
# Each agent maintains its own Q-table: `Q[state, action]`.
# At each step, the agent updates its Q-value using the standard Bellman update,
# treating the other agent's policy as fixed (part of the environment).
#
# *(Why does this cause non-stationarity? Because as agent 1's policy changes,
# the distribution of next states that agent 0 sees also changes, invalidating
# agent 0's Q-values that were computed under the old policy of agent 1.)*

# %%
class IndependentQLearner:
    """Tabular Q-learner for use in MARL (independent learning).

    Each agent maintains its own Q-table and updates independently.
    Q-table shape: (n_states, n_actions).
    """

    def __init__(self, agent_id: int, n_states: int, n_actions: int,
                 alpha: float, gamma: float, epsilon: float):
        """
        Args:
            agent_id: identifier for this agent (for logging)
            n_states: total number of discrete states (grid_size^2)
            n_actions: number of actions (4 for gridworld)
            alpha: learning rate for Q-update
            gamma: discount factor
            epsilon: exploration probability for epsilon-greedy
        """
        raise NotImplementedError

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection.

        With probability epsilon: sample random action.
        With probability 1-epsilon: choose argmax_a Q[state, a].

        Args:
            state: integer state index

        Returns:
            integer action
        """
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """Standard Q-learning update.

        Q[s, a] <- Q[s, a] + alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])
        If done: bootstrap target = r (no next state value).

        Args:
            state: integer state index
            action: action taken
            reward: reward received
            next_state: integer next state index
            done: whether episode ended

        Returns:
            float: TD error = r + gamma * max Q(s', .) - Q(s, a)

        *(Why not use other agents' Q-tables? In independent Q-learning, each agent
        only uses its own information. This is the defining feature of IQL — it's
        equivalent to running N single-agent RL algorithms in parallel.)*
        """
        raise NotImplementedError


# %% [markdown]
# ## Part 3: Training Loop — Independent Q-Learning
#
# The training loop below is provided. It runs two IndependentQLearners on CoopGridWorld.
#
# **TensorBoard metrics:**
# - `train/joint_success_rate` (rolling 100-episode window)
# - `train/episode_length`
#
# **Verification:** Joint success rate >= 0.8 within 5000 episodes.

# %%
%load_ext tensorboard
%tensorboard --logdir runs/

# %%
from rllearn.logging import make_writer


def train_iql(
    grid_size: int = 5,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 3000,
    seed: int = 42,
    reward_mode: str = "cooperative",  # "cooperative" or "individual"
) -> Tuple[List[float], List[int]]:
    """Train two independent Q-learners on CoopGridWorld.

    reward_mode:
        "cooperative": both agents get +1 when both reach goals (shared reward)
        "individual": each agent gets +1 when it individually reaches its goal
    Returns:
        (success_rates, episode_lengths)
    """
    np.random.seed(seed)
    random.seed(seed)

    env = CoopGridWorld(grid_size=grid_size)
    n_states = grid_size * grid_size
    n_actions = 4

    agents = [
        IndependentQLearner(
            agent_id=i,
            n_states=n_states,
            n_actions=n_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon_start,
        )
        for i in range(2)
    ]

    writer = make_writer(f"iql_coop_gridworld_{reward_mode}")
    success_history = []
    episode_lengths = []
    rolling_successes = []

    for ep in range(n_episodes):
        # Linear epsilon decay
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * ep / epsilon_decay
        )
        for agent in agents:
            agent.epsilon = epsilon

        obs_list = env.reset()
        states = [int(obs[0]) for obs in obs_list]
        ep_len = 0
        ep_success = False

        for step in range(50):  # max_steps per episode
            actions = [agents[i].select_action(states[i]) for i in range(2)]
            next_obs_list, reward_list, done_list, info = env.step(actions)
            next_states = [int(obs[0]) for obs in next_obs_list]
            done = done_list[0]  # both agents share done signal

            # Reward mode
            if reward_mode == "individual":
                # Each agent gets reward only when it is at its own goal
                # (Requires a custom reward signal — we approximate by checking per-agent goal)
                goals = [
                    (grid_size - 1, grid_size - 1),  # agent 0 goal
                    (0, grid_size - 1),               # agent 1 goal
                ]
                individual_rewards = [0.0, 0.0]
                for i in range(2):
                    pos = divmod(states[i], grid_size)
                    if pos == goals[i]:
                        individual_rewards[i] = 1.0
                rewards_used = individual_rewards
            else:
                rewards_used = reward_list  # cooperative: both get +1 only if both at goals

            # Update each Q-learner independently
            for i in range(2):
                agents[i].update(states[i], actions[i], rewards_used[i],
                                 next_states[i], done)

            states = next_states
            ep_len += 1

            if done:
                ep_success = info.get("both_at_goal", False)
                break

        success_history.append(1.0 if ep_success else 0.0)
        episode_lengths.append(ep_len)
        rolling_successes.append(1.0 if ep_success else 0.0)

        # Rolling 100-episode success rate
        if len(rolling_successes) > 100:
            rolling_successes.pop(0)
        success_rate = np.mean(rolling_successes)

        writer.add_scalar("train/joint_success_rate", success_rate, ep)
        writer.add_scalar("train/episode_length", ep_len, ep)

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1:5d} | Success rate (last 100): {success_rate:.2f} "
                  f"| Ep len: {ep_len} | Epsilon: {epsilon:.3f}")

    writer.close()
    env.close()
    return success_history, episode_lengths


# %%
print("Training Independent Q-Learners (cooperative reward)...")
success_coop, lengths_coop = train_iql(
    grid_size=5,
    n_episodes=5000,
    reward_mode="cooperative",
    seed=42,
)

# Verification
rolling_100 = np.convolve(success_coop, np.ones(100)/100, mode='valid')
max_success = float(np.max(rolling_100))
final_success = float(np.mean(success_coop[-100:]))
print(f"\nMax rolling-100 success rate: {max_success:.2f}")
print(f"Final 100-episode success rate: {final_success:.2f}")

if final_success >= 0.8:
    print("✓ CoopGridWorld: IQL converged (success rate >= 0.8)")
else:
    print(f"✗ CoopGridWorld: not yet converged (need >= 0.8). Check implementations.")

# %% [markdown]
# ## Part 4: Ablation — Cooperative vs Individual Reward
#
# Train with individual reward (each agent rewarded only for its own goal)
# and compare joint success rate against cooperative reward.
#
# **Expected:** Individual reward leads to lower joint success — agents prioritize
# their own goals without coordinating, so they rarely arrive simultaneously.

# %%
print("Training Independent Q-Learners (individual reward)...")
success_indiv, lengths_indiv = train_iql(
    grid_size=5,
    n_episodes=5000,
    reward_mode="individual",
    seed=42,
)

# %%
def rolling_mean(values, window=100):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(rolling_mean(success_coop, 100), label="Cooperative reward", color='steelblue')
axes[0].plot(rolling_mean(success_indiv, 100), label="Individual reward", color='darkorange')
axes[0].axhline(y=0.8, color='gray', linestyle='--', label="Target: 0.8")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Joint Success Rate (rolling 100)")
axes[0].set_title("IQL: Cooperative vs Individual Reward")
axes[0].legend()

axes[1].plot(rolling_mean(lengths_coop, 100), label="Cooperative reward", color='steelblue')
axes[1].plot(rolling_mean(lengths_indiv, 100), label="Individual reward", color='darkorange')
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Episode Length (rolling 100)")
axes[1].set_title("IQL: Episode Length Comparison")
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Cooperative final success rate: {float(np.mean(success_coop[-100:])):.2f}")
print(f"Individual final success rate:  {float(np.mean(success_indiv[-100:])):.2f}")

# %% [markdown]
# ## Part 5: Reflection
#
# **Q1:** Why does independent Q-learning struggle with coordination?
# Specifically: what happens to agent 0's Q-values when agent 1's policy changes
# during training? How does this violate the stationarity assumption of Q-learning?

# %% [markdown]
# **Answer Q1:**
# (fill in)

# %% [markdown]
# **Q2:** How does Centralized Training / Decentralized Execution (CTDE) help?
# In CTDE (e.g., QMIX), a centralized critic sees the full joint state $(s, a^1, a^2)$
# during training. How does this solve the non-stationarity problem? Why can the
# agents still act decentrally at execution time?

# %% [markdown]
# **Answer Q2:**
# (fill in)

# %% [markdown]
# **Q3:** The QMIX mixing network enforces monotonicity:
# $\partial Q_\text{tot} / \partial Q^i \geq 0$ for all $i$.
# Why is this constraint necessary? What would go wrong if a centralized Q-function
# simply summed individual Q-values (without the monotonicity constraint)?
# Consider what happens at the optimal joint action.

# %% [markdown]
# **Answer Q3:**
# (fill in)
