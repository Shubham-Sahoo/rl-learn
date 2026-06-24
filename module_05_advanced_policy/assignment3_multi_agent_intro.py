# %% [markdown]
# # Module 05, Assignment 3: Multi-Agent RL Introduction

# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random

# %% [markdown]
# ## Part 1: Implement the Cooperative Gridworld

# %%
class CoopGridWorld:
    """Two-agent cooperative gridworld. Both must reach their goals simultaneously."""

    ACTIONS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    def __init__(self, grid_size: int = 5, n_agents: int = 2, max_steps: int = 50):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_steps = max_steps
        # Goals
        self.goals = [
            (grid_size - 1, grid_size - 1),  # agent 0 goal
            (0, grid_size - 1),               # agent 1 goal
        ]
        self.positions = [None, None]
        self.step_count = 0

    def reset(self) -> List[np.ndarray]:
        """Reset environment."""
        self.positions = [(0, 0), (self.grid_size - 1, 0)]
        self.step_count = 0
        return [np.array([self._pos_to_state(pos)]) for pos in self.positions]

    def step(self, actions: List[int]) -> Tuple[List, List[float], List[bool], dict]:
        """Execute joint action."""
        self.step_count += 1

        new_positions = []
        for i, (pos, action) in enumerate(zip(self.positions, actions)):
            dr, dc = self.ACTIONS[action]
            new_r, new_c = self._clip_pos(pos[0] + dr, pos[1] + dc)
            new_positions.append((new_r, new_c))
        self.positions = new_positions

        # Check both at goals
        both_at_goal = all(
            self.positions[i] == self.goals[i] for i in range(self.n_agents)
        )

        reward = 1.0 if both_at_goal else 0.0
        done = both_at_goal or self.step_count >= self.max_steps

        obs_list = [np.array([self._pos_to_state(pos)]) for pos in self.positions]
        reward_list = [reward] * self.n_agents
        done_list = [done] * self.n_agents
        info = {"both_at_goal": both_at_goal, "step": self.step_count}

        return obs_list, reward_list, done_list, info

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

# %%
class IndependentQLearner:
    """Tabular Q-learner for use in MARL (independent learning)."""

    def __init__(self, agent_id: int, n_states: int, n_actions: int,
                 alpha: float, gamma: float, epsilon: float):
        self.agent_id = agent_id
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """Standard Q-learning update."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return float(td_error)


# %% [markdown]
# ## Part 3: Training Loop — Independent Q-Learning

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
    reward_mode: str = "cooperative",
) -> Tuple[List[float], List[int]]:
    """Train two independent Q-learners on CoopGridWorld."""
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

    for ep in range(n_episodes):
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

        for step in range(50):
            actions = [agents[i].select_action(states[i]) for i in range(2)]
            next_obs_list, reward_list, done_list, info = env.step(actions)
            next_states = [int(obs[0]) for obs in next_obs_list]
            done = done_list[0]

            if reward_mode == "individual":
                goals = [
                    (grid_size - 1, grid_size - 1),
                    (0, grid_size - 1),
                ]
                ind_rewards = []
                for i in range(2):
                    pos_idx = next_states[i]
                    pos = (pos_idx // grid_size, pos_idx % grid_size)
                    ind_rewards.append(1.0 if pos == goals[i] else 0.0)
                rewards_to_use = ind_rewards
            else:
                rewards_to_use = reward_list

            for i in range(2):
                agents[i].update(states[i], actions[i], rewards_to_use[i],
                                 next_states[i], done)

            states = next_states
            ep_len += 1

            if info["both_at_goal"]:
                ep_success = True

            if done:
                break

        success_history.append(1 if ep_success else 0)
        episode_lengths.append(ep_len)

        if ep % 100 == 0:
            recent_sr = np.mean(success_history[-100:]) if len(success_history) >= 100 else np.mean(success_history)
            writer.add_scalar("train/joint_success_rate", recent_sr, ep)
            writer.add_scalar("train/episode_length", ep_len, ep)

    writer.close()

    # Compute rolling success rates
    window = 100
    success_rates = np.convolve(success_history, np.ones(window) / window, mode='valid').tolist()
    return success_rates, episode_lengths


# %% [markdown]
# ## Part 4: Verification

# %%
print("Training IQL with cooperative reward...")
success_rates_coop, ep_lengths_coop = train_iql(
    grid_size=5, n_episodes=5000, seed=42, reward_mode="cooperative"
)

if success_rates_coop:
    final_sr = success_rates_coop[-1]
    print(f"Final joint success rate (coop): {final_sr:.2f}")
    assert final_sr >= 0.8, f"IQL did not converge: success_rate={final_sr:.2f} (need >= 0.8)"
    print("✓ IQL converged with cooperative reward")

# %% [markdown]
# ## Part 5: Ablation — Individual vs Cooperative Reward

# %%
print("Training IQL with individual reward...")
success_rates_ind, ep_lengths_ind = train_iql(
    grid_size=5, n_episodes=5000, seed=42, reward_mode="individual"
)

plt.figure(figsize=(10, 4))
if success_rates_coop:
    plt.plot(success_rates_coop, label="Cooperative reward", color='steelblue')
if success_rates_ind:
    plt.plot(success_rates_ind, label="Individual reward", color='darkorange')
plt.axhline(y=0.8, color='red', linestyle='--', label='Target: 0.8')
plt.xlabel("Episode (window=100)")
plt.ylabel("Joint Success Rate")
plt.title("IQL: Cooperative vs Individual Reward")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 6: Reflection

# %% [markdown]
# **Answers:**
# 1. Non-stationarity: as agent 1 learns and its policy changes, agent 0's Q-values become
#    invalid since they were computed assuming agent 1's old policy. This violates the Markov
#    property from the perspective of either agent.
# 2. CTDE resolves non-stationarity by training a centralized critic that conditions on the full
#    global state and all agents' actions, while execution uses decentralized policies.
# 3. Individual reward leads to agents pursuing their own goals without waiting for the other,
#    resulting in lower joint success rates compared to shared cooperative reward.
