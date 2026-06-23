# RL Curriculum Repository — Design Spec
**Date:** 2026-06-23  
**Scope:** Full repository scaffold for a 12-week deep RL curriculum, from MDP foundations to VLM alignment

---

## Overview

A self-study deep RL curriculum implemented as a git repository. Seven modules, 19 assignment notebooks, self-contained lecture notes with full derivations, and a shared internal package (`rllearn/`) that grows as assignments are completed. Targets someone with strong ML fundamentals who wants to go from RL basics to RLHF/GRPO/VLM alignment research.

---

## 1. Repository Structure

```
rl-learn/
├── pyproject.toml                  # uv-managed; defines rllearn package + all deps
├── README.md                       # curriculum overview (already written)
├── .python-version                 # pinned Python version for uv
│
├── rllearn/                        # shared internal package (pre-scaffolded stubs)
│   ├── __init__.py
│   ├── buffers.py                  # ReplayBuffer, PrioritizedReplayBuffer
│   ├── networks.py                 # MLP, DuelingNet, ActorCritic, PolicyNet, GaussianPolicy
│   ├── envs.py                     # gym wrappers (frame stack, normalize obs, record video)
│   ├── utils.py                    # GAE, compute_returns, set_seed, explained_variance
│   └── logging.py                  # TensorBoard SummaryWriter wrapper (make_writer)
│
├── module_01_foundations/
│   ├── lecture_notes.md
│   ├── assignment1_mdp_gridworld.ipynb
│   └── assignment2_dynamic_programming.ipynb
│
├── module_02_value_based/
│   ├── lecture_notes.md
│   ├── assignment1_tabular_qlearning.ipynb
│   ├── assignment2_deep_qnetwork.ipynb
│   └── assignment3_rainbow_components.ipynb
│
├── module_03_policy_based/
│   ├── lecture_notes.md
│   ├── assignment1_reinforce.ipynb
│   ├── assignment2_actor_critic.ipynb
│   └── assignment3_ppo_from_scratch.ipynb
│
├── module_04_model_based/
│   ├── lecture_notes.md
│   ├── assignment1_dyna_q.ipynb
│   └── assignment2_world_models.ipynb
│
├── module_05_advanced_policy/
│   ├── lecture_notes.md
│   ├── assignment1_trpo_natural_gradient.ipynb
│   ├── assignment2_soft_actor_critic.ipynb
│   └── assignment3_multi_agent_intro.ipynb
│
├── module_06_rl_for_llms/
│   ├── lecture_notes.md
│   ├── assignment1_reward_modeling.ipynb
│   ├── assignment2_rlhf_pipeline.ipynb
│   └── assignment3_grpo_implementation.ipynb
│
├── module_07_alignment_frontier/
│   ├── lecture_notes.md
│   ├── assignment1_dpo_vs_rlhf.ipynb
│   ├── assignment2_process_reward_models.ipynb
│   └── assignment3_vlm_alignment.ipynb
│
├── projects/
│   ├── midterm_project.md
│   └── final_project.md
│
└── docs/
    └── superpowers/specs/
        └── 2026-06-23-rl-curriculum-design.md   ← this file
```

---

## 2. Shared Package — `rllearn/`

### Philosophy
Pre-scaffolded from day one. Every file exists with stub functions that raise `NotImplementedError`. Later notebooks can `from rllearn.buffers import ReplayBuffer` immediately — the import works, but the implementation is broken until the student completes the earlier assignment. This teaches package structure alongside algorithms.

### Stub Pattern

```python
# rllearn/buffers.py
class ReplayBuffer:
    """Circular replay buffer for off-policy RL.
    
    Implemented in: Module 02, Assignment 2.
    Used in: Module 02 A2, A3; Module 05 A2 (SAC).
    """
    def __init__(self, capacity: int):
        # TODO (Module 02, Assignment 2): initialize storage
        # Hint: use a collections.deque or pre-allocated numpy arrays
        raise NotImplementedError

    def push(self, state, action, reward, next_state, done):
        # TODO: store a single transition
        raise NotImplementedError

    def sample(self, batch_size: int):
        # TODO: return a random minibatch
        # Returns: (states, actions, rewards, next_states, dones) as numpy arrays
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
```

### `rllearn/` Module Map

| File | Contents | First implemented |
|---|---|---|
| `buffers.py` | `ReplayBuffer`, `PrioritizedReplayBuffer`, `RolloutBuffer` | Module 02 A2 |
| `networks.py` | `MLP`, `DuelingNet`, `ActorCritic`, `GaussianPolicyHead` | Module 02 A2 |
| `envs.py` | `make_env`, `NormalizeObsWrapper`, `RecordEpisodeStats` | Module 02 A2 |
| `utils.py` | `compute_returns`, `compute_gae`, `set_seed`, `explained_variance` | Module 03 A2 |
| `logging.py` | `make_writer`, `log_hparams` | Module 02 A2 |

---

## 3. Environment Setup

**Package manager:** `uv`  
**Setup:** `uv sync && uv run jupyter notebook`

```toml
# pyproject.toml
[project]
name = "rl-learn"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3",
    "gymnasium[classic-control,box2d,mujoco]>=0.29",
    "numpy>=1.26",
    "tensorboard>=2.17",
    "jupyter>=1.0",
    "matplotlib>=3.9",
    "tqdm>=4.66",
    "trl>=0.9",
    "transformers>=4.43",
    "datasets>=2.20",
    "peft>=0.12",
]

[tool.uv]
dev-dependencies = ["pytest>=8.0", "ipykernel>=6.29"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["rllearn"]
```

---

## 4. Notebook Format

Every assignment notebook follows this exact section structure:

### Header Block (Markdown)
- Prerequisites: which `rllearn` components must be implemented first, which lecture note sections to read
- Learning objectives: 3–4 bullets of what the student will understand after completing

### Part 1: Theory Recap (Markdown cells only)
- Key equations rendered in LaTeX
- 2–3 sentence intuition per equation
- No implementation — just the math about to be coded

### Part 2: Implementation (Code skeleton cells)
- Functions with docstrings, shape annotations, `# TODO` markers
- Each TODO includes: the equation it implements, expected input/output shapes, a `# Common mistake:` callout for the hard parts
- `raise NotImplementedError` as the default body (forces the student to implement before running)

### Part 3: Training & Verification (Partially filled)
- Training loop provided — not a TODO
- TensorBoard logging wired in
- Explicit success threshold stated: e.g., "CartPole should solve (mean reward ≥ 475 over 100 episodes) within 500 episodes"
- `%tensorboard --logdir runs/` cell at the top

### Part 4: Ablations (Guided experiments)
- 3–4 experiments: specific hyperparameter to change, what to observe, why it matters
- Each ablation is one code cell + one Markdown cell for the student's observation

### Part 5: Reflection (Markdown cell, student fills in)
- 2–3 questions connecting the implementation to the research bridge
- Example: "Why does Double DQN fix overestimation? Where does the same problem appear in RLHF reward hacking?"

---

## 5. Lecture Notes Format

Each `lecture_notes.md` is a self-contained chapter. No external dependencies required to understand it — all derivations are written out in full.

### Section Structure

```
# Module N: [Title]
> One-sentence framing of why this module matters for LLM/alignment research

## 1. Intuition
Plain English. No equations. Analogy or motivating example.

## 2. Formal Setup
Full symbol definitions. Every variable introduced explicitly before use.

## 3. Derivations
Step-by-step. Every non-obvious step gets a "(Why? Because...)" inline annotation.
Equations in display LaTeX with boxed final results.

## 4. Algorithm
Numbered pseudocode block. Clean and unambiguous.

## 5. Failure Modes & Hyperparameter Intuitions
What breaks and why. What each hyperparameter actually controls.
Empirical rules of thumb.

## 6. Research Bridge
How this exact concept appears in RLHF / GRPO / VLM alignment.
Links to the specific equation in the frontier paper.

## Appendix: Proofs
Full proofs for key theorems. Skippable on first read.
```

### Derivation Style

Equations in display math with boxed results:

```markdown
$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^\pi(s_t, a_t)\right]}$$
```

Every non-obvious step annotated inline:

```markdown
*Why $\nabla_\theta \log p_\theta(x)$?* Because $\nabla_\theta p_\theta(x) = p_\theta(x) \cdot \nabla_\theta \log p_\theta(x)$
by the chain rule on log — this moves the gradient inside the expectation so we can estimate it with samples.
```

---

## 6. TensorBoard Logging

**In-notebook pattern:**

```python
# Top of every training notebook
%load_ext tensorboard
%tensorboard --logdir runs/

from rllearn.logging import make_writer
writer = make_writer("runs/dqn_cartpole")

# Inside training loop (provided, not a TODO):
writer.add_scalar("train/episode_reward", ep_reward, global_step)
writer.add_scalar("train/td_loss", loss.item(), global_step)
writer.add_scalar("train/q_value_mean", q_values.mean().item(), global_step)
writer.close()
```

`rllearn/logging.py` is a thin wrapper: `SummaryWriter` with a consistent `runs/<algo>_<env>_<timestamp>` naming convention. No abstraction beyond that.

---

## 7. Solutions Branch

- Branch name: `solutions`
- Contains all `# TODO` filled in with correct implementations
- All `raise NotImplementedError` replaced with working code
- Rebased on `master` when lecture notes or structure changes
- **Not linked from README** — students find it via git if they choose to

---

## 8. Scope Boundaries

**In scope:**
- All 7 module directories with lecture notes and assignment notebooks
- `rllearn/` package with pre-scaffolded stubs
- `pyproject.toml`, `.python-version`, `README.md`
- `projects/midterm_project.md` and `projects/final_project.md`

**Out of scope:**
- CI/CD or automated testing of student implementations
- Docker/devcontainer setup
- Pre-trained model checkpoints
- Video recordings or slides

---

## 9. Implementation Order

1. `pyproject.toml` + `.python-version` + `uv sync` verification
2. `rllearn/` package — all stub files
3. Module 01: `lecture_notes.md` + 2 notebooks
4. Module 02: `lecture_notes.md` + 3 notebooks (implements `ReplayBuffer`, `MLP`, `make_env`)
5. Module 03: `lecture_notes.md` + 3 notebooks (implements `GAE`, `RolloutBuffer`, `ActorCritic`)
6. Module 04: `lecture_notes.md` + 2 notebooks
7. Module 05: `lecture_notes.md` + 3 notebooks (implements `GaussianPolicyHead`, `PrioritizedReplayBuffer`)
8. Module 06: `lecture_notes.md` + 3 notebooks
9. Module 07: `lecture_notes.md` + 3 notebooks
10. `projects/` files
11. Initial commit on `master`; create `solutions` branch
