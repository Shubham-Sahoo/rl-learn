# Deep RL Curriculum — From Foundations to VLM Alignment

A self-study deep reinforcement learning curriculum that takes you from MDP foundations to RLHF/GRPO/VLM alignment research in 12 weeks. Seven modules, 19 hands-on assignment notebooks, and a pre-scaffolded `rllearn/` package that grows as you complete assignments.

**Target audience:** Strong ML fundamentals (linear algebra, calculus, PyTorch basics) but new to RL.

---

## Philosophy

This curriculum is built on three core convictions:

1. **Intuition before equations.** Every chapter starts with plain English, pictures, or analogies. Then we formalize. Equations are annotated with inline explanations of why they're written the way they are.

2. **Build from scratch.** You implement core algorithms yourself — not tweaking hyperparameters in a pre-written trainer. This means GridWorld MDPs by hand, Q-learning with numpy tables, DQN with raw PyTorch. You'll understand what can go wrong.

3. **Connect to the frontier.** Every concept links concretely to modern LLM alignment work: RLHF pipeline architecture, reward hacking in language models, process reward models for reasoning, VLM hallucination via RL. Theory and research are the same conversation.

---

## Repository Structure

```
rl-learn/
├── pyproject.toml                          # uv-managed dependencies
├── README.md                               # (this file)
├── .python-version                         # Python 3.11
│
├── rllearn/                                # Shared package (stubs → implementations)
│   ├── __init__.py
│   ├── buffers.py                          # ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer
│   ├── networks.py                         # MLP, DuelingNet, ActorCritic, PolicyNet, GaussianPolicyHead, TwinQNetwork
│   ├── envs.py                             # make_env, NormalizeObsWrapper, RecordEpisodeStats
│   ├── utils.py                            # compute_returns, compute_gae, set_seed, explained_variance
│   └── logging.py                          # TensorBoard wrapper: make_writer
│
├── module_01_foundations/
│   ├── lecture_notes.md                    # MDPs, Bellman equations, Dynamic Programming
│   ├── assignment1_mdp_gridworld.ipynb     # Build stochastic GridWorld, policy evaluation
│   └── assignment2_dynamic_programming.ipynb # Policy iteration, value iteration
│
├── module_02_value_based/
│   ├── lecture_notes.md                    # Q-learning, DQN, Double DQN, Dueling DQN, PER
│   ├── assignment1_tabular_qlearning.ipynb # Q-learning on Taxi; SARSA on CliffWalking
│   ├── assignment2_deep_qnetwork.ipynb     # Implement ReplayBuffer + MLP, train DQN on CartPole
│   └── assignment3_rainbow_components.ipynb # Double DQN, Dueling DQN, Prioritized Replay on LunarLander
│
├── module_03_policy_based/
│   ├── lecture_notes.md                    # Policy gradient, REINFORCE, Actor-Critic, PPO, GAE
│   ├── assignment1_reinforce.ipynb         # REINFORCE from scratch on CartPole
│   ├── assignment2_actor_critic.ipynb      # Implement compute_gae, train A2C on LunarLander
│   └── assignment3_ppo_from_scratch.ipynb  # Implement PPO-Clip, train on MuJoCo (HalfCheetah/Ant)
│
├── module_04_model_based/
│   ├── lecture_notes.md                    # Dyna-Q, world models, RSSM, model exploitation
│   ├── assignment1_dyna_q.ipynb            # Dyna-Q agent with planning steps
│   └── assignment2_world_models.ipynb      # Transition + reward models, model-based rollouts
│
├── module_05_advanced_policy/
│   ├── lecture_notes.md                    # Fisher Information, TRPO, SAC, maximum entropy RL, MARL intro
│   ├── assignment1_trpo_natural_gradient.ipynb # Conjugate gradient, natural gradient steps
│   ├── assignment2_soft_actor_critic.ipynb # SAC: twin Q-nets, reparameterization, automatic temperature
│   └── assignment3_multi_agent_intro.ipynb # Independent learners in cooperative gridworld
│
├── module_06_rl_for_llms/
│   ├── lecture_notes.md                    # Reward modeling, RLHF pipeline, GRPO, DPO
│   ├── assignment1_reward_modeling.ipynb   # Bradley-Terry model on IMDB sentiment
│   ├── assignment2_rlhf_pipeline.ipynb     # KL penalty, full RLHF training on GPT-2
│   └── assignment3_grpo_implementation.ipynb # Group sampling, group-normalized rewards
│
├── module_07_alignment_frontier/
│   ├── lecture_notes.md                    # DPO vs RLHF, PRMs, VLM alignment, hallucination
│   ├── assignment1_dpo_vs_rlhf.ipynb       # DPO loss, implicit reward extraction
│   ├── assignment2_process_reward_models.ipynb # PRM for step-level scoring, best-of-N with PRM
│   └── assignment3_vlm_alignment.ipynb     # CHAIR metric, hallucination reward model, VLM-DPO
│
├── projects/
│   ├── midterm_project.md                  # 3 project options (Weeks 5–7)
│   └── final_project.md                    # 3 research project options (Weeks 9–12)
│
└── docs/
    └── superpowers/
        ├── specs/2026-06-23-rl-curriculum-design.md  # Full design spec
        └── plans/2026-06-23-rl-curriculum.md         # Implementation plan
```

---

## Module Summaries

### Module 01: MDP Foundations
**Weeks 1–2**

The foundations of credit assignment in sequential decision-making. You'll build a stochastic GridWorld from scratch and implement policy evaluation by solving Bellman equations iteratively. We derive the formal MDP tuple $(S, A, P, R, \gamma)$ and show that every agent's behavior is governed by the same recursive equations. Policy iteration and value iteration are two sides of the same coin: iteratively improving your estimate of the optimal policy. By the end, you understand why deep RL can work: the Bellman equations don't change if you replace a lookup table with a neural network.

**Key concepts:** Markov property, value functions ($V^\pi$, $Q^\pi$, $V^*$, $Q^*$), Bellman equations, policy evaluation, policy improvement, dynamic programming. **No rllearn imports.** Pure numpy.

---

### Module 02: Value-Based Methods
**Weeks 2–4**

Moving from tabular to deep. Q-Learning is a temporal difference (TD) algorithm that updates estimates one step at a time—more efficient than Monte Carlo but more biased. Deep Q-Networks (DQN) scale this to continuous observation spaces using experience replay and target networks to stabilize learning. The module covers the key insights: why replay decorrelates samples (reducing variance), why a separate target network is crucial (breaking feedback loops), and how Double DQN fixes overestimation bias in Q-values. You'll implement the ReplayBuffer, MLP, and environment wrappers that all downstream algorithms depend on. Rainbow components (dueling networks, prioritized replay) show the landscape of improvements.

**Key concepts:** Off-policy learning, TD learning, experience replay, target networks, overestimation bias, Double DQN, Dueling DQN, prioritized experience replay. **Implements:** ReplayBuffer, MLP, make_env, PrioritizedReplayBuffer.

---

### Module 03: Policy-Based Methods & Actor-Critic
**Weeks 4–6**

Why estimate state values when you can directly optimize the policy? The policy gradient theorem reveals that you only need to weight actions by their advantage ($Q - V$). REINFORCE is the simplest algorithm: sample trajectories and weight log-probabilities by returns. It's unbiased but high-variance. The baseline trick (subtracting $V$) reduces variance without biasing the gradient. Actor-Critic methods replace Monte Carlo returns with learned value estimates (TD), enabling online learning. Generalized Advantage Estimation (GAE) interpolates between TD(0) and Monte Carlo, giving you a Goldilocks advantage estimate. PPO-Clip builds on this: instead of trust regions (TRPO), it clips probability ratios to prevent catastrophic policy collapse. By assignment 3, you'll implement PPO from scratch—the most widely-used RL algorithm in practice.

**Key concepts:** Policy gradient theorem, REINFORCE, baseline, actor-critic, advantage function, GAE, trust regions, PPO. **Implements:** compute_returns, compute_gae, ActorCritic, RolloutBuffer, NormalizeObsWrapper.

---

### Module 04: Model-Based RL
**Weeks 6–7**

What if the agent builds a mental model of the environment and uses it to plan? Dyna-Q learns a model from experience and runs "imagination" to improve the policy—fewer environment interactions needed. The tradeoff: model errors compound over long imagined trajectories. World models learn to predict future observations (transition model) and rewards in a learned latent space. This module is about when model-based RL helps (sample efficiency in known environments) and when it hurts (exploration, model error). By the end, you understand the connection to modern work: chain-of-thought reasoning as environment simulation, AlphaZero's tree search, constitutional AI self-play.

**Key concepts:** Model learning, planning, imagination, Dyna-Q, world models, model exploitation, model error compounding. **Minimal rllearn imports**—mostly builds new patterns.

---

### Module 05: Advanced Policy Optimization
**Weeks 7–8**

Refining the policy gradient. The Fisher Information Matrix describes the geometry of policy space—moving in this geometry (natural gradient) is more stable than Euclidean parameter-space movement. TRPO uses this insight to enforce a KL constraint. Soft Actor-Critic (SAC) takes a different angle: add entropy regularization to the reward, turning a hard optimization problem into a soft one. Maximum entropy RL naturally prevents premature convergence and improves exploration. You'll implement SAC's twin Q-networks (reducing overestimation) and the reparameterization trick (enabling differentiable sampling). The final assignment brings multiple agents together, showing independent learners and cooperative credit assignment.

**Key concepts:** Natural gradient, Fisher Information, TRPO, maximum entropy RL, SAC, twin Q-networks, reparameterization trick, multiagent RL. **Implements:** GaussianPolicyHead, TwinQNetwork.

---

### Module 06: RL for Language Models
**Weeks 8–10**

Bridging RL and LLMs. The RLHF pipeline is conceptually simple: train a reward model (Bradley-Terry preference model) on human preference pairs, then use PPO to fine-tune the language model's policy while staying close to the reference model (via KL penalty). You'll implement this end-to-end. GRPO (Group-Relative Policy Optimization) removes the need for a value network by normalizing rewards within a group of samples. DPO (Direct Preference Optimization) goes further: it's a preference optimization algorithm that doesn't require an explicit reward model—the optimal policy has a closed form given preference pairs. By the end, you've seen the complete alignment pipeline: from training data to reference models to preference labels to fine-tuned language models.

**Key concepts:** Reward modeling, Bradley-Terry model, RLHF pipeline, KL penalty, soft trust regions, GRPO, DPO, token-level credit assignment.

---

### Module 07: Alignment Frontier
**Weeks 10–12**

The cutting edge of alignment research. Process Reward Models score individual reasoning steps rather than full completions—enabling step-level credit assignment and best-of-N search over reasoning paths. VLM alignment introduces new challenges: hallucinations (claiming objects that don't exist in an image), sycophancy (agreeing with incorrect image descriptions), grounding failures. The CHAIR metric quantifies hallucinations. You'll implement DPO adapted for multimodal inputs and a hallucination reward model. The final reflection: what are the limits of current RL-for-alignment approaches? Why is scalable oversight hard? What open problems matter most?

**Key concepts:** Process reward models, step-level scoring, best-of-N with PRMs, VLM alignment, hallucination metrics, scalable oversight.

---

## Schedule Overview

| Week | Module | Topics | Assignments | rllearn Implements |
|------|--------|--------|-------------|-------------------|
| 1–2 | 01: Foundations | MDPs, Bellman, DP | GridWorld, DP Algs | — |
| 2–4 | 02: Value-Based | Q-learning, DQN, Double DQN, Dueling DQN, PER | Tabular Q, DQN, Rainbow | ReplayBuffer, MLP, make_env, PrioritizedReplayBuffer |
| 4–6 | 03: Policy-Based | REINFORCE, A2C, GAE, PPO | REINFORCE, A2C, PPO | compute_returns, compute_gae, ActorCritic, RolloutBuffer |
| 6–7 | 04: Model-Based | Dyna-Q, World Models | Dyna-Q, World Models | — |
| 7–8 | 05: Advanced Policy | TRPO, SAC, MaxEnt RL, MARL | TRPO, SAC, CoopMA | GaussianPolicyHead, TwinQNetwork |
| 8–10 | 06: RL for LLMs | Reward Modeling, RLHF, GRPO | RM, RLHF, GRPO | — |
| 10–12 | 07: Alignment Frontier | PRMs, DPO, VLM alignment | DPO, PRMs, VLM-DPO | — |
| 5–7 | Midterm Project | Open-ended RL project | Custom environment or offline RL | — |
| 9–12 | Final Project | Research project | GRPO, VLM alignment, or diffusion RL | — |

---

## Setup

### Prerequisites

- **Python 3.11+**
- **PyTorch 2.3+** (CPU or GPU)
- **Gymnasium 0.29+** (for classic control, Box2D, MuJoCo)
- **GPU recommended** for Modules 6–7 (LLM fine-tuning), but assignments can run on CPU with longer walltime

### Installation

Clone the repository and install dependencies via `uv`:

```bash
git clone https://github.com/your-username/rl-learn.git
cd rl-learn

# Install Python 3.11 (if not already installed)
# e.g., via pyenv: pyenv install 3.11.0

# Install uv (https://docs.astral.sh/uv/getting-started/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

### Running Assignments

Each module directory contains Jupyter notebooks. Start the notebook server:

```bash
uv run jupyter notebook
```

Then open the desired assignment (e.g., `module_01_foundations/assignment1_mdp_gridworld.ipynb`).

### TensorBoard Logging

Training code logs metrics to TensorBoard. View live results with:

```bash
uv run tensorboard --logdir runs/
```

Then navigate to `http://localhost:6006/` in your browser.

---

## How to Use This Repository

1. **Read the lecture notes first.** Each module's `lecture_notes.md` is self-contained; no external textbooks required. All equations are derived in full.

2. **Work through assignments in order.** Each assignment builds on the previous module's rllearn implementations. The notebooks are structured as:
   - **Theory Recap:** Key equations with intuitive explanations
   - **Implementation:** Skeleton code with TODOs and docstrings
   - **Training & Verification:** Pre-written training loop; you verify success criteria
   - **Ablations:** Guided experiments to test your understanding
   - **Reflection:** Questions connecting implementation to research

3. **Implement the rllearn package incrementally.** When an assignment says "Before continuing, implement `ReplayBuffer` in `rllearn/buffers.py`," do it. These stubs are used by downstream assignments. By the end, `rllearn/` is a full RL toolkit.

4. **Take the projects seriously.** The midterm and final projects let you apply the curriculum to novel problems. See `projects/midterm_project.md` and `projects/final_project.md`.

---

## Design & Implementation Philosophy

This curriculum embeds several pedagogical choices:

- **Pre-scaffolded package:** Every `rllearn/` module exists from day one with `raise NotImplementedError`. This teaches package structure alongside algorithms and allows forward references (later notebooks import from the package before implementations exist).

- **Full derivations:** No "trust me" steps. Every non-obvious algebraic move has an inline "(Why? Because...)" explanation. Every equation is boxed and rendered in GitHub-compatible LaTeX.

- **Concrete success thresholds:** "CartPole mean reward ≥ 450 over 50 episodes within 300 training episodes." Not vague. You know when you've succeeded.

- **TensorBoard everywhere:** Logging is wired into every training loop. Visualize learning in real-time.

- **Solutions branch:** Correct implementations live on the `solutions` branch. If stuck, you can peek—but the structure encourages trying first.

---

## Key Resources

- **Design Spec:** See `docs/superpowers/specs/2026-06-23-rl-curriculum-design.md` for architectural details, interface contracts, and module dependencies.
- **Implementation Plan:** See `docs/superpowers/plans/2026-06-23-rl-curriculum.md` for task breakdown and commit structure.

---

## Troubleshooting

### ImportError: No module named 'rllearn'
Ensure you've run `uv sync` and are using `uv run` to execute Python scripts/notebooks.

### Notebook kernel won't start
After `uv sync`, run:
```bash
uv run ipython kernel install --user --name rl-learn
```
Then select this kernel in the notebook.

### MuJoCo license errors (Module 3+)
MuJoCo is free for personal use. If you get license errors, check https://mujoco.org/. For CI/cloud environments, set `MUJOCO_GL=osmesa`.

### Out of memory on GPU (Module 6–7)
Reduce batch size or use quantization (4-bit via `bitsandbytes`). The assignments provide CPU fallbacks.

---

## Contributing & Extensions

This is a living curriculum. If you find errors, improve a derivation, or add a module:

1. Open an issue describing the improvement
2. Create a PR with a clear commit message
3. Assignments should follow the structure: Theory → Implementation (TODOs) → Verification → Ablations → Reflection

---

## License

[MIT License](LICENSE) — use freely in courses, personal study, or commercial projects.

---

## Acknowledgments

This curriculum synthesizes ideas from:
- Barto & Sutton's *Reinforcement Learning: An Introduction* (Bellman equations, DP, value/policy-based methods)
- Schulman et al.'s PPO paper and trajectory optimization lineage
- InstructGPT/RLHF papers (Ouyang et al., Christiano et al.)
- Recent work on process reward models (Lightman et al.) and VLM alignment (Chen et al.)

The pedagogical structure draws inspiration from CS231n (Stanford's CNN course) and Fast.ai's top-down teaching philosophy.

---

**Questions or feedback?** Open an issue or email us. Happy learning!
