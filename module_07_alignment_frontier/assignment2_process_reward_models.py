# %% [markdown]
# # Module 07, Assignment 2: Process Reward Models
#
# ## Prerequisites
# - Module 06 A1: Reward modeling basics
# - Lecture notes sections 1–4
#
# ## Learning Objectives
# 1. Understand step-level credit assignment via PRM
# 2. Implement Monte Carlo step labeling
# 3. Implement Best-of-N and PRM beam search
# 4. Compare PRM vs ORM reranking on GSM8K-style problems

# %% [markdown]
# ## Part 0: Theory Recap — PRM vs ORM
#
# **ORM (Outcome Reward Model):**
# $$r_\text{ORM}(x, y_{1:T}) = \mathbf{1}[y_T = y^*] \quad \text{(sparse, terminal)}$$
#
# **PRM value interpretation:**
# $$V_\text{PRM}(s_t) = P(\text{correct final answer} \mid \text{reasoning prefix}_t)$$
#
# **PRM sequence score (product aggregation):**
# $$\text{score}_\text{prod}(y_{1:T}) = \prod_{t=1}^{T} V_\text{PRM}(s_t)$$
#
# **Monte Carlo step label:**
# $$\hat{p}_t = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}[\text{rollout}_j \text{ from step } t \text{ reaches correct answer}]$$
#
# *(Why is Monte Carlo labeling a form of TD(∞) estimation? Because we roll out all remaining
# steps to the terminal state, then average — this is exactly the Monte Carlo value estimate
# applied to the step-level Markov chain.)*

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Callable, Dict, Optional

from rllearn.logging import make_writer

# %% [markdown]
# ## Part 1: Process Reward Model
#
# Implement a simple PRM that scores individual reasoning steps represented as embeddings.
#
# **Architecture:** MLP(input_dim → hidden_dim → hidden_dim → 1) with sigmoid output.
# The sigmoid ensures the output is in [0, 1], interpretable as a probability.
#
# *(Why sigmoid instead of softmax? Because we're doing binary classification per step
# (correct/incorrect), not multi-class. Each step is judged independently.)*

# %%
class ProcessRewardModel(nn.Module):
    """Scores individual reasoning steps.

    Input: step embedding (pre-computed representation of a reasoning step)
    Output: step score in [0, 1] — P(step is correct / on the right track)
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        # TODO: MLP(input_dim → hidden_dim → hidden_dim → 1) with sigmoid output
        # Use ReLU activations between hidden layers.
        raise NotImplementedError

    def forward(self, step_embedding: torch.Tensor) -> torch.Tensor:
        """Return step score in [0,1] of shape (batch,).

        Args:
            step_embedding: Shape (batch, input_dim)

        Returns:
            scores: Shape (batch,) with values in [0, 1]
        """
        raise NotImplementedError

# %% [markdown]
# ## Part 2: Monte Carlo Step Labeling
#
# Estimate P(correct | step_t) by sampling N completions from step_t onward and checking
# whether each completion reaches the correct final answer.
#
# **Implementation notes:**
# - `oracle_fn` takes a complete solution string and returns True/False
# - You call `oracle_fn` on each of the N rolled-out completions
# - Return the fraction that returned True
#
# *(Why do we sample from the *current* step rather than the beginning? Because we want the
# label to be conditional on the prefix — P(correct | prefix_t), not P(correct | start).)*

# %%
def monte_carlo_step_label(solution_steps: List[str], step_idx: int,
                           oracle_fn: Callable[[str], bool],
                           policy_fn: Callable[[str], str],
                           n_rollouts: int = 16) -> float:
    """
    Estimate P(correct final answer | correct through step_idx) by sampling
    n_rollouts continuations from step_idx and checking oracle correctness.

    Args:
        solution_steps: List of reasoning steps (strings), len = T
        step_idx: The step index we are labeling (0-indexed)
        oracle_fn: Function that takes a complete solution string and returns True/False
        policy_fn: Function that takes a prefix string and returns a completion string
        n_rollouts: Number of rollout samples N

    Returns:
        label: Float in [0, 1] — fraction of rollouts reaching the correct answer
    """
    raise NotImplementedError


def label_all_steps(solution_steps: List[str],
                    oracle_fn: Callable[[str], bool],
                    policy_fn: Callable[[str], str],
                    n_rollouts: int = 16) -> List[float]:
    """
    Label every step in a solution using Monte Carlo estimation.

    Returns list of floats of length len(solution_steps), one per step.
    Step 0 label = P(correct | step_0 correct), step T-1 label ≈ oracle(final step).
    """
    return [monte_carlo_step_label(solution_steps, t, oracle_fn, policy_fn, n_rollouts)
            for t in range(len(solution_steps))]

# %% [markdown]
# ## Part 3: Best-of-N and Beam Search
#
# ### Best-of-N with PRM
#
# Generate N independent solutions, score each with the PRM (product of step scores),
# return the highest-scoring solution.
#
# *(Why use product? Because we treat each step as an independent check — all steps must be
# correct for the solution to be valid. Product = joint probability under independence.)*
#
# ### PRM Beam Search
#
# Instead of scoring complete solutions, guide generation step-by-step:
# 1. Start with `beam_width` copies of the empty prefix
# 2. At each step, expand each beam by generating one next step
# 3. Score all (beam_width × branching_factor) candidates with PRM
# 4. Keep only the top `beam_width` candidates by PRM score
# 5. Return the top-1 beam after `max_steps` steps

# %%
def score_solution_with_prm(prm: ProcessRewardModel,
                            step_embeddings: List[torch.Tensor],
                            aggregation: str = "product") -> float:
    """
    Score a complete solution using the PRM.

    Args:
        prm: Trained ProcessRewardModel
        step_embeddings: List of step embeddings, one per reasoning step
        aggregation: "product" or "min"

    Returns:
        score: Float scalar
    """
    # TODO: forward pass through PRM for each step embedding, then aggregate
    raise NotImplementedError


def best_of_n_with_prm(policy_fn: Callable[[str], Tuple[str, List[torch.Tensor]]],
                       prm: ProcessRewardModel,
                       prompt: str,
                       n: int = 16,
                       aggregation: str = "product") -> str:
    """
    Generate n solutions; score each with PRM (product of step scores);
    return highest-scoring solution.

    Args:
        policy_fn: Function that takes a prompt and returns (solution_str, step_embeddings)
        prm: Trained ProcessRewardModel
        prompt: The problem prompt
        n: Number of candidates to generate
        aggregation: "product" or "min" for step score combination

    Returns:
        best_solution: The solution string with the highest PRM score
    """
    raise NotImplementedError


def prm_beam_search(policy_step_fn: Callable[[str], List[Tuple[str, torch.Tensor]]],
                    prm: ProcessRewardModel,
                    prompt: str,
                    beam_width: int = 4,
                    max_steps: int = 8) -> str:
    """
    Beam search over reasoning steps guided by PRM step scores.
    At each step, expand all beams, score, keep top beam_width.

    Args:
        policy_step_fn: Function that takes a current prefix string and returns
                        list of (next_step_str, step_embedding) candidates
        prm: Trained ProcessRewardModel
        prompt: The problem prompt
        beam_width: Number of beams to maintain
        max_steps: Maximum number of reasoning steps

    Returns:
        best_solution: The complete solution string from the top beam
    """
    raise NotImplementedError

# %% [markdown]
# ## Part 4: Toy Math Task — Training and Evaluation
#
# We use a **toy arithmetic word problem generator** that provides:
# - Problems with known correct answers (enables Monte Carlo labeling)
# - Multi-step solutions decomposed into labeled reasoning steps
# - An oracle function for correctness checking
#
# **ORM baseline:** Score = 1 if final answer correct, 0 otherwise.
# **PRM:** Step-level scores via Monte Carlo labeling.
#
# **Verification target:** PRM Best-of-16 improves accuracy by ≥ 5% over ORM Best-of-16
# on held-out problems.

# %%
# --- Toy Task: Arithmetic Word Problems ---

def generate_arithmetic_problem(difficulty: int = 2,
                                seed: Optional[int] = None) -> Dict:
    """
    Generate a toy multi-step arithmetic word problem.

    Returns dict with:
        problem: str (the question)
        steps: List[str] (intermediate reasoning steps)
        answer: int (correct final answer)
        step_embeddings: List[torch.Tensor] (dummy embeddings for PRM training)
    """
    rng = random.Random(seed)
    nums = [rng.randint(1, 20) for _ in range(difficulty + 1)]
    ops = [rng.choice(["+", "-", "*"]) for _ in range(difficulty)]

    # Build the problem description
    parts = [str(nums[0])]
    running = nums[0]
    steps = []
    for i, (op, num) in enumerate(zip(ops, nums[1:])):
        parts.append(f"{op} {num}")
        if op == "+":
            running = running + num
        elif op == "-":
            running = running - num
        else:
            running = running * num
        steps.append(f"Step {i+1}: {' '.join(parts)} = {running}")

    problem = f"Calculate: {' '.join(parts)}"
    answer = running

    # Dummy step embeddings (in real use, these come from an LLM encoder)
    dim = 64
    step_embeddings = [
        torch.randn(1, dim) + (0.5 if i < len(steps) - 1 else 0.0)
        for i in range(len(steps))
    ]

    return {
        "problem": problem,
        "steps": steps,
        "answer": answer,
        "step_embeddings": step_embeddings,
    }


def toy_oracle(solution_str: str, correct_answer: int) -> bool:
    """Check if the solution string contains the correct answer."""
    return str(correct_answer) in solution_str.split()[-3:]


def make_prm_dataset(n_train: int = 200, n_val: int = 50,
                     n_rollouts: int = 16) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate PRM training data with Monte Carlo step labels.
    Each example: step_embedding + label in [0,1].

    PROVIDED — do not modify.
    """
    def noisy_policy(prefix: str) -> str:
        """Toy policy that sometimes makes errors."""
        # With 80% probability, return a plausible (noisy) continuation
        if random.random() < 0.8:
            return prefix + " [correct step]"
        else:
            return prefix + " [error: wrong calculation]"

    train_examples = []
    for i in range(n_train):
        prob = generate_arithmetic_problem(difficulty=2, seed=i)
        for step_idx, (emb, step_str) in enumerate(zip(prob["step_embeddings"], prob["steps"])):
            # Oracle: does the full solution contain the correct answer?
            def oracle(sol, ans=prob["answer"]): return toy_oracle(sol, ans)
            # Simple MC label: proportion of dummy rollouts that "succeed"
            # (In real training, you'd call monte_carlo_step_label)
            label = 1.0 - 0.2 * step_idx / max(len(prob["steps"]) - 1, 1)
            label = float(np.clip(label + np.random.normal(0, 0.1), 0, 1))
            train_examples.append({"embedding": emb.squeeze(0), "label": label})

    val_problems = [generate_arithmetic_problem(difficulty=2, seed=n_train + i)
                    for i in range(n_val)]

    return train_examples, val_problems


def train_prm(prm: ProcessRewardModel, train_data: List[Dict],
              n_epochs: int = 10, lr: float = 1e-3,
              device: str = "cpu", run_name: str = "prm_training") -> List[float]:
    """
    Train the PRM on step-level binary cross-entropy.

    PROVIDED — do not modify. Implement ProcessRewardModel above.
    """
    writer = make_writer(run_name)
    optimizer = torch.optim.Adam(prm.parameters(), lr=lr)
    prm.to(device).train()

    losses = []
    global_step = 0
    for epoch in range(n_epochs):
        random.shuffle(train_data)
        epoch_losses = []
        for item in train_data:
            emb = item["embedding"].unsqueeze(0).to(device)
            label = torch.tensor([[item["label"]]], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            pred = prm(emb).unsqueeze(-1)  # (1, 1)
            loss = F.binary_cross_entropy(pred, label)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            writer.add_scalar("train/prm_bce", loss.item(), global_step)
            global_step += 1
        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | BCE loss: {mean_loss:.4f}")

    writer.close()
    return losses


def evaluate_best_of_n(prm: Optional[ProcessRewardModel], val_problems: List[Dict],
                       n_candidates: int = 16,
                       use_prm: bool = True,
                       noise_std: float = 0.3) -> float:
    """
    Evaluate Best-of-N accuracy on val_problems.
    Candidates are generated by adding noise to the true step embeddings.

    PROVIDED — do not modify.
    """
    correct = 0
    for prob in val_problems:
        candidates = []
        for _ in range(n_candidates):
            # Simulate candidates: noisy versions of the true embeddings
            noisy_embs = [emb + torch.randn_like(emb) * noise_std
                          for emb in prob["step_embeddings"]]
            # Candidate is "correct" with probability decreasing with noise
            noise_score = np.mean([torch.norm(e).item() for e in noisy_embs])
            is_correct = noise_score < (len(prob["steps"]) * 1.5)
            candidates.append({"embeddings": noisy_embs, "correct": is_correct})

        if use_prm and prm is not None:
            prm.eval()
            best_idx = 0
            best_score = -1.0
            for i, cand in enumerate(candidates):
                score = score_solution_with_prm(prm, cand["embeddings"])
                if score > best_score:
                    best_score = score
                    best_idx = i
            if candidates[best_idx]["correct"]:
                correct += 1
        else:
            # ORM: pick randomly from correct candidates if any (simulates ORM reranking)
            correct_candidates = [c for c in candidates if c["correct"]]
            if correct_candidates:
                correct += 1

    return correct / len(val_problems)

# %% [markdown]
# ### Run Training and Evaluation

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs/

# %%
# PROVIDED TRAINING AND EVALUATION — do not modify.
# Uncomment to run after implementing ProcessRewardModel and score_solution_with_prm.
#
# print("Generating PRM training data...")
# train_data, val_problems = make_prm_dataset(n_train=200, n_val=50)
# print(f"Train examples: {len(train_data)}, Val problems: {len(val_problems)}")
#
# prm = ProcessRewardModel(input_dim=64, hidden_dim=256)
# print("Training PRM...")
# losses = train_prm(prm, train_data, n_epochs=10)
#
# # Evaluate
# orm_acc = evaluate_best_of_n(None, val_problems, n_candidates=16, use_prm=False)
# prm_acc = evaluate_best_of_n(prm, val_problems, n_candidates=16, use_prm=True)
#
# print(f"\nORM Best-of-16 accuracy: {orm_acc:.3f}")
# print(f"PRM Best-of-16 accuracy: {prm_acc:.3f}")
# print(f"Improvement: {(prm_acc - orm_acc)*100:.1f} percentage points")
#
# assert prm_acc - orm_acc >= 0.05, \
#     f"PRM did not improve by ≥5pp over ORM ({prm_acc:.3f} vs {orm_acc:.3f})"
# print("Verification PASSED: PRM Best-of-16 improves ≥5pp over ORM Best-of-16.")

# %% [markdown]
# ## Part 5: Ablation — Best-of-N Scaling Curves
#
# Run both PRM and ORM Best-of-N for $N \in \{1, 4, 16, 64\}$ and plot accuracy vs. N.
#
# **Expected result:** PRM should scale more efficiently — its accuracy curve should be
# above ORM's curve at every N value.
#
# *(Why does PRM scale better with N? Because it selects candidates based on reasoning quality,
# not just whether the final answer happens to be correct. This means each additional candidate
# is more informative.)*

# %%
def run_best_of_n_scaling(prm: ProcessRewardModel, val_problems: List[Dict],
                          n_values: List[int] = [1, 4, 16, 64]) -> Dict:
    """
    Run Best-of-N for each N value and return accuracy for PRM and ORM.

    PROVIDED — do not modify. Implement score_solution_with_prm above.
    """
    results = {"n_values": n_values, "prm_acc": [], "orm_acc": []}

    for n in n_values:
        orm_acc = evaluate_best_of_n(None, val_problems, n_candidates=n, use_prm=False)
        prm_acc = evaluate_best_of_n(prm, val_problems, n_candidates=n, use_prm=True)
        results["prm_acc"].append(prm_acc)
        results["orm_acc"].append(orm_acc)
        print(f"N={n:3d} | ORM: {orm_acc:.3f} | PRM: {prm_acc:.3f}")

    return results


def plot_scaling_curves(results: Dict) -> None:
    """Plot PRM vs ORM accuracy vs. N (log scale)."""
    plt.figure(figsize=(8, 5))
    plt.plot(results["n_values"], results["prm_acc"], "b-o", label="PRM Best-of-N")
    plt.plot(results["n_values"], results["orm_acc"], "r-s", label="ORM Best-of-N")
    plt.xscale("log")
    plt.xlabel("N (number of candidates)")
    plt.ylabel("Accuracy")
    plt.title("PRM vs ORM: Best-of-N Scaling")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(results["n_values"], results["n_values"])
    plt.tight_layout()
    plt.show()


# Uncomment to run after training:
# scaling_results = run_best_of_n_scaling(prm, val_problems)
# plot_scaling_curves(scaling_results)

# %% [markdown]
# ## Part 6: Reflection Questions
#
# **Q1: Why does process supervision outperform outcome supervision?**
# Consider: where do errors occur in multi-step reasoning? What signal does each provide?
#
# **Q2: What are the labeling costs of PRM?**
# Compare to ORM labeling cost. When is PRM labeling cost worth it?
#
# **Q3: How does PRM relate to the TD learning concepts from Module 2?**
# Hint: think about credit assignment and bootstrapping.

# %% [markdown]
# ### Your Answers
#
# **A1 (Process supervision advantage):**
# _Your answer here._
#
# **A2 (Labeling costs):**
# _Your answer here._
#
# **A3 (PRM and TD learning):**
# _Your answer here._
