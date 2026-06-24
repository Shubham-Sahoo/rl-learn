# %% [markdown]
# # Module 06, Assignment 3: GRPO — Group Relative Policy Optimization
#
# **Prerequisites:**
# - Module 03 A3: PPO-Clip loss (for comparison)
# - Lecture notes section 5
#
# **Learning Objectives:**
# 1. Implement group sampling and group-normalized advantages
# 2. Apply GRPO to a verifiable reasoning task
# 3. Compare GRPO vs PPO with value head on wall-clock efficiency
# 4. Understand why GRPO eliminates the value network

# %% [markdown]
# ## Part 1: Theory Recap
#
# ### The Value Network Problem in PPO
#
# PPO requires a value network $V_\psi(s_t)$ to compute advantages via GAE:
#
# $$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$
#
# For LLMs, $s_t$ is the full context at token $t$. Training a value network on this sequence
# requires backpropagating through the transformer for every token — expensive in memory and
# compute.
#
# ### GRPO's Solution
#
# Instead of learning $V(s)$, **estimate it empirically** using $G$ samples from the same prompt:
#
# $$\hat{V}(x) \approx \mu_r = \frac{1}{G} \sum_{i=1}^G r_i$$
#
# Normalize rewards within the group to get advantages:
#
# $$\tilde{A}_i = \frac{r_i - \mu_r}{\sigma_r + \varepsilon}$$
#
# **This is exactly variance reduction via baseline** — but the baseline is computed from
# the group rather than a learned value network.

# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import numpy as np

# %% [markdown]
# ## Part 2: Implement GRPO Components
#
# ### Group Sampling
#
# For each prompt, sample $G$ completions from the current policy. All $G$ completions
# see the *same* prompt — this is what "group relative" means: rewards are compared
# *within the group*, not across different prompts.

# %%
def grpo_group_sample(policy, tokenizer, prompt: str,
                      group_size: int = 8,
                      max_new_tokens: int = 64) -> Tuple[List[str], torch.Tensor]:
    """
    Sample group_size responses from policy for the same prompt.

    Args:
        policy: AutoModelForCausalLM (or compatible)
        tokenizer: matching tokenizer
        prompt: input prompt string
        group_size: number of responses to sample (G in the paper)
        max_new_tokens: maximum tokens to generate per response

    Returns:
        responses: list of decoded response strings, length group_size
        log_probs: tensor of shape (group_size,), per-sequence log-probability
                   log P(y_i | x) = sum_t log π(a_t | s_t) over generated tokens

    Hint: Encode the prompt, then call model.generate() with do_sample=True,
    num_return_sequences=group_size. Use output_scores=True and
    transition_scores to get log probabilities.

    Common mistake: log_probs should be summed over tokens, not averaged.
    The importance weight ρ = π_θ(y) / π_old(y) uses sequence log-probs.
    """
    raise NotImplementedError


def normalize_group_rewards(rewards: List[float]) -> torch.Tensor:
    """
    Normalize within group: A_i = (r_i - mean(r)) / (std(r) + 1e-8)
    Returns tensor of shape (group_size,).

    *(Why ε=1e-8?)* Common mistake: not adding epsilon to std — division by zero when
    all rewards are equal (e.g., all G responses are correct or all wrong). When σ_r = 0,
    Ã_i should be 0 for all i (we can't tell which response was better), which is
    exactly what you get with ε in the denominator.

    Args:
        rewards: list of scalar rewards, length group_size

    Returns:
        advantages: tensor of shape (group_size,), normalized to zero mean unit variance
    """
    raise NotImplementedError


def grpo_loss(log_probs_new: torch.Tensor, log_probs_old: torch.Tensor,
              advantages: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    """
    PPO-Clip loss using GRPO normalized advantages (no value network needed).
    Identical formula to PPO-Clip but advantages are group-normalized rewards.

    Args:
        log_probs_new: (group_size,) log P(y_i | x) under current policy π_θ
        log_probs_old: (group_size,) log P(y_i | x) under old policy π_θ_old (no grad)
        advantages: (group_size,) group-normalized advantages from normalize_group_rewards
        clip_eps: PPO clipping epsilon (default 0.2)

    Returns:
        loss: scalar loss (negative of clipped objective, for gradient descent)

    Steps:
    1. Compute importance ratios: ρ_i = exp(log_probs_new - log_probs_old)
    2. Compute clipped surrogate: min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i)
    3. Return negative mean (we minimize, but want to maximize expected return)

    *(Why clip?)* Without clipping, a single high-advantage sample could produce a huge
    gradient step, destabilizing training. Clipping limits how much any one sample
    can change the policy in a single update.
    """
    raise NotImplementedError


# %% [markdown]
# ## Part 3: Verifiable Reasoning Task
#
# We apply GRPO to a math problem: "What is {a} + {b}?"
#
# **Why math?** Math has a ground-truth verifier — the oracle is perfect (unlike a learned RM).
# This eliminates reward hacking. DeepSeek-R1 used the same idea at scale.
#
# **Reward signal:** +1 if the model's response contains the correct answer, 0 otherwise.

# %%
def math_oracle_reward(response: str, a: int, b: int) -> float:
    """Return 1.0 if response contains the correct sum, 0.0 otherwise."""
    correct = str(a + b)
    # Extract numbers from response and check if any match
    import re
    numbers = re.findall(r'-?\d+', response)
    return 1.0 if correct in numbers else 0.0


def generate_math_prompts(n_prompts: int = 100,
                          max_val: int = 50) -> List[Tuple[str, int, int]]:
    """Generate (prompt, a, b) tuples for addition problems."""
    rng = np.random.default_rng(42)
    prompts = []
    for _ in range(n_prompts):
        a = int(rng.integers(0, max_val))
        b = int(rng.integers(0, max_val))
        prompt = f"What is {a} + {b}? Answer with just the number."
        prompts.append((prompt, a, b))
    return prompts


# %% [markdown]
# ## Part 4: GRPO Training Loop
#
# The training loop is provided. It runs GRPO on the math addition task using GPT-2 Small.
# GPT-2 is a poor math solver initially — the goal is to show improvement, not perfection.

# %%
def compute_solve_rate(policy, tokenizer, prompts: List[Tuple[str, int, int]],
                       device: torch.device) -> float:
    """Compute fraction of prompts where the policy gives the correct answer."""
    policy.eval()
    correct = 0
    with torch.no_grad():
        for prompt, a, b in prompts[:50]:  # Evaluate on first 50 for speed
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = policy.generate(
                **inputs, max_new_tokens=16, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            correct += math_oracle_reward(response, a, b)
    policy.train()
    return correct / min(50, len(prompts))


def run_grpo_training(model_name: str = "gpt2",
                      group_size: int = 8,
                      n_steps: int = 500,
                      lr: float = 5e-6) -> nn.Module:
    """
    Train GPT-2 Small on math addition using GRPO.

    Logs to TensorBoard:
    - train/solve_rate: fraction of prompts answered correctly (target: +10pp from baseline)
    - train/mean_reward: mean reward within each group across all prompts
    - train/reward_std: standard deviation of rewards (higher = more learning signal)

    Args:
        model_name: HuggingFace model ID
        group_size: number of responses sampled per prompt (G)
        n_steps: total GRPO update steps
        lr: learning rate for AdamW

    Returns:
        Trained policy model
    """
    from rllearn.logging import make_writer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | group_size={group_size} | n_steps={n_steps}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    policy_old = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    policy_old.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    writer = make_writer(f"grpo_math_g{group_size}")

    # Generate training prompts
    all_prompts = generate_math_prompts(n_prompts=200)
    eval_prompts = generate_math_prompts(n_prompts=100)

    # Baseline solve rate
    baseline_rate = compute_solve_rate(policy, tokenizer, eval_prompts, device)
    print(f"Baseline solve rate: {baseline_rate:.3f}")
    writer.add_scalar("train/solve_rate", baseline_rate, 0)

    rng = np.random.default_rng(0)

    for step in range(1, n_steps + 1):
        # Sample a random prompt for this step
        prompt, a, b = all_prompts[rng.integers(len(all_prompts))]

        # --- Group sampling from current policy ---
        responses, log_probs_old = grpo_group_sample(
            policy_old, tokenizer, prompt,
            group_size=group_size, max_new_tokens=16
        )
        log_probs_old = log_probs_old.to(device).detach()

        # --- Score with oracle ---
        rewards = [math_oracle_reward(r, a, b) for r in responses]

        # --- Normalize within group ---
        advantages = normalize_group_rewards(rewards).to(device)

        # --- Compute log probs under current policy ---
        inputs = tokenizer(
            [prompt + r for r in responses],
            return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)

        outputs = policy(**inputs)
        logits = outputs.logits  # (G, seq_len, vocab)

        # Compute per-sequence log-probs for current policy
        # (simplified: use mean log-prob of generated tokens)
        log_probs_new = torch.zeros(group_size, device=device)
        for i, resp in enumerate(responses):
            resp_ids = tokenizer(resp, return_tensors="pt").input_ids[0].to(device)
            if len(resp_ids) == 0:
                continue
            prompt_len = inputs["input_ids"].shape[1] - len(resp_ids)
            if prompt_len < 0:
                prompt_len = 0
            token_logits = logits[i, prompt_len:prompt_len + len(resp_ids), :]
            token_log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
            per_token_lp = token_log_probs.gather(1, resp_ids.unsqueeze(1)).squeeze(1)
            log_probs_new[i] = per_token_lp.sum()

        # --- GRPO loss ---
        loss = grpo_loss(log_probs_new, log_probs_old, advantages)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Update old policy periodically
        if step % 10 == 0:
            policy_old.load_state_dict(policy.state_dict())

        # Logging
        mean_reward = float(np.mean(rewards))
        reward_std = float(np.std(rewards))
        writer.add_scalar("train/mean_reward", mean_reward, step)
        writer.add_scalar("train/reward_std", reward_std, step)
        writer.add_scalar("train/loss", loss.item(), step)

        if step % 50 == 0:
            solve_rate = compute_solve_rate(policy, tokenizer, eval_prompts, device)
            writer.add_scalar("train/solve_rate", solve_rate, step)
            print(f"Step {step:4d} | loss={loss.item():.4f} | "
                  f"mean_reward={mean_reward:.3f} | solve_rate={solve_rate:.3f}")

    writer.close()
    print(f"\nFinal solve rate should be ≥ {baseline_rate + 0.10:.3f} "
          f"(baseline {baseline_rate:.3f} + 10pp target)")
    return policy


# %% [markdown]
# ### Run Training

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs/

# %%
# Uncomment to train (~15 min on CPU for 500 steps with group_size=8):
# trained_policy = run_grpo_training(group_size=8, n_steps=500)

# %% [markdown]
# **Verification:** After 500 GRPO steps with group_size=8:
# - `train/solve_rate` should increase by ≥ 10 percentage points from baseline
# - `train/reward_std` should be > 0 (if it's 0, all responses are identical — check sampling)
# - Loss should generally decrease, though noisy

# %% [markdown]
# ## Part 5: Group Size Ablation
#
# Group size $G$ controls the quality of the Monte Carlo value estimate $\hat{V}(x) = \mu_r$.
# Larger $G$ = better estimate but more inference cost.

# %%
def run_group_size_ablation(group_sizes: List[int] = [2, 4, 8, 16],
                            n_steps: int = 200):
    """Compare GRPO convergence speed across different group sizes.

    Expected findings:
    - G=2: high variance advantages, slow/unstable convergence
    - G=4: reasonable baseline estimate, decent convergence
    - G=8: good balance (DeepSeek-R1 uses G=8 in math tasks)
    - G=16: slower per-step (more inference), but lower variance per update

    Wall-clock efficiency: G=8 typically wins — good variance reduction with reasonable cost.
    """
    import matplotlib.pyplot as plt

    results = {}
    eval_prompts = generate_math_prompts(n_prompts=100)

    for G in group_sizes:
        print(f"\n--- Group Size G={G} ---")
        from rllearn.logging import make_writer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        policy = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        policy_old = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        policy_old.requires_grad_(False)

        optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-6)
        writer = make_writer(f"grpo_g{G}_ablation")
        all_prompts = generate_math_prompts(n_prompts=200)
        rng = np.random.default_rng(42)

        solve_rates = []

        for step in range(1, n_steps + 1):
            prompt, a, b = all_prompts[rng.integers(len(all_prompts))]
            responses, log_probs_old = grpo_group_sample(
                policy_old, tokenizer, prompt, group_size=G, max_new_tokens=16
            )
            log_probs_old = log_probs_old.to(device).detach()
            rewards = [math_oracle_reward(r, a, b) for r in responses]
            advantages = normalize_group_rewards(rewards).to(device)

            inputs = tokenizer(
                [prompt + r for r in responses],
                return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            outputs = policy(**inputs)
            logits = outputs.logits

            log_probs_new = torch.zeros(G, device=device)
            for i, resp in enumerate(responses):
                resp_ids = tokenizer(resp, return_tensors="pt").input_ids[0].to(device)
                if len(resp_ids) == 0:
                    continue
                prompt_len = inputs["input_ids"].shape[1] - len(resp_ids)
                token_logits = logits[i, max(0, prompt_len):prompt_len + len(resp_ids), :]
                token_log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                per_token_lp = token_log_probs.gather(1, resp_ids.unsqueeze(1)).squeeze(1)
                log_probs_new[i] = per_token_lp.sum()

            loss = grpo_loss(log_probs_new, log_probs_old, advantages)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                policy_old.load_state_dict(policy.state_dict())
            if step % 25 == 0:
                sr = compute_solve_rate(policy, tokenizer, eval_prompts, device)
                solve_rates.append((step, sr))
                writer.add_scalar("train/solve_rate", sr, step)

        writer.close()
        results[G] = solve_rates
        print(f"G={G} final solve rate: {solve_rates[-1][1]:.3f}")

    # Plot
    plt.figure(figsize=(10, 6))
    for G, data in results.items():
        steps, rates = zip(*data) if data else ([], [])
        plt.plot(steps, rates, marker="o", label=f"G={G}", linewidth=2)
    plt.xlabel("GRPO Steps")
    plt.ylabel("Solve Rate")
    plt.title("GRPO: Group Size vs Convergence Speed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grpo_group_size_ablation.png", dpi=150)
    plt.show()
    return results


# Uncomment to run:
# ablation = run_group_size_ablation(group_sizes=[2, 4, 8, 16], n_steps=200)

# %% [markdown]
# ## Part 6: Reflection Questions
#
# **Q1. Why does GRPO not need a value network?**
# > GRPO uses group sampling to compute a Monte Carlo estimate of the value function:
# > $\hat{V}(x) = \frac{1}{G}\sum_i r_i$. This estimate is unbiased (no learned approximation
# > error) but has variance that decreases as $O(1/G)$. A learned value network also estimates
# > $V(x)$ but can be more sample-efficient when the state space is large and the reward signal
# > is dense. For verifiable tasks with binary rewards (+1/0), the Monte Carlo estimate works
# > well because the problem is simple enough that the group average is a reliable baseline.
# > GRPO trades off inference cost ($G$ forward passes per prompt) for training stability
# > (no value head, no critic loss, no coefficient tuning).
#
# **Q2. When would GRPO fail vs PPO with value head?**
# > GRPO fails when:
# > 1. **Sparse rewards across groups:** If most prompts have all $G$ responses either correct
# >    or all wrong ($\sigma_r \approx 0$), advantages collapse to zero and there's no
# >    learning signal. PPO with a value head can still learn from the TD error even without
# >    reward variance within a batch.
# > 2. **Dense process rewards:** When intermediate token-level rewards are available (process
# >    reward models), PPO with GAE can use them. GRPO only works with outcome-level rewards.
# > 3. **Small group sizes:** With $G=2$, the baseline estimate is very noisy. PPO's value
# >    network provides a lower-variance baseline at the cost of additional computation.
# > 4. **Long episodes:** The Monte Carlo estimate ignores per-step credit. PPO with GAE
# >    assigns credit to specific tokens; GRPO assigns the same normalized reward to the
# >    full sequence.
#
# **Q3 (Bonus). The GRPO advantage is normalized to zero mean. What does this imply about the policy gradient?**
# > Write your answer here.
