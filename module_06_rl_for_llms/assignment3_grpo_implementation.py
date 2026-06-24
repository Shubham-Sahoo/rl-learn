# %% [markdown]
# # Module 06, Assignment 3: GRPO — Group Relative Policy Optimization

# %%
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import numpy as np

# %% [markdown]
# ## Part 2: Implement GRPO Components

# %%
def grpo_group_sample(policy, tokenizer, prompt: str,
                      group_size: int = 8,
                      max_new_tokens: int = 64) -> Tuple[List[str], torch.Tensor]:
    """Sample group_size responses from policy for the same prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(policy.parameters()).device)

    with torch.no_grad():
        outputs = policy.generate(
            input_ids.repeat(group_size, 1),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    sequences = outputs.sequences
    prompt_len = input_ids.shape[1]
    gen_sequences = sequences[:, prompt_len:]

    # Compute log probs of generated tokens
    scores = outputs.scores  # tuple of (group_size, vocab) tensors
    log_probs = torch.zeros(group_size, device=sequences.device)
    for t, score_t in enumerate(scores):
        log_prob_t = torch.log_softmax(score_t, dim=-1)
        # Get log prob of chosen token
        tok_ids = gen_sequences[:, t]
        log_probs += log_prob_t.gather(1, tok_ids.unsqueeze(1)).squeeze(1)

    responses = [
        tokenizer.decode(gen_sequences[i], skip_special_tokens=True)
        for i in range(group_size)
    ]

    return responses, log_probs


def normalize_group_rewards(rewards: List[float]) -> torch.Tensor:
    """Normalize within group: A_i = (r_i - mean(r)) / (std(r) + 1e-8)"""
    r = torch.tensor(rewards, dtype=torch.float32)
    mean_r = r.mean()
    std_r = r.std()
    advantages = (r - mean_r) / (std_r + 1e-8)
    return advantages


def grpo_loss(log_probs_new: torch.Tensor, log_probs_old: torch.Tensor,
              advantages: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    """PPO-Clip loss using GRPO normalized advantages."""
    ratio = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.mean(torch.min(surr1, surr2))
    return loss


# %% [markdown]
# ## Part 3: Verifiable Reasoning Task

# %%
def math_oracle_reward(response: str, a: int, b: int) -> float:
    """Return 1.0 if response contains the correct sum, 0.0 otherwise."""
    correct = str(a + b)
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

# %%
def compute_solve_rate(policy, tokenizer, prompts: List[Tuple[str, int, int]],
                       device: torch.device) -> float:
    """Compute fraction of prompts where the policy gives the correct answer."""
    policy.eval()
    correct = 0
    with torch.no_grad():
        for prompt, a, b in prompts[:50]:
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
    """Train GPT-2 Small on math addition using GRPO."""
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

    all_prompts = generate_math_prompts(n_prompts=200)
    eval_prompts = generate_math_prompts(n_prompts=100)

    baseline_sr = compute_solve_rate(policy, tokenizer, eval_prompts, device)
    print(f"Baseline solve rate: {baseline_sr:.2%}")

    for step in range(n_steps):
        prompt_idx = step % len(all_prompts)
        prompt, a, b = all_prompts[prompt_idx]

        # Sample group from old policy
        responses, log_probs_old = grpo_group_sample(
            policy_old, tokenizer, prompt, group_size=group_size)

        # Compute rewards
        rewards = [math_oracle_reward(r, a, b) for r in responses]
        advantages = normalize_group_rewards(rewards).to(device)

        # Recompute log probs under current policy
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        log_probs_new_list = []
        for response in responses:
            resp_ids = tokenizer(response, return_tensors="pt").input_ids.to(device)
            full_ids = torch.cat([input_ids, resp_ids], dim=1)
            with torch.enable_grad():
                out = policy(full_ids, labels=full_ids)
                # Approximate sequence log prob as negative loss * seq len
                log_prob_new = -out.loss * resp_ids.shape[1]
            log_probs_new_list.append(log_prob_new)

        log_probs_new = torch.stack(log_probs_new_list)
        log_probs_old_d = log_probs_old.to(device).detach()

        loss = grpo_loss(log_probs_new, log_probs_old_d, advantages)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Sync old policy periodically
        if step % 10 == 0:
            policy_old.load_state_dict(policy.state_dict())

        writer.add_scalar("train/mean_reward", np.mean(rewards), step)
        writer.add_scalar("train/reward_std", np.std(rewards), step)

        if step % 50 == 0:
            sr = compute_solve_rate(policy, tokenizer, eval_prompts, device)
            writer.add_scalar("train/solve_rate", sr, step)
            print(f"Step {step:4d} | Mean reward: {np.mean(rewards):.2f} | Solve rate: {sr:.2%}")

    writer.close()
    return policy


# %% [markdown]
# ## Part 5: Reflection

# %% [markdown]
# **Answers:**
# 1. GRPO eliminates the value network by using group statistics as a baseline. This saves
#    memory and compute — no separate critic network with its own backward pass.
# 2. Larger group size gives a better estimate of the baseline (lower variance), but requires
#    more forward passes per prompt. Optimal G balances compute cost vs. variance reduction.
# 3. Verifiable rewards (math, code execution) eliminate reward hacking since the oracle is
#    perfect. Non-verifiable rewards (open-ended text quality) still require a learned RM.
