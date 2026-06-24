# %% [markdown]
# # Module 06, Assignment 1: Training a Reward Model
#
# **Prerequisites:**
# - Basic PyTorch (nn.Module, LSTM)
# - Lecture notes sections 1–2
#
# **Learning Objectives:**
# 1. Understand the Bradley-Terry preference model
# 2. Train a reward model on comparison pairs
# 3. Evaluate ranking accuracy on held-out pairs
# 4. Understand reward hacking and how reward model quality affects RLHF

# %% [markdown]
# ## Part 1: Theory Recap
#
# The **Bradley-Terry model** defines the probability that response $y_w$ is preferred over
# $y_l$ given prompt $x$:
#
# $$P(y_w \succ y_l \mid x) = \sigma\!\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)$$
#
# We fit $R_\phi$ by maximizing the likelihood of the observed preferences:
#
# $$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(R_\phi(x,y_w) - R_\phi(x,y_l)\right)\right]$$
#
# **Why ranking accuracy, not MSE?**
# We care about *relative* ordering, not absolute reward values. Two reward functions that
# differ by a constant assign identical rankings. MSE would penalize constant offsets that
# don't affect policy quality at all.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Tuple, List

# %% [markdown]
# ## Part 2: Implement the Reward Model
#
# **Architecture:** `embedding → LSTM → linear head → scalar`
#
# The LSTM reads the token sequence and produces a hidden state. We take the final hidden
# state and project it to a single scalar reward. Positive reward = preferred response.
#
# **Why LSTM here?** We want to keep the model simple and trainable on CPU. In real RLHF
# pipelines, the backbone would be a transformer (same architecture as the policy), initialized
# from the SFT checkpoint.

# %%
class SentimentRewardModel(nn.Module):
    """LSTM-based reward model trained on IMDB preferences."""

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        # TODO: nn.Embedding + nn.LSTM + linear head → scalar output
        # Hint: use batch_first=True in the LSTM so input shape is (batch, seq, embed_dim)
        # The linear head maps hidden_dim → 1 and we squeeze to get shape (batch,)
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """Return scalar reward of shape (batch,).

        Args:
            input_ids: (batch, seq_len) token indices
            lengths: (batch,) actual sequence lengths before padding

        Hint: use pack_padded_sequence / pad_packed_sequence to handle variable-length
        sequences efficiently. Pass enforce_sorted=False so you don't need to sort.
        """
        raise NotImplementedError


# %% [markdown]
# ## Part 3: Implement the Loss and Evaluation
#
# ### Bradley-Terry Loss
#
# $$\mathcal{L} = -\text{mean}\!\left[\log\sigma(r_{\text{winner}} - r_{\text{loser}})\right]$$
#
# **Why F.logsigmoid instead of log(sigmoid(..))?**
# `F.logsigmoid` is numerically stable — it avoids computing `sigmoid` and then taking `log`,
# which can underflow for large negative inputs. Use `F.logsigmoid(r_winner - r_loser)` directly.

# %%
def bradley_terry_loss(r_winner: torch.Tensor, r_loser: torch.Tensor) -> torch.Tensor:
    """
    L = -mean(log(sigma(r_winner - r_loser)))
    r_winner, r_loser: scalar rewards for preferred/dispreferred responses, shape (batch,).

    Hint: torch.nn.functional.logsigmoid is numerically stable.
    """
    raise NotImplementedError


def evaluate_ranking_accuracy(model: nn.Module, eval_loader: DataLoader,
                               device: torch.device) -> float:
    """Fraction of pairs where model correctly ranks winner above loser.

    Args:
        model: SentimentRewardModel
        eval_loader: yields (winner_ids, winner_lengths, loser_ids, loser_lengths)
        device: torch device

    Returns:
        Accuracy in [0, 1]. Random chance = 0.5. Target: ≥ 0.80 after 5 epochs.
    """
    raise NotImplementedError


# %% [markdown]
# ## Part 4: Dataset Construction
#
# We construct preference pairs from IMDB movie reviews:
# - **Preferred (winner):** 5-star reviews (positive sentiment, rating = 1 in IMDB dataset)
# - **Dispreferred (loser):** 1-star reviews (negative sentiment, rating = 0 in IMDB dataset)
#
# *(Why IMDB for reward modeling?)* It has a natural ground-truth preference signal (star ratings)
# and is small enough to train on CPU. In real RLHF, preferences come from human annotators.

# %%
class IMDBPreferenceDataset(Dataset):
    """Pairs of (preferred, dispreferred) IMDB reviews.

    Positive reviews (label=1) are 'winners'; negative reviews (label=0) are 'losers'.
    We pair them randomly to create comparison examples.
    """

    def __init__(self, split: str = "train", max_pairs: int = 2000,
                 max_length: int = 128):
        dataset = load_dataset("imdb", split=split)

        # Separate by sentiment
        positives = [d["text"] for d in dataset if d["label"] == 1][:max_pairs]
        negatives = [d["text"] for d in dataset if d["label"] == 0][:max_pairs]

        # Build a simple character-level vocabulary from the data
        all_text = " ".join(positives[:500] + negatives[:500])
        chars = sorted(set(all_text))
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = PAD
        self.vocab_size = len(self.char2idx) + 1
        self.max_length = max_length

        n_pairs = min(len(positives), len(negatives), max_pairs)
        self.winners = positives[:n_pairs]
        self.losers = negatives[:n_pairs]

    def _encode(self, text: str) -> Tuple[torch.Tensor, int]:
        indices = [self.char2idx.get(c, 0) for c in text[:self.max_length]]
        length = len(indices)
        # Pad to max_length
        indices += [0] * (self.max_length - length)
        return torch.tensor(indices, dtype=torch.long), length

    def __len__(self) -> int:
        return len(self.winners)

    def __getitem__(self, idx: int):
        w_ids, w_len = self._encode(self.winners[idx])
        l_ids, l_len = self._encode(self.losers[idx])
        return w_ids, torch.tensor(w_len), l_ids, torch.tensor(l_len)


# %% [markdown]
# ## Part 5: Training Loop
#
# The training loop below is provided. Run it after implementing `SentimentRewardModel`,
# `bradley_terry_loss`, and `evaluate_ranking_accuracy`.

# %%
def train_reward_model(max_pairs: int = 2000, n_epochs: int = 5,
                       batch_size: int = 32, lr: float = 1e-3) -> SentimentRewardModel:
    """Train the reward model and log to TensorBoard."""
    from rllearn.logging import make_writer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Dataset ---
    print("Loading IMDB dataset...")
    train_dataset = IMDBPreferenceDataset(split="train", max_pairs=max_pairs)
    eval_dataset = IMDBPreferenceDataset(split="test", max_pairs=500)

    # Share vocabulary between train and eval
    eval_dataset.char2idx = train_dataset.char2idx
    eval_dataset.vocab_size = train_dataset.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = SentimentRewardModel(
        vocab_size=train_dataset.vocab_size,
        embed_dim=64,
        hidden_dim=128,
        n_layers=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = make_writer("reward_model_imdb")
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []

        for w_ids, w_len, l_ids, l_len in train_loader:
            w_ids, w_len = w_ids.to(device), w_len.to(device)
            l_ids, l_len = l_ids.to(device), l_len.to(device)

            r_winner = model(w_ids, w_len)
            r_loser = model(l_ids, l_len)

            loss = bradley_terry_loss(r_winner, r_loser)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # Eval
        acc = evaluate_ranking_accuracy(model, eval_loader, device)
        writer.add_scalar("eval/ranking_accuracy", acc, epoch)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}/{n_epochs} | loss={mean_loss:.4f} | ranking_acc={acc:.3f}")

    writer.close()
    return model


# %% [markdown]
# ### Run Training

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs/

# %%
# Uncomment to train:
# trained_model = train_reward_model(max_pairs=2000, n_epochs=5)
# torch.save(trained_model.state_dict(), "reward_model.pt")

# %% [markdown]
# **Verification:** After 5 epochs with 2000 pairs, ranking accuracy should be ≥ 0.80.
# If it's below 0.75, check:
# 1. Did you implement `pack_padded_sequence` in `forward`? (Without it, padding tokens pollute the LSTM state)
# 2. Is the loss decreasing each epoch?
# 3. Try reducing `lr` to `3e-4` if training is unstable

# %% [markdown]
# ## Part 6: Data Efficiency Ablation
#
# How many preference pairs do we need? This is critical in real RLHF where human annotations
# are expensive ($5–15 per comparison pair).

# %%
def run_data_efficiency_ablation():
    """Train reward models on different dataset sizes and compare ranking accuracy."""
    import matplotlib.pyplot as plt

    sizes = [100, 500, 2000]
    final_accuracies = []

    for max_pairs in sizes:
        print(f"\n--- Training with {max_pairs} pairs ---")
        model = train_reward_model(max_pairs=max_pairs, n_epochs=5)

        # Evaluate on held-out set
        eval_dataset = IMDBPreferenceDataset(split="test", max_pairs=500)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
        device = next(model.parameters()).device
        acc = evaluate_ranking_accuracy(model, eval_loader, device)
        final_accuracies.append(acc)
        print(f"Final ranking accuracy ({max_pairs} pairs): {acc:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, final_accuracies, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Number of Preference Pairs")
    plt.ylabel("Ranking Accuracy")
    plt.title("Reward Model Data Efficiency")
    plt.xscale("log")
    plt.axhline(0.80, color="red", linestyle="--", label="Target (0.80)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_model_data_efficiency.png", dpi=150)
    plt.show()

    return dict(zip(sizes, final_accuracies))


# Uncomment to run:
# results = run_data_efficiency_ablation()

# %% [markdown]
# **Expected pattern:** Accuracy improves quickly from 100→500 pairs, then plateaus. This
# is the "data efficiency" of the Bradley-Terry model — it's sample efficient because it
# only needs to learn relative ordering, not absolute values.

# %% [markdown]
# ## Part 7: Reflection Questions
#
# Answer these in a markdown cell below each question.
#
# **Q1. What happens if the reward model overfits?**
# > If $R_\phi$ memorizes the training set, it will assign high rewards to specific phrasings
# > rather than learning the underlying preference signal. When PPO optimizes against this RM,
# > it will find phrases that score high on the training distribution but not on actual human
# > preferences — this is **reward hacking**. In real RLHF, the RM is updated periodically
# > (iterative RLHF) to prevent this.
#
# **Q2. Why is ranking accuracy the right metric (not MSE)?**
# > The reward model is only used to *rank* responses — PPO maximizes $R_\phi$ but only the
# > relative ordering matters for which policy is better. Two RMs that differ by $R_\phi(x,y) +
# > C$ for any constant $C$ produce exactly the same policy. MSE would incorrectly penalize
# > constant offsets. Ranking accuracy directly measures what we care about: does the RM
# > correctly identify which response a human would prefer?
#
# **Q3 (Bonus). What is "reward model collapse" and how do you detect it?**
# > Write your answer here.
