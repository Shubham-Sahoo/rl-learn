# %% [markdown]
# # Module 06, Assignment 1: Training a Reward Model

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

# %%
class SentimentRewardModel(nn.Module):
    """LSTM-based reward model trained on IMDB preferences."""

    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """Return scalar reward of shape (batch,)."""
        embedded = self.embedding(input_ids)  # (batch, seq, embed)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        # Take the last layer's hidden state
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        reward = self.head(last_hidden).squeeze(-1)
        return reward


# %% [markdown]
# ## Part 3: Implement the Loss and Evaluation

# %%
def bradley_terry_loss(r_winner: torch.Tensor, r_loser: torch.Tensor) -> torch.Tensor:
    """L = -mean(log(sigma(r_winner - r_loser)))"""
    return -F.logsigmoid(r_winner - r_loser).mean()


def evaluate_ranking_accuracy(model: nn.Module, eval_loader: DataLoader,
                               device: torch.device) -> float:
    """Fraction of pairs where model correctly ranks winner above loser."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for w_ids, w_len, l_ids, l_len in eval_loader:
            w_ids, w_len = w_ids.to(device), w_len.to(device)
            l_ids, l_len = l_ids.to(device), l_len.to(device)
            r_w = model(w_ids, w_len)
            r_l = model(l_ids, l_len)
            correct += (r_w > r_l).sum().item()
            total += len(r_w)
    model.train()
    return correct / total if total > 0 else 0.0


# %% [markdown]
# ## Part 4: Dataset Construction

# %%
class IMDBPreferenceDataset(Dataset):
    """Pairs of (preferred, dispreferred) IMDB reviews."""

    def __init__(self, split: str = "train", max_pairs: int = 2000,
                 max_length: int = 128):
        dataset = load_dataset("imdb", split=split)

        positives = [d["text"] for d in dataset if d["label"] == 1][:max_pairs]
        negatives = [d["text"] for d in dataset if d["label"] == 0][:max_pairs]

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

# %%
def train_reward_model(max_pairs: int = 2000, n_epochs: int = 5,
                       batch_size: int = 32, lr: float = 1e-3) -> SentimentRewardModel:
    """Train the reward model and log to TensorBoard."""
    from rllearn.logging import make_writer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    print("Loading IMDB dataset...")
    train_dataset = IMDBPreferenceDataset(split="train", max_pairs=max_pairs)
    eval_dataset = IMDBPreferenceDataset(split="test", max_pairs=500)
    eval_dataset.char2idx = train_dataset.char2idx
    eval_dataset.vocab_size = train_dataset.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

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

        acc = evaluate_ranking_accuracy(model, eval_loader, device)
        writer.add_scalar("eval/ranking_accuracy", acc, epoch)
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}/{n_epochs} | loss={mean_loss:.4f} | ranking_acc={acc:.3f}")

    writer.close()
    return model


# %%
# Uncomment to train:
# trained_model = train_reward_model(max_pairs=2000, n_epochs=5)
# torch.save(trained_model.state_dict(), "reward_model.pt")

# %% [markdown]
# **Verification:** After 5 epochs with 2000 pairs, ranking accuracy should be >= 0.80.

# %% [markdown]
# ## Part 7: Reflection Questions

# %% [markdown]
# **Q1:** If the reward model overfits, PPO will find high-scoring phrases that don't generalize
# to real human preferences — reward hacking. Iterative RLHF updates the RM periodically.

# %% [markdown]
# **Q2:** Ranking accuracy measures the right thing for RL: we care about relative ordering,
# not absolute values. Two RMs that differ by a constant produce identical policies.

# %% [markdown]
# **Q3:** With only 100 pairs, the RM may not generalize well. The "data wall" in RLHF is
# that human annotation is expensive ($5-15 per pair), so data efficiency matters enormously.
