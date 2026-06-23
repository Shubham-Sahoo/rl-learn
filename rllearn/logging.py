from __future__ import annotations
import time
from torch.utils.tensorboard import SummaryWriter


def make_writer(run_name: str, log_dir: str = "runs") -> SummaryWriter:
    """Create a TensorBoard SummaryWriter with timestamped run directory.

    Usage in notebooks:
        %load_ext tensorboard
        %tensorboard --logdir runs/
        writer = make_writer("dqn_cartpole")
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    full_path = f"{log_dir}/{run_name}_{timestamp}"
    return SummaryWriter(log_dir=full_path)
