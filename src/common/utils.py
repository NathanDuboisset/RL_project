import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch. Environments are seeded separately, through `env.reset(seed=...)`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
