from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    data_path: str = "plantvillage dataset"
    img_size: int = 256
    crop_size: int = 224
    batch_size: int = 32
    num_epochs: int = 10
    num_workers: int = 0
    seed: int = 42


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
