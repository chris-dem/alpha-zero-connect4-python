"""
Simulation arguments for easier initialization
"""
from dataclasses import dataclass

from constants import COLS

@dataclass
class Arguments:
    in_dims: int = 3
    out_dims: int = COLS
    temperature_limit: int = 15
    num_simulations: int = 100
    device: str = "cpu"
    dtype: str = "float32"
    num_iters: int = 1_000
    num_eps: int = 10
    num_epochs: int = 100
    lr: float = 1e-4
    batch_size: int = 128
    checkpoint_path = "brain_weights"
