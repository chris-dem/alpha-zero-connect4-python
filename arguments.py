"""
Simulation arguments for easier initialization
"""
from dataclasses import dataclass

from constants import COLS

@dataclass
class Arguments:
    in_dims: int = 1
    out_dims: int = COLS
    temperature_limit: int = 10
    num_simulations: int = 1_000 # MTCS simulations
    device: str = "mps"
    dtype: str = "float32"
    num_iters: int = 20 # Number of iterations
    num_eps: int = 50 # Number of games
    num_epochs: int = 20_000
    lr: float = 1e-4
    batch_size: int = 5000
    checkpoint_path = "brain_weights"
