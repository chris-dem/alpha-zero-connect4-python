from dataclasses import dataclass

@dataclass
class Arguments:
    in_dims: int
    out_dims: int 
    temperature_limit: float
    num_simulations: int = 1_000
    device: str = "cpu"
