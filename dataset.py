"""
Dataset for the connect 4 AI
"""

import torch
from torch.utils.data import Dataset
from game import GameState


class GameDataSet(Dataset):
    """
    Dataset API
    """

    def __init__(self, examples: list[tuple[GameState, list[float], float]]):
        self.examples = [
            (
                a.canonical_representation().to(torch.float32).permute(2, 0, 1),
                torch.tensor(b).to(torch.float32),
                torch.tensor(c).to(torch.float32),
            )
            for a, b, c in examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, float]:
        board, vis, value = self.examples[idx]
        return board, vis, value
