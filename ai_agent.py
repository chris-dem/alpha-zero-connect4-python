from dataclasses import dataclass
import torch
import numpy as np

from mcts import MCTS
from piece import Turn
from board import Board


@dataclass
class Player:
    model: torch.Module
    mcts: MCTS
    player: Turn
    temperature: float
    train_logger: list[tuple[Turn, Board, list[float]]] = []

    def run(self, state, is_training=True):

        self.mcts.run(self.model, state, to_play=self.player)
        action_probs = [0 for _ in range(self.game.get_action_size())]
        for k, v in self.mcts.children.items():
            action_probs[k] = v.visit_count

        ## TODO add temperature scherduler
        action_probs = action_probs / np.sum(action_probs)
        return self.mcts.root.select_action(temperature)
