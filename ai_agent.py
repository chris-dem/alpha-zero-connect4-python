from dataclasses import dataclass, field
import torch
import numpy as np

from mcts import MCTS
from temperature_scheduler import TemperatureScheduler, AlphazeroScheduler
from piece import Turn
from board import Board


@dataclass
class Player:
    """
    AI Agent class
    """

    player: Turn
    mcts: MCTS
    temperature: TemperatureScheduler = field(
        default_factory=lambda: AlphazeroScheduler(30)
    )
    train_logger: list[tuple[Turn, Board, list[float]]] = []

    def run(self, game, state, step, is_training=True):

        self.mcts.run(state, to_play=self.player)
        action_probs = [0 for _ in range(game.get_action_size())]
        # Might be better to consider the priors
        for k, v in self.mcts.children.items():
            action_probs[k] = v.visit_count

        if is_training:
            self.train_logger.append((self.player, state, action_probs))

        action_probs = action_probs / np.sum(action_probs)
        return self.mcts.root.select_action(temperature.temperature(step))


class TemperatureSchedule:
    def __init__(self, start_temp, end_temp, end_temp_decay):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.end_temp_decay = end_temp_decay
        self.current_temp = start_temp

    def step(self):
        self.current_temp = max(self.end_temp, self.current_temp * self.end_temp_decay)
