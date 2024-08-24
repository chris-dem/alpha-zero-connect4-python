from dataclasses import dataclass, field
from typing import Optional, cast
import numpy as np

from constants import COLS
from game import GameState
from mcts import MCTS, Node
from temperature_scheduler import TemperatureScheduler, AlphazeroScheduler
from piece import Turn


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
    train_logger: list[tuple[Turn, GameState, list[float]]] = []

    def run(self,
            state: GameState,
            action: Optional[int] = None,
            step: int = 0, is_training=True) -> int:
        """
        Play move give current state and previous action
        """

        assert state.turn == self.player, "Wrong player"
        self.mcts.run(state, action)
        action_probs = [0] * COLS
        # Might be better to consider the priors
        root = cast(Node, self.mcts.root)
        for k, v in root.children.items():
            action_probs[k] = v.visit_count

        if is_training:
            self.train_logger.append((self.player, state, list(action_probs)))

        action_probs = action_probs / np.sum(action_probs)
        return self.mcts.root.select_action(temperature.temperature(step))
