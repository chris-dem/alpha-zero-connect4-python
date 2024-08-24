"""
Game module
"""

from dataclasses import dataclass, field
from typing import Optional


import board as b
from piece import Turn


@dataclass(eq=True, frozen=True)
class GameState:
    """
    Describes the state of the game, which is defined as the
    tuple of board and turn
    """
    turn: Turn = Turn.RED
    board: b.Board = field(default_factory=b.Board)

    def is_winning(self) -> Optional[Turn]:
        """
        Return winner if there is one
        """
        return self.board.is_winning()
