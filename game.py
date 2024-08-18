"""
Game module
"""

from dataclasses import dataclass, field


import board as b
from constants import COLS, ROWS
from piece import Piece, Turn
from pipe import select, where


@dataclass(eq=True, frozen=True)
class GameState:
    """
    Describes the state of the game, which is defined as the
    tuple of board and turn
    """

    board: b.Board = field(default_factory=b.Board)
    turn: Turn = Turn.BLACK

def get_possible_moves(game_state: GameState) -> list[tuple[int, int]]:
    """
    Get all possible moves
    """
    arr = []
    for r in range(ROWS):
        p = list(range(COLS)
            | select(lambda c : game_state.board.get_piece(r, c))
            | where(lambda t : t is not None))
        for i,x in enumerate(p):
            if i == 0:
                if x.col != 0:
                    arr.append((x.row * COLS + x.col, x.row * COLS + x.col - 1))

    return arr
