"""
Game module
"""

from dataclasses import dataclass, field
from typing import Optional
from functools import reduce
from typing import cast, Self

import board as b
from constants import CHECK, COLS, FULL_BOARD
from piece import EndStatus, Turn


@dataclass(eq=True, frozen=True)
class GameState:
    """
    Describes the state of the game, which is defined as the
    tuple of board and turn
    """
    turn: Turn = Turn.RED
    board: b.Board = field(default_factory=b.Board)

    def is_winning(self) -> Optional[EndStatus]:
        """
        Return winner if there is one
        """
        compress_red = reduce(lambda acc, a: (acc << 6) | a, self.board.board_red[::-1], 0)
        compress_ylw = reduce(lambda acc, a: (acc << 6) | a, self.board.board_ylw[::-1], 0)
        if compress_red | compress_ylw == FULL_BOARD:
            return EndStatus.DRAW
        red = any((i & compress_red) == i for i in CHECK)
        if red:
            return EndStatus.RED
        ylw = any((i & compress_ylw) == i for i in CHECK)
        if ylw:
            return EndStatus.YELLOW
        if compress_red | compress_ylw == FULL_BOARD:
            return EndStatus.DRAW
        return None

    def is_move_legal(self, col: int) -> bool:
        """
        Given a column return whether a move is legal
        """
        return (
            self.is_winning() is None
            and self.board.board_red[col] | self.board.board_ylw[col] != 127
        )

    def move(self, col: int) -> Self:
        if not self.is_move_legal(col):
            return self
        bred = self.board.board_red[col]
        bylw = self.board.board_ylw[col]

        col_val = bred | bylw
        val = ((col_val << 1) | 1) ^ col_val
        r = self.board.board_red
        y = self.board.board_ylw
        ret = None
        match self.turn:
            case Turn.RED:
                r = list(r)
                r[col] = r[col] | val
                r = cast(tuple[int, int, int, int, int, int, int], tuple(r))
                ret = b.Board(board_red=r, board_ylw=y)
            case _:
                y = list(y)
                y[col] = val | y[col]
                y = cast(tuple[int, int, int, int, int, int, int], tuple(y))
                ret = b.Board(board_red=r, board_ylw=y)
        return cast(Self, GameState(Turn(not self.turn.value), ret))

    def get_valid_moves(self) -> list[bool]:
        """
        Return list of valid moves
        """
        return [self.is_move_legal(i) for i in range(COLS)]
