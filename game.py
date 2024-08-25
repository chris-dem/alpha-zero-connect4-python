"""
Game module
"""

from dataclasses import dataclass, field
from typing import Optional
from functools import reduce
from typing import cast, Self

import torch
from torch.nn.functional import one_hot
import board as b
from constants import CHECK, COLS, FULL_BOARD, ROWS
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
        compress_red = reduce(
            lambda acc, a: (acc << 6) | a, self.board.board_red[::-1], 0
        )
        compress_ylw = reduce(
            lambda acc, a: (acc << 6) | a, self.board.board_ylw[::-1], 0
        )
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

    def canonical_representation(self) -> torch.Tensor:
        """
        Return canonical representation
        """
        r, v = None, None
        match self.turn:
            case Turn.RED:
                r, v = self.board.board_red, self.board.board_ylw
            case _:
                v, r = self.board.board_red, self.board.board_ylw
        cols = []
        for c2, c1 in zip(r, v):
            twos = convert_to_tensor(c2) * 2
            ones = convert_to_tensor(c1)
            cols.append(twos + ones)
        index_repr = torch.stack(cols, dim=1)
        # Current class = 2, enemy is 1, empty is 0
        return one_hot(tensor=index_repr, dim=3)


def convert_to_tensor(x: int) -> torch.Tensor:
    """
    Convert integer to bitmask representation
    """
    mask = 2 ** torch.arange(ROWS - 1, -1, -1)
    return mask.bitwise_and(x).ne(0).byte()
