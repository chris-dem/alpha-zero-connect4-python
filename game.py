"""
Game module
"""

from dataclasses import dataclass, field
from typing import Optional
from functools import reduce
from typing import cast, Self

import torch
import board as b
from constants import CHECK, COLS, FULL_BOARD, ROWS, print_num
from piece import EndStatus, Turn


@dataclass(eq=True, frozen=True)
class GameState:
    """
    Describes the state of the game, which is defined as the
    tuple of board and turn
    """

    turn: Turn
    board: b.Board = field(default_factory=b.Board)

    def is_winning(self) -> Optional[EndStatus]:
        """
        Return winner if there is one
        """
        compress_red = reduce(
            lambda acc, a: (acc << ROWS) | a, self.board.board_red[::-1], 0
        )
        compress_ylw = reduce(
            lambda acc, a: (acc << ROWS) | a, self.board.board_ylw[::-1], 0
        )
        if compress_red | compress_ylw == FULL_BOARD:
            return EndStatus.DRAW
        red = any((i & compress_red) == i for i in CHECK)
        if red:
            return EndStatus.RED
        ylw = any((i & compress_ylw) == i for i in CHECK)
        if ylw:
            return EndStatus.YELLOW
        return None

    def print_debug(self) -> str:
        """
        Print board
        """
        s = ""
        for r in range(ROWS - 1, -1, -1):
            for c in range(COLS):
                v = self.board.get_piece(r, c)
                match v:
                    case b.Piece.EMPTY:
                        s += " "
                    case b.Piece.RED:
                        s += "R"
                    case b.Piece.YELLOW:
                        s += "Y"
            s += "\n"
        return s

    def is_move_legal(self, col: int) -> bool:
        """
        Given a column return whether a move is legal
        """
        return self.board.board_red[col] | self.board.board_ylw[col] < (2**ROWS - 1)

    def move(self, col: int) -> Self:
        assert self.is_move_legal(col)
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
        assert (
            ret.board_red[col] | ret.board_ylw[col]
            == ret.board_red[col] ^ ret.board_ylw[col]
        )
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
        cols = [None] * COLS
        for i in range(COLS):  # try 1 for us -1 for enemy 0 if nothing
            twos = convert_to_tensor(r[i])
            ones = convert_to_tensor(v[i])
            cols[i] = twos - ones
        ret = torch.stack(cast(list[torch.Tensor], cols), dim=1)[:, :, None]
        return ret


mask = 2 ** torch.arange(ROWS - 1, -1, -1)


def convert_to_tensor(x: int) -> torch.Tensor:
    """
    Convert integer to bitmask representation
    """
    return mask.bitwise_and(x).ne(0).long()
