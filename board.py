"""
Board module
"""

from dataclasses import dataclass
from typing import Optional, cast, Self
from functools import reduce
from enum import Enum
import pygame
from math import floor, log2
from constants import BLACK, CHECK, COLS, RED, ROWS, SQUARE_SIZE, WHITE
from piece import Piece, Turn, draw_piece

class GameWinner(Enum):
    """
    Describe which player won
    """
    BLACK = 0
    WHITE = 1
    NOP = 2

@dataclass(frozen=True, eq=True)
class Board:
    """
    Board class
    """
    board_red: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)
    board_ylw: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)

    def draw(self, screen):
        """
        Draw board for pygame
        """
        self.draw_squares(screen)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.get_piece(row, col)
                draw_piece(screen, piece, row, col)

    def get_piece(self, row: int, col: int) -> Piece:
        """
        Given row and col return piece
        """
        assert 0 <= row < ROWS
        assert COLS > col >= 0
        ylw = self.board_ylw[row]
        red = self.board_red[row]
        if ylw & (1 << col):
            return Piece.YELLOW
        if red & (1 << col):
            return Piece.RED
        return Piece.EMPTY

    def draw_squares(self, screen: pygame.Surface):
        """
        Draw base board
        """
        screen.fill(BLACK)
        for row in range(ROWS):
            for col in range(COLS):
                color = WHITE
                pygame.draw.rect(
                    screen,
                    color,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                )
                pygame.draw.rect(
                    screen,
                    BLACK,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    2,
                )

    def is_winning(self) -> Optional[Turn]:
        """
        Return winner if there is one
        """
        compress_red = reduce(lambda acc,a: acc << 1 | a, self.board_red, 0)
        compress_ylw = reduce(lambda acc,a: acc << 1 | a, self.board_ylw, 0)
        if compress_red in CHECK:
            return Turn.RED
        if compress_ylw in CHECK:
            return Turn.YELLOW
        return None

    def is_move_legal(self, col: int) -> bool:
        return self.is_winning() is None and self.board_red[col] | self.board_ylw[col] != 127

    def get_height(self, col: int) -> int:
        col = self.board_red[col] | self.board_ylw[col]
        col = col ^ (col << 1)
        return floor(log2(col)) if col > 0 else 0

    def move(self, col: int, turn: Turn) -> Self:
        if not self.is_move_legal(col):
            return self
        bred = self.board_red[col]
        bylw = self.board_ylw[col]

        col = bred | bylw
        val = bred | (col << 1 ^ col)
        r = self.board_red
        y = self.board_ylw
        match turn:
            case Turn.RED:
                r = list(r)
                r[col] = val
                r = cast(tuple[int, int, int, int, int, int, int],tuple(r))
                return cast(Self, Board(board_red=r, board_ylw=y))
            case Turn.YELLOW:
                y = list(y)
                y[col] = val
                y = cast(tuple[int, int, int, int, int, int, int],tuple(y))
                return cast(Self, Board(board_red=r, board_ylw=y))
