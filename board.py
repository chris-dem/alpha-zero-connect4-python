"""
Board module
"""

from dataclasses import dataclass
from math import floor, log2
import pygame
from constants import BLACK, COLS, ROWS, SQUARE_HEIGHT,SQUARE_WIDTH, WHITE
from piece import Piece, draw_piece


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
        ylw = self.board_ylw[col]
        red = self.board_red[col]
        if ylw & (1 << row):
            return Piece.YELLOW
        if red & (1 << row):
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
                    (col * SQUARE_WIDTH, row * SQUARE_HEIGHT, SQUARE_WIDTH, SQUARE_HEIGHT),
                )
                pygame.draw.rect(
                    screen,
                    BLACK,
                    (col * SQUARE_WIDTH, row * SQUARE_HEIGHT, SQUARE_WIDTH, SQUARE_HEIGHT),
                    2,
                )

    def get_height(self, col: int) -> int:
        col = self.board_red[col] | self.board_ylw[col]
        col = col ^ (col << 1)
        return floor(log2(col)) if col > 0 else 0
