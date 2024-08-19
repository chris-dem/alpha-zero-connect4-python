"""
Piece module
"""

from enum import Enum
import pygame

from constants import GREY, RED, ROWS, SQUARE_SIZE, PADDING, OUTLINE, WHITE_BASE 


class Turn(Enum):
    """
    Turn enum
    """

    YELLOW = True
    RED = False


class Piece(Enum):
    """
    Piece enum
    """

    YELLOW = 0
    RED = 1
    EMPTY = 2

def calculate_position(row, col):
    """
    Convert row column to screen coords
    """
    x = col * SQUARE_SIZE + SQUARE_SIZE // 2
    y = (ROWS - 1 - row) * SQUARE_SIZE + SQUARE_SIZE // 2
    return x, y


def draw_piece(screen, piece: Piece, row: int, col: int):
    """
    Draw piece
    """
    radius = SQUARE_SIZE // 2 - PADDING
    pos = calculate_position(row, col)
    pygame.draw.circle(screen, GREY, pos, radius + OUTLINE)
    if piece != Piece.EMPTY:
        color = RED if piece == Piece.RED else WHITE_BASE
        pygame.draw.circle(
            screen, pygame.Color(color), pos, radius
        )
