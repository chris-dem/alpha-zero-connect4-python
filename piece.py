"""
Piece module
"""

from enum import Enum
import pygame

from constants import GREY, MAX_SCORE, RED, ROWS, SQUARE_HEIGHT, PADDING, OUTLINE, SQUARE_WIDTH, WHITE_BASE 

class Turn(Enum):
    """
    Turn enum
    """

    YELLOW = True
    RED = False

class EndStatus(Enum):
    """
    End status
    """

    YELLOW = 0
    RED = 1
    DRAW = 2

def convert_status_to_score(status: EndStatus):
    match status:
        case EndStatus.YELLOW:
            return MAX_SCORE
        case EndStatus.RED:
            return -MAX_SCORE
        case _:
            return 0

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
    x = col * SQUARE_WIDTH + SQUARE_WIDTH // 2
    y = (ROWS - 1 - row) * SQUARE_HEIGHT + SQUARE_HEIGHT // 2
    return x, y


def draw_piece(screen, piece: Piece, row: int, col: int):
    """
    Draw piece
    """
    radius = min(SQUARE_WIDTH, SQUARE_HEIGHT) // 2 - PADDING
    pos = calculate_position(row, col)
    pygame.draw.circle(screen, GREY, pos, radius + OUTLINE)
    if piece != Piece.EMPTY:
        color = RED if piece == Piece.RED else WHITE_BASE
        pygame.draw.circle(
            screen, pygame.Color(color), pos, radius
        )
