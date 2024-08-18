"""
Piece module
"""

from enum import Enum 
import pygame
from constants import BLACK_PIECE, HEIGHT, KING_PIECE, RED, SELECTED, SQUARE_SIZE, GREY, WHITE_PIECE, WIDTH
class Turn(Enum):
    BLACK = True,
    WHITE = False,

class PieceType(Enum):
    BLACK = BLACK_PIECE
    WHITE = WHITE_PIECE
    KING = KING_PIECE


class Piece:
    """
    Class containing piece logic
    """

    PADDING = 15
    OUTLINE = 2

    def __init__(self, row, col, piece_type):
        self.row = row
        self.col = col
        self.piece_type = piece_type

        self.x = 0
        self.y = 0
        self.is_selected = False
        self.x, self.y = Piece.calc_pos(row, col)

    @staticmethod
    def calc_pos(r: int, c: int):
        """
        Calculate graphical positions
        """
        x = SQUARE_SIZE * c + SQUARE_SIZE // 2
        y = SQUARE_SIZE * r + SQUARE_SIZE // 2
        return x, y 

    def draw(self, screen):
        radius = SQUARE_SIZE // 2 - self.PADDING
        pygame.draw.circle(screen, GREY, (self.x, self.y), radius + self.OUTLINE)
        pygame.draw.circle(
            screen, pygame.Color(self.piece_type.value), (self.x, self.y), radius
        )

        if self.is_selected:
            circle = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            circle.set_alpha(100)
            pygame.draw.circle(
                circle, SELECTED, (radius, radius), radius
            )
            screen.blit(circle, (self.x - radius, self.y - radius))

    @staticmethod
    def compare_piece_turn(piece, turn) -> bool:
        match (Piece.eval(piece.piece_type), turn):
            case (0, Turn.BLACK) | (1, Turn.WHITE):
                return True
            case _:
                return False

    @staticmethod
    def eval(pt1: PieceType) -> int:
        match pt1:
            case (PieceType.WHITE | PieceType.KING):
                return 1
            case _:
                return 0

    def move(self, row, col):
        self.row = row
        self.col = col
        self.x, self.y = Piece.calc_pos(row, col)

    def __repr__(self) -> str:
        return f"Piece({self.piece_type}, {self.row}, {self.col})"
