"""
Board module
"""

from collections.abc import Callable
from typing import Optional
from functools import reduce
from itertools import chain
from enum import Enum
import bitarray
import pygame
from piece import Piece, PieceType
from constants import (
    BLACK_BASE,
    BLACK_BASE_SQUARES,
    EXIT_BASE,
    EXIT_BASE_SQUARES,
    RED,
    ROWS,
    COLS,
    SQUARE_SIZE,
    WHITE,
    BLACK,
    WHITE_BASE,
    WHITE_BASE_SQUARES,
)

class GameWinner(Enum):
    """
    Describe which player won
    """
    BLACK = 0
    WHITE = 1
    NOP = 2

class Board:
    """
    Board class
    """

    def __init__(self):
        self.board: list[Optional[Piece]] = [None] * (ROWS * COLS)
        self.create_board()

    def conv_coord(self, r: int, c: int) -> int:
        return r * COLS + c

    def create_board(self):
        for r, c in WHITE_BASE_SQUARES:
            self.board[self.conv_coord(r, c)] = Piece(r, c, PieceType.WHITE)

        self.board[self.conv_coord(ROWS // 2, COLS // 2)] = Piece(
            ROWS // 2, COLS // 2, PieceType.KING
        )

        for r, c in BLACK_BASE_SQUARES:
            self.board[self.conv_coord(r, c)] = Piece(r, c, PieceType.BLACK)

    def draw(self, screen):
        self.draw_squares(screen)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[self.conv_coord(row, col)]
                if piece is not None:
                    piece.draw(screen)

    def draw_squares(self, screen: pygame.Surface):
        screen.fill(BLACK)
        for row in range(ROWS):
            for col in range(COLS):
                ind = (row, col)
                color = WHITE
                if ind in WHITE_BASE_SQUARES:
                    color = WHITE_BASE
                elif ind in EXIT_BASE_SQUARES:
                    color = EXIT_BASE
                elif ind in BLACK_BASE_SQUARES:
                    color = BLACK_BASE
                pygame.draw.rect(
                    screen,
                    color,
                    (row * SQUARE_SIZE, col * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                )
                pygame.draw.rect(
                    screen,
                    BLACK,
                    (row * SQUARE_SIZE, col * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    2,
                )

    def get_piece(self, row, col) -> Optional[Piece]:
        if row < 0 or row >= ROWS or col < 0 or col >= COLS:
            return None
        return self.board[self.conv_coord(row, col)]

    def move(self, piece, row, col):
        (
            self.board[self.conv_coord(piece.row, piece.col)],
            self.board[self.conv_coord(row, col)],
        ) = (
            self.board[self.conv_coord(row, col)],
            self.board[self.conv_coord(piece.row, piece.col)],
        )
        piece.move(row, col)
        if piece.piece_type == PieceType.KING and (row, col) in EXIT_BASE_SQUARES:
            return GameState.White
        if piece.piece_type == PieceType.BLACK:
            arr = (self.get_piece(row + r,col + c) for (r,c) in [(-1,0),(0, -1), (1, 0), (0, 1)])
            arr = list(a for a in arr if a is not None and a.piece_type == PieceType.KING)
            if len(arr) > 0:
                vals = (
                        self.get_piece(arr[0].row + r,arr[0].col + c)
                        for (r,c) in [(-1,0),(0, -1), (1, 0), (0, 1)])
                if all(a is not None and a.piece_type == PieceType.BLACK for a in vals):
                    return GameState.Black
                return GameState.Black

        for (n, nn) in map(
                lambda s: (self.get_piece(s[0][0] + row, s[0][1] + col), 
                           self.get_piece(s[1][0] + row, s[1][1] + col)), [
                ((0,1), (0,2)),
                ((0,-1),(0,-2)),
                ((1,0),(2,0)),
                ((-1,0),(-2,0)),
        ]
        ):
            if n is None or nn is None:
                continue
            num = Piece.eval(piece.piece_type) | (Piece.eval(n.piece_type) << 1) | (Piece.eval(nn.piece_type) << 2)
            if num in (2,5):
                self.board[self.conv_coord(n.row, n.col)] = None # erase no
        return GameState.Nop

    def get_valid_moves(self, piece: Piece):
        arr = []
        r, c = piece.row, piece.col

        vert, hor = (
            reduce(
                generate_reduce(r),
                (
                    (
                        (self.get_piece(r0, c), r0),
                        (self.get_piece(ROWS - r0 - 1, c), ROWS - r0 - 1),
                    )
                    for r0 in range(0, ROWS)
                ),
                (-1, ROWS)
            ),
            reduce(
                generate_reduce(c),
                (
                    (
                        (self.get_piece(r, c0), c0),
                        (self.get_piece(r, COLS - c0 - 1), COLS - c0 - 1),
                    )
                    for c0 in range(0, COLS)
                ),
                (-1, COLS)
            ),
        )
        for i in chain(range(vert[0] + 1, r), range(r + 1, vert[1])):
            arr.append((i, c))
        for i in chain(range(hor[0] + 1, c), range(c + 1, hor[1])):
            arr.append((r, i))
        ret = set(arr)
        if piece.piece_type != PieceType.KING:
            ret.difference_update(
                set([(0, 0), (0, COLS - 1), (ROWS - 1, 0), (ROWS - 1, COLS - 1)]))
        return ret

    def __hash__(self) -> int:
        val = 0
        white = bitarray.bitarray(11*11)
        black = bitarray.bitarray(11*11)
        for ind,a in enumerate(self.board):
            if a is not None:
                if a.piece_type == PieceType.WHITE:
                    white[ind] = True
                elif a.piece_type == PieceType.BLACK:
                    black[ind] = True
                else:
                    val = ind
        return hash((white, black, val))

    def __eq__(self, other) -> bool:
        return all(a == b for a, b in zip(self.board, other.board))

def generate_reduce(val) -> Callable[[tuple[int, int], tuple[tuple[Optional[Piece], int], tuple[Optional[Piece], int]]],tuple[int, int]]:
    return lambda acc, y: (
        max(acc[0], y[0][1]) if y[0][0] is not None and y[0][1] < val else acc[0],
        min(acc[1], y[1][1]) if y[1][0] is not None and y[1][1] > val else acc[1],
    )
