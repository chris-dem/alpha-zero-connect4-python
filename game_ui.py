"""
Game UI module
"""

from dataclasses import dataclass, field

import board as b
from piece import Piece, Turn

class GameUI:
    def __init__(self, screen):
        self._init()
        self.selected = None
        self.current_state = GameState()
        self.valid_moves = {}
        self.screen = screen
        self.winner = None

    def _init(self):
        self.selected = None
        self.state = GameState()
        self.valid_moves = {}

    def update(self):
        self.board.draw(self.screen)
        self.draw_valid_moves(self.valid_moves)
        pygame.display.update()

    def reset(self):
        self._init()

    def select(self, row, col):
        if self.selected:
            result = self._move(row, col)
            if not result:
                self.selected.is_selected = False
                self.selected = None
                return True

        piece = self.board.get_piece(row, col)
        if piece is not None and Piece.compare_piece_turn(piece, self.turn):
            self.selected = piece
            self.selected.is_selected = True
            self.valid_moves = self.board.get_valid_moves(piece)
            return True

        return False

    def _move(self, row, col):
        if self.selected and (row, col) in self.valid_moves:
            res = self.board.move(self.selected, row, col)
            if res == GameState.Nop:
                self.change_turn()
            else:
                self.winner = res
        else:
            return False

        return True

    def draw_valid_moves(self, moves):
        for move in moves:
            row, col = move
            pygame.draw.circle(self.screen, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 15)

    def change_turn(self):
        self.valid_moves = {}
        if self.turn == Turn.BLACK:
            self.turn = Turn.WHITE
        else:
            self.turn = Turn.BLACK
