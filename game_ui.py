"""
Game UI module
"""


from piece import Turn
from game import GameState

class GameUI:
    def __init__(self, screen, turn: Turn):
        self.current_state = GameState(turn)
        self.init_state = turn
        self.screen = screen
        self.winner = None

    def _init(self):
        self.init_state = Turn(not self.init_state.value)
        self.current_state = GameState(self.init_state)
        self.winner = None

    def update(self):
        self.current_state.board.draw(self.screen)

    def reset(self):
        self._init()

    def select(self, col):
        self.current_state = self.current_state.move(col)

        winner = self.current_state.is_winning()
        if  winner is not None:
            self.winner = winner
