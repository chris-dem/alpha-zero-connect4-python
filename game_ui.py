"""
Game UI module
"""

from typing import Optional
import torch
from ai_agent import Player
from arguments import Arguments
from mcts import MCTS
from model import Model
from piece import Turn
from game import GameState
from temperature_scheduler import AlphazeroScheduler


class GameUI:
    def __init__(self, screen, turn: Turn):
        self.current_state = GameState(turn)
        self.init_state = turn
        self.screen = screen
        self.winner = None
        self.init_player()
        if turn == Turn.RED:
            self.play_player()

    def play_player(self, prev_action: Optional[int] = None):
        action = self.ai.run(self.current_state, prev_action)
        self.act_select(action)

    def init_player(self):
        state_dict = torch.load("./checkpoints/brain_weights")
        args = Arguments()
        model = Model(args)
        model.load_state_dict(state_dict["state_dict"])
        model.eval()
        self.ai = Player(Turn.RED, MCTS(model, args), AlphazeroScheduler(args.temperature_limit))

    def _init(self):
        self.init_state = Turn(not self.init_state.value)
        self.current_state = GameState(self.init_state)
        self.winner = None

    def update(self):
        self.current_state.board.draw(self.screen)

    def reset(self):
        self._init()

    def act_select(self, col):
        self.current_state = self.current_state.move(col)
        winner = self.current_state.is_winning()
        if winner is not None:
            self.winner = winner

    def select(self, col):
        self.current_state = self.current_state.move(col)
        self.play_player(col)
