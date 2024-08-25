"""
Trainer for the hnefatafl AI
"""

import os
from random import random

from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from arguments import Arguments
from dataset import GameDataSet
from game import GameState
from piece import Turn, convert_status_to_score
from mcts import MCTS
from ai_agent import Player
from model import Model


class Trainer:

    def __init__(self, args: Arguments):
        self.args = args
        device = torch.device("mps")
        self.model = Model(self.args).to(device)
        self._init()

    def _init(self):
        device = torch.device("mps")
        m1 = Model(self.args).to(device)
        m2 = Model(self.args).to(device)
        m1.load_state_dict(self.model.state_dict())
        m2.load_state_dict(self.model.state_dict())
        self.players = (
            Player(Turn.RED, MCTS(m1, self.args)),
            Player(Turn.YELLOW, MCTS(m2, self.args)),
        )

    def exceute_episode(self):

        current_player = round(random())
        state = GameState(self.players[current_player].player)
        action = None
        it = 0

        while True:
            curr_ai = self.players[current_player]
            action = curr_ai.run(state, action, it, True)
            curr_ai.mcts.move_head(action)
            state = state.move(action)
            assert state == curr_ai.mcts.root.state
            reward = state.is_winning()
            reward = convert_status_to_score(reward) if reward is not None else None
            it += 1
            current_player = 1 - current_player
            if reward is not None:
                ret: list[tuple[GameState, list[float], float]] = []
                for (
                    hist_current_player,
                    hist_state,
                    hist_action_probs,
                ) in (
                    self.players[0].train_logger + self.players[1].train_logger
                ):
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append(
                        (
                            hist_state,
                            hist_action_probs,
                            reward * ((-1) ** (hist_current_player != current_player)),
                        )
                    )

                return ret

    def learn(self):
        """
        Training loop
        """
        for _ in trange(0, self.args.num_iters + 1, desc="Number of iterations"):
            train_examples = []
            for _ in trange(0, self.args.num_eps, desc="Episodes", leave=False):
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

            self.train(train_examples)
            filename = self.args.checkpoint_path
            self.save_checkpoint(folder="./checkpoints", filename=filename)

    def train(self, examples: list[tuple[GameState, list[float], float]]):
        model = self.players[0].mcts.model
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        pi_losses = []
        v_losses = []
        dataset = GameDataSet(examples)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        for _ in trange(0, self.args.num_epochs, desc="Epochs", leave=False):
            model.train()

            for batch in tqdm(loader, desc="Batches", leave=False):
                device = torch.device("mps")
                boards, pis, vs = batch

                # predict
                boards = boards.contiguous().to(device)
                target_pis = pis.contiguous().to(device)
                target_vs = vs.contiguous().to(device)

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(out_pi, target_pis)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            # print(out_pi[0].detach())
            # print(target_pis[0])

    def loss_pi(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """
        Policy head loss function
        """
        loss = F.cross_entropy(inputs, outputs)
        return loss

    def loss_v(self, targets: torch.Tensor, outputs: torch.Tensor):
        """
        Value head loss function
        """
        loss = F.mse_loss(targets, outputs)
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
            },
            filepath,
        )
