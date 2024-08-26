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
from temperature_scheduler import AlphazeroScheduler


class Trainer:

    def __init__(self, args: Arguments):
        self.args = args
        device = torch.device("mps")
        self.model = Model(self.args).to(device)
        self._init(args)
    
    def load_checkpoint(self, folder, filename):
        state_dict = torch.load(os.path.join(folder, filename))
        self.model.load_state_dict(state_dict["state_dict"])

    def _init(self, args: Arguments):
        m1 = Model(self.args)
        m2 = Model(self.args)
        m1.load_state_dict(self.model.state_dict())
        m2.load_state_dict(self.model.state_dict())
        self.players = (
            Player(Turn.RED, MCTS(m1, self.args),AlphazeroScheduler(args.temperature_limit)),
            Player(Turn.YELLOW, MCTS(m2, self.args),AlphazeroScheduler(args.temperature_limit)),
        )

    def exceute_episode(self):
        current_player = first_player = round(random())
        state = GameState(self.players[current_player].player)
        action = None
        it = 0

        while True:
            curr_ai = self.players[current_player]
            action = curr_ai.run(state, action, it, True)
            curr_ai.mcts.move_head(action)
            state = state.move(action)
            assert state == curr_ai.mcts.root.state
            v = state.is_winning()
            reward = (
                convert_status_to_score(v, state.turn)
                if v is not None
                else None
            )
            it += 1
            current_player = 1 - current_player
            if reward is not None:
                ret: list[tuple[GameState, list[float], float]] = []
                for ind, (
                    hist_current_player,
                    hist_state,
                    hist_action_probs,
                    _,
                ) in enumerate(
                    self.players[0].train_logger + self.players[1].train_logger
                ):
                    ret.append(
                        (
                            hist_state,
                            hist_action_probs,
                            reward * ((-1) ** (hist_current_player != state.turn)),
                        )
                    )
                    # moves.append(
                    #     self.players[(first_player + ind) % 2].train_logger[ind // 2]
                    # )
                # with open("moves.txt", "a") as f:
                #
                #     print("Game", file=f)
                #     print(v, file=f)
                #     for el in moves:
                #         print(el[0], file=f)
                #         print(el[1].print_debug(), file=f)
                #         print(el[2], file=f)
                #         print(el[3], file=f)
                #         print("-------", file=f)
                #     print("-------", file=f)
                return ret,max(len(i.train_logger) for i in self.players) , convert_status_to_score(v, self.players[first_player].player)

    def clear(self):
        for player in self.players:
            player.train_logger.clear()

    def learn(self):
        """
        Training loop
        """
        for _ in trange(0, self.args.num_iters + 1, desc="Number of iterations"):
            train_examples = []
            val = {
                "wins": [],
                "losses": [],
                "draws": 0,
            }
            for _ in trange(0, self.args.num_eps, desc="Episodes", leave=False):
                ex, m, player = self.exceute_episode()
                if player == 0:
                    val["draws"] += 1
                else:
                    if player > 0:
                        val["wins"].append(m)
                    else:
                        val["losses"].append(m)
                train_examples.extend(ex)
                self.clear()

            print(f"wins: {np.mean(val['wins'])}, losses: {np.mean(val['losses'])}, draws: {val['draws']}")

            self.train(train_examples)
            filename = self.args.checkpoint_path
            self.save_checkpoint(folder="./checkpoints", filename=filename)

    def train(
        self,
        examples: list[tuple[GameState, list[float], float]],
    ):
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        dataset = GameDataSet(examples)

        with tqdm(
            range(self.args.num_epochs),
            desc="Epochs",
            postfix=[0, 0, 0],
        ) as t:
            for _ in t:
                model.train()
                pi_losses = []
                v_losses = []
                outs = []
                loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
                for batch in tqdm(loader, desc="Batches", leave=False):
                    device = torch.device("mps")
                    boards, pis, vs = batch

                    # predict
                    boards = boards.contiguous().to(device)
                    target_pis = pis.contiguous().to(device)
                    target_vs = vs.contiguous().to(device)

                    # compute output
                    out_pi, out_v = self.model(boards)
                    outs.append(out_v.cpu().detach().numpy())
                    l_pi = self.loss_pi(out_pi, target_pis)
                    l_v = self.loss_v(out_v, target_vs)
                    total_loss = l_pi + l_v

                    pi_losses.append(float(l_pi))
                    v_losses.append(float(l_v))

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                t.postfix[0], t.postfix[1], t.postfix[2] = (
                    np.mean(pi_losses).item(),
                    np.mean(v_losses).item(),
                    np.mean(np.concatenate(outs)).item(),
                )
                t.update()

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
        loss = F.mse_loss(targets, outputs[:, None])
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
