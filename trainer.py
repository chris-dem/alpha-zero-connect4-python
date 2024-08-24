"""
Trainer for the hnefatafl AI
"""
import os
from random import shuffle

import numpy as np
import torch
from torch import optim
from random import random
from arguments import Arguments
from game_ui import GameUI
from piece import Turn
from mcts import MCTS
from ai_agent import Player
from model import Model

class Trainer:

    def __init__(self, game : GameUI, model: torch.Module, args: Arguments):
        self.game = game
        self.args = args
        m1 = Model(self.args)
        m1.load_state_dict(model.state_dict())
        m2 = Model(self.args)
        m2.load_state_dict(model.state_dict())
        self.player_red = Player(Turn.RED, MCTS(m1, self.args))
        self.player_ylw = Player(Turn.YELLOW, MCTS(m2, self.args))

    def exceute_episode(self):

        train_examples = []
        current_player = Turn.RED if random() < 0.5 else Turn.YELLOW
        state = self.game.current_state

        while True:
            # # canonical_board = self.game.get_canonical_board(state, current_player)
            # action_probs = [0 for _ in range(self.game.get_action_size())]
            # for k, v in self.mcts.children.items():
            #     action_probs[k] = v.visit_count
            #
            # action_probs = action_probs / np.sum(action_probs)
            # train_examples.append((state, current_player, action_probs))
            #
            # action = root.select_action(temperature=0)
            # state, current_player = self.game.get_next_state(state, current_player, action)
            # reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

                return ret

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):

            print("{}/{}".format(i, self.args['numIters']))

            train_examples = []

            for eps in range(self.args['numEps']):
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args['checkpoint_path']
            self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args['epochs']):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                boards = boards.contiguous().cuda()
                target_pis = target_pis.contiguous().cuda()
                target_vs = target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])

    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.model.state_dict(),
        }, filepath)
