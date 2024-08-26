"""
Monte carlo Tree search

Negative eval: Black
Positive eval: White
"""

from dataclasses import dataclass, field
import math
from typing import Optional, Self, cast

import torch
import numpy as np
import numpy.typing as npt
from scipy.special import softmax

from arguments import Arguments
from game import GameState
from piece import Turn, convert_status_to_score


def ucb_score(parent, child, expl_param=math.sqrt(2)):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = (
        child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    )
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + expl_param * prior_score


@dataclass
class Node:
    """
    Node class
    Contains:
        visit_count: How many times this node has been visited
        to_play: Which player is to play
        prior: Prior probability
        value_sum: (I guess combined value of its children)
        children: Action-Board dictionary
    """

    prior: float
    state: GameState
    visit_count: int = 0
    value_sum: float = 0
    children: dict[int, Self] = field(default_factory=dict)

    def expanded(self):
        """
        Return number of children
        """
        return len(self.children) > 0

    def value(self):
        """
        Average value of its children
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature: float) -> int:
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())
        if temperature < 1e-6:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            #  As the temperature increase, we are more bound to choose better choices
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / np.sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)
        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, state: GameState, action_probs: npt.ArrayLike):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                new_state = self.state.move(a)
                self.children[a] = cast(Self, Node(prob, new_state))

    def __hash__(self) -> int:
        return self.state.__hash__()

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        return f"{self.state} Prior: {self.prior:.2f} Count: {self.visit_count,} Value: {\
                                self.value()}"


@dataclass
class MCTS:
    """
    Monte Carlo Tree Search agent
    """

    model: torch.nn.Module
    args: Arguments
    root: Optional[Node] = None

    def select_action(self, state: GameState):
        st = state.canonical_representation()
        action_probs, value = self.model.predict(st)
        valid_moves = np.array(state.get_valid_moves())
        action_probs = softmax(action_probs) * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)  # mask invalid moves

        return action_probs, value

    def move_head(self, action: int):
        if self.root is not None:
            self.root = self.root.children[action]

    def run(self, state: GameState, action: Optional[int] = None):
        """
        Run mcts given current state and previous action
        """
        if (
            self.root is not None
            and action is not None
            and action in self.root.children.keys()
            and self.root.children[action].state == state
        ):
            self.root = self.root.children[action]
        else:
            self.root = Node(0, state)
            self.root.state = state
        assert self.root is not None
        if not self.root.expanded():
            action_probs, value = self.select_action(state)
            self.root.expand(state, action_probs)

        for _ in range(self.args.num_simulations):
            node = self.root
            search_path: list[Optional[Node]] = [node]
            action = None

            # SELECT
            while node is not None and node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            assert len(search_path) > 1
            parent = cast(Node, search_path[-2])
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            state = state.move(cast(int, action))
            # Get the board from the perspective of the other player
            value = state.is_winning()
            value = (
                convert_status_to_score(value, state.turn)
                if value is not None
                else None
            )
            if value is None:
                # If the game has not ended:
                # EXPAND
                action_probs, value = self.select_action(state)

                node = cast(Node, node)
                node.expand(state, action_probs)

            self.backpropagate(cast(list[Node], search_path), value, Turn(not parent.state.turn))

    def backpropagate(self, search_path: list[Node], value: float, to_play: Turn):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.state.turn == to_play else -value
            node.visit_count += 1
