import tensorflow as tf
import numpy as np
import random
import sys
import time
from tictactoe import Board, bitboard_to_list
from abc import ABC, abstractmethod


class Player(ABC):
    @abstractmethod
    def make_move(self, board):
        pass


class HumanPlayer(Player):
    def make_move(self, board):
        """Ask for a move until a legal one is given."""
        move = None
        while move not in board.moves():
            try:
                text = input("Enter move (1-9): ")
                move = 1 << int(text) - 1
            except:
                if text.lower().startswith("q"):
                    sys.exit(0)
                print("No! Enter a single number, 1-9")
        return board.do_move(move)


class TicTacNET(Player):
    def __init__(self, model_file="tictacNET.h5"):
        self.model = tf.keras.models.load_model(model_file)

    def make_move(self, board):
        """Pick the most preferred, legal move."""
        print("Consulting neurons...")
        time.sleep(random.random() * 3)

        inputs = np.asarray(
            [sum(map(bitboard_to_list, board.squares), []) + [board.turn]]
        )
        outputs = np.argsort(self.model.predict(inputs)[0])
        for output in outputs:
            move = 1 << output
            if move in board.moves():
                return board.do_move(move)
        assert False, "No move could be made!"


def play(ai_starts=False):
    """Play a human vs. AI in an infinite death match."""
    human = HumanPlayer()
    ai = TicTacNET()
    scores = [0, 0]
    while True:
        b = Board()
        print(b, end="\n\n")
        i = random.randint(0, 1)  # Decide who starts.
        while not b.is_decided:
            player = ai if i & 1 else human
            b = player.make_move(b)
            print(b, end="\n\n")
            i += 1
        score = b.score
        if score and i & 1:
            scores[0] += 1
        elif score:
            scores[1] += 1
        else:
            scores[0] += 0.5
            scores[1] += 0.5
        print("\n\t\033[1mScore: {} - {}\033[0m\n".format(*scores))


if __name__ == "__main__":
    play()
