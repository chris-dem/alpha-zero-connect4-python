"""
Main file ussed to train the AI and save the state
"""
import sys

from arguments import Arguments
from trainer import Trainer

def main():
    args = Arguments()
    trainer = Trainer(args)
    if sys.argv[1] == "final":
        trainer.load_checkpoint("checkpoints", "final")
    elif sys.argv[1] == "load":
        trainer.load_checkpoint("checkpoints", "brain_weights")
    trainer.learn()
    trainer.save_checkpoint("checkpoints", "final")

if __name__ == "__main__":
    main()
