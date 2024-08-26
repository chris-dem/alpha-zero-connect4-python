"""
Main file ussed to train the AI and save the state
"""

from arguments import Arguments
from trainer import Trainer

def main():
    args = Arguments()
    trainer = Trainer(args)
    trainer.learn()
    trainer.save_checkpoint("checkpoints", "final")

if __name__ == "__main__":
    main()
