"""
Main file ussed to train the AI and save the state
"""

from arguments import Arguments
from trainer import Trainer

def main():
    args = Arguments()
    trainer = Trainer(args)
    trainer.learn()

if __name__ == "__main__":
    main()
