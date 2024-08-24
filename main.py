"""
Main game file
"""

import sys
import pygame
from game_ui import GameUI
from piece import Turn, calculate_position
from constants import COLS, PADDING, RED, ROWS, SQUARE_HEIGHT, SQUARE_WIDTH,WHITE_BASE, WIDTH, HEIGHT

# Initialize Pygame
pygame.init()

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers")


def get_row_col_from_mouse(pos):
    x, y = pos
    row = ROWS - (y // SQUARE_HEIGHT) - 1
    col = min(COLS - 1, x // SQUARE_WIDTH)
    return row, col


def draw_mouse_under(screen, turn: Turn, r: int, c: int):
    radius = min(SQUARE_WIDTH, SQUARE_HEIGHT) // 2 - PADDING
    x, y = calculate_position(r, c)
    color = RED if turn == Turn.RED else WHITE_BASE
    pygame.draw.circle(screen, color, (x, y), radius)


def main():
    run = True
    clock = pygame.time.Clock()
    game = GameUI(screen, Turn.RED)

    while run and game.winner is None:
        clock.tick(60)  # Limit the frame rate to 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                _,col = get_row_col_from_mouse(pos)
                game.select(col)
        if game.winner is not None:
            continue
        game.update()
        _, y = get_row_col_from_mouse(pygame.mouse.get_pos())
        if game.current_state.board.is_move_legal(y):
            h = game.current_state.board.get_height(y)
            draw_mouse_under(screen, game.current_state.turn, h, y)
        pygame.display.flip()
    print(game.winner)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
