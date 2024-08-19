"""
Main game file
"""

import sys
import pygame
from game_ui import GameUI
from piece import Piece, Turn, calculate_position
from constants import PADDING, RED, ROWS, SQUARE_SIZE, WIDTH, HEIGHT, GREY

# Initialize Pygame
pygame.init()

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers")


def get_row_col_from_mouse(pos):
    x, y = pos
    row = ROWS - (y // SQUARE_SIZE) - 1
    col = x // SQUARE_SIZE
    return row, col


def draw_mouse_under(screen, r: int, c: int):
    radius = SQUARE_SIZE // 2 - PADDING
    x, y = calculate_position(r, c)
    pygame.draw.circle(screen, RED, (x, y), radius)
    # screen.blit(circle, (x - radius,y - radius))


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
                col = get_row_col_from_mouse(pos)
                game.select(col)

        game.update()
        _, y = get_row_col_from_mouse(pygame.mouse.get_pos())
        if game.current_state.board.is_move_legal(y):
            h = game.current_state.board.get_height(y)
            draw_mouse_under(screen, h, y)
        pygame.display.flip()
    print(game.winner)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
