"""
Main game file
"""
import sys
import pygame
from game import Game
from piece import Piece
from constants import *

# Initialize Pygame
pygame.init()

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers")

def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

def draw_mouse_under(r: int, c: int):

    radius = SQUARE_SIZE // 2 - Piece.PADDING
    circle = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    circle.set_alpha(100)
    pygame.draw.circle(
        circle, GREY, (radius, radius), radius
    )
    x, y = Piece.calc_pos(r, c)
    screen.blit(circle, (x - radius,y - radius))

def main():
    run = True
    clock = pygame.time.Clock()
    game = Game(screen)

    while run and game.winner is None:
        clock.tick(60)  # Limit the frame rate to 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                game.select(row, col)
                # if selected_piece:
                #     board.move(selected_piece, row, col)
                #     selected_piece.is_selected = False
                #     selected_piece = None
                # else:
                #     piece = board.board[board.conv_coord(row, col)]
                #     if piece is not None:
                #         selected_piece = piece
                #         selected_piece.is_selected = True

        game.update()
        x, y = get_row_col_from_mouse(pygame.mouse.get_pos())
        draw_mouse_under(x, y)
        pygame.display.flip()
    print(game.winner)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
