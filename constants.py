"""
Global constant file
"""

import itertools as itools

# Screen dimensions
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 11, 11
SQUARE_SIZE = WIDTH // COLS

# Colors
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)
WHITE = (255, 255, 255)
WHITE_BASE = (50, 168, 74)
BLACK_BASE = (50, 152, 168)
EXIT_BASE = (140, 50, 168)
SELECTED = (214, 211, 150)

## PIECE COLORS
WHITE_PIECE = "#e7264c"
BLACK_PIECE = "#a192cc"
KING_PIECE = "#cb42f5"

WHITE_BASE_SQUARES = set(
    itools.chain(
        tuple((5 + a, 5 + b) for (a, b) in itools.product([-1, 0, 1], [-1, 0, 1])),
        tuple((5 + a, 5 + b) for (a, b) in [(0, -2), (0, 2), (2, 0), (-2, 0)]),
    )
)

EXIT_BASE_SQUARES = set([(0, 0), (0, 10), (10, 0), (10, 10)])

BLACK_BASE_SQUARES = set(
    [
        (3, 0),
        (4, 0),
        (5, 0),
        (5, 1),
        (6, 0),
        (7, 0),
        (3, 10),
        (4, 10),
        (5, 10),
        (5, 9),
        (6, 10),
        (7, 10),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 5),
        (0, 6),
        (0, 7),
        (10, 3),
        (10, 4),
        (10, 5),
        (9, 5),
        (10, 6),
        (10, 7),
    ]
)
