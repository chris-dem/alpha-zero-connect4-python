"""
Global constant file
"""

from itertools import product
from functools import reduce
from bitarray import bitarray


# Screen dimensions
WIDTH, HEIGHT =  600, 600
ROWS, COLS = 6, 7
SQUARE_WIDTH = WIDTH // COLS
SQUARE_HEIGHT = HEIGHT // ROWS
PADDING = 15
OUTLINE = 2

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

# Value
MAX_SCORE = 1

# Number vals
ROW_CHECK = [
    (0b1000001000001000001 << ROWS * n4) << nr
    for nr, n4 in product(range(ROWS), range(4))
]

COL_CHECK = [(0b1111 << n4) << ROWS * nc for nc, n4 in product(range(COLS), range(3))]


def generate_diag_check():
    arr = []
    for v in [(3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3)]:
        start = v
        while start[0] + 3 < ROWS and start[1] + 3 < COLS:
            db = bitarray("0" * ROWS * COLS)
            for i in range(4):
                ind = (start[1] + i) * ROWS + (start[0] + i)
                db[ind] = 1
            arr.append(db)
            start = start[0] + 1, start[1] + 1

    for v in [(3, 0), (4, 0), (5, 0), (5, 1), (5, 2), (5, 3)]:
        start = v
        while start[0] - 3 >= 0 and start[1] + 3 < COLS:
            db = bitarray("0" * ROWS * COLS)
            for i in range(4):
                ind = (start[0] - i) + (start[1] + i) * ROWS
                db[ind] = 1
            arr.append(db)
            start = start[0] - 1, start[1] + 1
    return list(map(conv_to_num, arr))


def conv_to_num(bt: bitarray) -> int:
    return reduce(lambda a, b: a << 1 | b, bt[::-1], 0)


DIAG_CHECK = generate_diag_check()

CHECK = ROW_CHECK + COL_CHECK + DIAG_CHECK
FULL_BOARD = 2**(ROWS*COLS) - 1

def print_num(n: int):
    st = f"{n:0{ROWS*COLS}b}"[::-1]

    s = []
    for i in range(ROWS):
        t = ""
        for j in range(COLS):
            t += st[i + j * ROWS]
        s.append(t)

    print(*s[::-1], sep="\n")


if __name__ == "__main__":

    def main():
        # print("====ROWS====")
        # for r in ROW_CHECK:
        #     print_num(r)
        #     print("--")
        # print("====COLS====")
        # for r in COL_CHECK:
        #     print_num(r)
        #     print("--")
        print("====DIAG====")
        for r in DIAG_CHECK:
            print_num(r)
            print("--")
        print_num(FULL_BOARD)

    main()
