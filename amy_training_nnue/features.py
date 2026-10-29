"""NNUE input features, bit-exact with amy's neural_net.c.

Amy (branch feature/neural-net-with-clamped-relu) uses king-relative
features: for each side, one feature per (piece type, own king square,
piece square).  There are 10 piece types (0..4 = own pawn..queen,
5..9 = opponent pawn..queen; the kings are *not* input features),
64 own king squares and 64 piece squares:

    N_FEATURES = 10 * 64 * 64 = 40960
    index = 4096 * piece_type + 64 * king_sq + piece_sq

Squares are stored in the *own side's perspective*: for white that is
the normal board, for black both the king square and the piece square
are rank-flipped (``sq ^ 0x38``, i.e. a1<->a8 style vertical flip, as
in nn_update_weights_for_king()/nn_update_weights_for_piece()).

Each side has its own ACCUMULATOR_SIZE-dim accumulator (int16 in amy)
initialized with the input bias plus the sum of the weights of the
active features; the side-to-move accumulator is the first half and the
opponent's the second half of the 2 * ACCUMULATOR_SIZE-dim network
input.
"""

from __future__ import annotations

import numpy as np

from .bpd import BLACK, WHITE

N_PIECE_TYPES = 10
N_FEATURES = 40960
ACCUMULATOR_SIZE = 128
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 32
# Vertical (rank) flip used for the black perspective.
FLIP = 0x38

# Amy's Piece enum: Pawn=1..Queen=5, King=6.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6


def feature_indices(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (white_accumulator_indices, black_accumulator_indices).

    ``board`` is a 64-element int8 array as in BpdRecord.board().
    Each returned array holds the 40960-space index of every active
    feature of that side's accumulator (typically 30 entries: 15 own
    and 15 opponent pieces, kings excluded).
    """
    if (board == 6).sum() != 1 or (board == -6).sum() != 1:
        raise ValueError("board must contain exactly one king per side")
    white_king = int(np.argmax(board == 6))
    black_king = int(np.argmax(board == -6))
    black_king_persp = black_king ^ FLIP

    white = []
    black = []
    for sq, value in enumerate(board):
        if value == 0:
            continue
        v = int(value)
        if v > 0:
            t = v - 1
            if t < QUEEN:  # own pawn..queen (the king is not a feature)
                white.append(4096 * t + 64 * white_king + sq)
                black.append(4096 * (t + 5) + 64 * black_king_persp + (sq ^ FLIP))
        else:
            t = -v - 1
            if t < QUEEN:
                black.append(4096 * t + 64 * black_king_persp + (sq ^ FLIP))
                white.append(4096 * (t + 5) + 64 * white_king + sq)
    return (np.asarray(white, dtype=np.int32), np.asarray(black, dtype=np.int32))


def perspective_square(side: int, sq: int) -> int:
    """Square in the given side's own perspective (rank flip for black)."""
    return sq if side == WHITE else sq ^ FLIP
