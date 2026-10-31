"""NNUE input features, bit-exact with amy's neural_net.c.

Amy (branch feature/king-half-mirrored) uses king-relative features in
a canonical frame per side.  Each side's own king defines the frame:

1. Own perspective: for black every square is rank-flipped
   (``sq ^ 0x38``, a1<->a8 style vertical flip).
2. Left-half canonicalization: if the side's own king (in the step-1
   frame) stands on files e..h, the *whole* frame is mirrored
   horizontally (``sq ^ 0x07``, file -> 7 - file), so the king is
   always in the left half (files a..d).  The two flips commute
   (independent bit masks), so a position and its horizontal mirror
   image produce the *same* feature vectors for both sides.

There are 10 piece types (0..4 = own pawn..queen, 5..9 = opponent
pawn..queen; the kings are *not* piece features), 32 canonical own
king positions, 64 piece squares, and plane 10 for the opponent king
relative to own king:

    N_FEATURES = 11 * 32 * 64 = 22528
    index = 2048 * plane + 64 * king_index + sq'

with ``sq'`` the piece square in the side's canonical frame (0..63;
pieces are not restricted to the left half, only the king) and
``king_index = file + 4 * rank`` of the canonical own king
(a1 = 0 ... d1 = 3, a2 = 4 ... d2 = 7, ..., a8 = 28 ... d8 = 31), as
in ``index_half_mirrored()`` of neural_net.c.

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
# 10 piece-type planes plus the opponent-king plane (plane 10).
N_FEATURES = 11 * 32 * 64  # 22528
ACCUMULATOR_SIZE = 256
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 32
# Vertical (rank) flip used for the black perspective.
FLIP = 0x38
# Horizontal (file) flip used to mirror the frame to the left half.
MIRROR = 0x07
# Plane stride: 32 king positions x 64 squares.
PLANE_SIZE = 32 * 64  # 2048

# Amy's Piece enum: Pawn=1..Queen=5, King=6.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6


def canonical_frame(side: int, king_sq: int) -> tuple[int, int]:
    """Return (flip_mask, king_index) of the side's canonical frame.

    ``flip_mask`` is the square transform applied to *every* square in
    the frame (0, FLIP, MIRROR, or FLIP | MIRROR); ``king_index`` is
    the own king's canonical position, 0..31, as in
    ``index_half_mirrored()`` of neural_net.c.
    """
    flip = FLIP if side == BLACK else 0
    k = king_sq ^ flip
    # Mirror the frame if the king stands on files e..h.
    if (k & 7) > 3:
        k ^= MIRROR
        flip |= MIRROR
    return flip, (k & 3) | ((k & 0x38) >> 1)


def feature_indices(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (white_accumulator_indices, black_accumulator_indices).

    ``board`` is a 64-element int8 array as in BpdRecord.board().
    Each returned array holds the 22528-space index of every active
    feature of that side's accumulator (typically 31 entries: 15 own
    and 15 opponent pieces, kings excluded, plus the opponent-king
    plane).
    """
    if (board == 6).sum() != 1 or (board == -6).sum() != 1:
        raise ValueError("board must contain exactly one king per side")
    white_king = int(np.argmax(board == 6))
    black_king = int(np.argmax(board == -6))
    w_flip, w_king = canonical_frame(WHITE, white_king)
    b_flip, b_king = canonical_frame(BLACK, black_king)

    white = []
    black = []
    for sq, value in enumerate(board):
        if value == 0:
            continue
        v = int(value)
        if v > 0:
            t = v - 1
            if t < QUEEN:  # own pawn..queen (the king is not a feature)
                white.append(PLANE_SIZE * t + 64 * w_king + (sq ^ w_flip))
                black.append(PLANE_SIZE * (t + 5) + 64 * b_king + (sq ^ b_flip))
        else:
            t = -v - 1
            if t < QUEEN:
                black.append(PLANE_SIZE * t + 64 * b_king + (sq ^ b_flip))
                white.append(PLANE_SIZE * (t + 5) + 64 * w_king + (sq ^ w_flip))
    # Plane 10: opponent king relative to own king (nn_build_accumulator).
    white.append(PLANE_SIZE * 10 + 64 * w_king + (black_king ^ w_flip))
    black.append(PLANE_SIZE * 10 + 64 * b_king + (white_king ^ b_flip))
    return (np.asarray(white, dtype=np.int32), np.asarray(black, dtype=np.int32))
