"""Tests for the NNUE feature mapping against hand-computed indices.

Index formula (amy neural_net.c):
    index = 4096 * piece_type + 64 * own_king_sq + own_perspective_sq
with piece_type 0..4 = own pawn..queen, 5..9 = opponent pawn..queen,
plus plane 10: 40960 + 64 * own_king_sq + opp_king_sq (opponent king
relative to own king), and black's perspective rank-flipped
(sq ^ 0x38).
"""

import numpy as np
import pytest

from amy_training_nnue.bpd import BLACK, WHITE, read_records
from amy_training_nnue.features import FLIP, N_FEATURES, feature_indices
from amy_training_nnue.position import position_from_record

INIT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def _board_from_epd(epd: str) -> np.ndarray:
    import string

    pieces = {
        "P": (1, 1),
        "N": (1, 2),
        "B": (1, 3),
        "R": (1, 4),
        "Q": (1, 5),
        "K": (1, 6),
        "p": (-1, 1),
        "n": (-1, 2),
        "b": (-1, 3),
        "r": (-1, 4),
        "q": (-1, 5),
        "k": (-1, 6),
    }
    board = np.zeros(64, dtype=np.int8)
    rows = epd.split(" ")[0].split("/")
    for rank, row in enumerate(rows):
        real_rank = 7 - rank
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                color, piece = pieces[ch]
                board[real_rank * 8 + file] = color * piece
                file += 1
    return board


def test_n_features():
    assert N_FEATURES == 11 * 64 * 64


def test_flip_is_rank_flip():
    assert FLIP == 0x38
    assert (0 ^ FLIP) == 56  # a1 <-> a8
    assert (63 ^ FLIP) == 7  # h8 <-> h1
    assert (8 ^ FLIP) == 48  # a2 <-> a6


def test_initial_position_white_features():
    board = _board_from_epd(INIT)
    w, b = feature_indices(board)
    # 15 own + 15 opponent non-king pieces plus the opponent-king plane
    # per accumulator.
    assert len(w) == 31
    assert len(b) == 31
    # All white pawns a2..h2: pt 0, king e1 (sq 4) -> 64*4 + 8..15
    assert set(range(64 * 4 + 8, 64 * 4 + 16)) <= set(w.tolist())
    # White rooks a1/h1: pt 3 -> 3*4096 + 64*4 + {0, 7}
    assert 3 * 4096 + 256 + 0 in w
    assert 3 * 4096 + 256 + 7 in w
    # White queen d1: pt 4
    assert 4 * 4096 + 256 + 3 in w
    # Black king e8 is *not* a feature; black pawns a7..h7 (sq 48..55)
    # are opponents (pt 5):
    assert 5 * 4096 + 256 + 48 in w
    assert 5 * 4096 + 256 + 55 in w
    assert 9 * 4096 + 256 + 60 not in w  # black king e8 is no piece feature
    # Plane 10: black king e8 (60) relative to white king e1 (4).
    assert 4096 * 10 + 64 * 4 + 60 in w
    # Black rooks a8/h8 as opponents: pt 8
    assert 8 * 4096 + 256 + 56 in w
    assert 8 * 4096 + 256 + 63 in w
    # Every index in range.
    assert min(w) >= 0 and max(w) < N_FEATURES
    assert min(b) >= 0 and max(b) < N_FEATURES


def test_initial_position_black_features():
    board = _board_from_epd(INIT)
    w, b = feature_indices(board)
    # Black's own pawns a7..h7 (sq 48..55), king e8 (60) -> in black's
    # perspective (sq ^ 0x38): pawns 48^0x38..55^0x38 = 8..15, king
    # 60^0x38 = 4 (e1).
    assert set(range(64 * 4 + 8, 64 * 4 + 16)) <= set(b.tolist())
    # Black's own queen d8: d8^0x38 = d1 (3) -> pt 4
    assert 4 * 4096 + 64 * 4 + 3 in b
    # Black's own rooks a8/h8: a8^0x38 = a1 (0), h8^0x38 = h1 (7), pt 3
    assert 3 * 4096 + 64 * 4 + 0 in b
    assert 3 * 4096 + 64 * 4 + 7 in b
    # White pawns a2..h2 as opponent (pt 5) in black's perspective:
    # a2^0x38 = a6 (48) .. h2^0x38 = h6 (55).
    assert 5 * 4096 + 64 * 4 + 48 in b
    assert 5 * 4096 + 64 * 4 + 55 in b
    # White queen d1^0x38 = d8 (59) as opponent (pt 9)
    assert 9 * 4096 + 64 * 4 + 59 in b
    # Plane 10: white king e1^0x38 = e8 (60) relative to black king
    # e8^0x38 = e1 (4).
    assert 4096 * 10 + 64 * 4 + 60 in b


def test_bare_kings_only_plane_10():
    """With no pieces on the board, the only active feature is the
    opponent-king plane; the kings themselves are not piece features."""
    board = _board_from_epd("4k3/8/8/8/8/8/8/4K3 w - -")
    w, b = feature_indices(board)
    # Kings: white e1 (sq 4), black e8 (sq 60).
    assert w == [4096 * 10 + 64 * 4 + 60]
    # In black's perspective (sq ^ 0x38): own king 60^0x38 = 4 (e1),
    # opponent king 4^0x38 = 60 (e8).
    assert b == [4096 * 10 + 64 * 4 + 60]


def test_black_perspective_matches_flipped_white():
    """The vertically flipped position with swapped colors must give:
    flipped_white_features == original_black_features and
    flipped_black_features == original_white_features.
    This holds by construction of the index formula for any position."""
    for epd in (
        INIT,
        # asymmetric: after 1.e4 e5 2.Nf3 Nc6
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -",
    ):
        board = _board_from_epd(epd)
        w, b = feature_indices(board)
        flipped = np.zeros(64, dtype=np.int8)
        for sq in range(64):
            flipped[sq] = -board[sq ^ FLIP]
        wf, bf = feature_indices(flipped)
        assert sorted(wf.tolist()) == sorted(b.tolist())
        assert sorted(bf.tolist()) == sorted(w.tolist())


def test_missing_king_rejected():
    board = np.zeros(64, dtype=np.int8)
    board[0] = 1
    with pytest.raises(ValueError):
        feature_indices(board)


def test_en_passant_square_is_empty_for_features():
    # black pawn just played d7-d5, ep target d6 (sq 43); the target
    # square must carry no feature for either accumulator
    board = _board_from_epd("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6")
    w, b = feature_indices(board)
    ep = 43  # d6
    assert 4096 * 0 + 64 * 4 + ep not in w  # own pawn (white view)
    assert 4096 * 5 + 64 * 4 + ep not in w  # opponent pawn (white view)
    assert 4096 * 0 + 64 * 4 + (ep ^ FLIP) not in b
    assert 4096 * 5 + 64 * 4 + (ep ^ FLIP) not in b
