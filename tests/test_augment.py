"""Tests for horizontal mirror dataset augmentation.

A position is mirrored across the vertical board axis (file flip,
``sq ^ MIRROR_FLIP``).  Positions without castling rights are
augmented with their mirror image; positions with castling rights are
duplicated so that every original position keeps the same weight in
the dataset.
"""

from pathlib import Path

import numpy as np
import pytest

from amy_training_nnue.bpd import WHITE, encode_board, read_records, write_records
from amy_training_nnue.dataset import BpdDataset
from amy_training_nnue.features import feature_indices
from amy_training_nnue.position import (
    MIRROR_FLIP,
    mirror_board,
    position_from_record,
)

DATA = Path(__file__).parent / "data" / "bpd_test.bin"


def _board_from_dict(pieces: dict[int, int]) -> np.ndarray:
    board = np.zeros(64, dtype=np.int8)
    for sq, value in pieces.items():
        board[sq] = value
    return board


# A middlegame position with all ten piece types active on both sides
# and both kings off their starting files.
MIDDLEGAME = {
    4: 6,  # K e1
    3: 5,  # Q d1
    0: 4,  # R a1
    7: 4,  # R h1
    21: 2,  # N f3
    26: 3,  # B c4
    8: 1,  # P a2
    27: 1,  # P d4
    28: 1,  # P e4
    60: -6,  # k e8
    59: -5,  # q d8
    63: -4,  # r h8
    42: -2,  # n c6
    52: -3,  # b e7
    48: -1,  # p a7
    49: -1,  # p b7
    35: -1,  # p d5
    36: -1,  # p e5
}


def test_mirror_flip_is_file_flip():
    assert MIRROR_FLIP == 0x07
    assert 0 ^ MIRROR_FLIP == 7  # a1 <-> h1
    assert 8 ^ MIRROR_FLIP == 15  # a2 <-> h2
    assert 56 ^ MIRROR_FLIP == 63  # a8 <-> h8


def test_mirror_board_is_involution_and_preserves_pieces():
    board = _board_from_dict(MIDDLEGAME)
    once = mirror_board(board)
    np.testing.assert_array_equal(mirror_board(once), board)
    assert np.array_equal(np.sort(once), np.sort(board))
    # The piece on a1 ends up on h1 and vice versa.
    assert once[7] == board[0]
    assert once[0] == board[7]
    # Kings stay kings, colors are preserved.
    assert (once == 6).sum() == 1
    assert (once == -6).sum() == 1


def _file_flip_features(feat: np.ndarray) -> np.ndarray:
    """File-flip the king-sq and piece-sq parts of feature indices.

    index = 4096 * plane + 64 * king_sq + piece_sq, and the mirror
    maps king_sq -> king_sq ^ MIRROR_FLIP, piece_sq -> piece_sq ^
    MIRROR_FLIP (including black's rank-flipped perspective squares
    and plane 10, since 0x07 commutes with the 0x38 rank flip).
    """
    plane = feat // 4096
    king_sq = (feat // 64) % 64
    piece_sq = feat % 64
    return 4096 * plane + 64 * (king_sq ^ MIRROR_FLIP) + (piece_sq ^ MIRROR_FLIP)


def test_mirrored_features_are_file_flip_of_original_features():
    """The mirror's feature indices are the original's indices with the
    file bits of both the king-sq and the piece-sq parts flipped."""
    board = _board_from_dict(MIDDLEGAME)
    fw, fb = feature_indices(board)
    mw, mb = feature_indices(mirror_board(board))
    assert sorted(mw.tolist()) == sorted(_file_flip_features(fw).tolist())
    assert sorted(mb.tolist()) == sorted(_file_flip_features(fb).tolist())


def test_dataset_mirroring_off_by_default():
    ds = BpdDataset([str(DATA)])
    assert ds.n == 5
    assert ds.n_mirrored == 0
    assert ds.n_duplicated == 0


def test_dataset_mirrors_clean_positions_and_duplicates_castling_positions():
    """bpd_test.bin holds 3 records with castling rights (duplicated
    verbatim) and 2 without (original + mirror image)."""
    base = BpdDataset([str(DATA)])
    aug = BpdDataset([str(DATA)], mirror_augment=True)

    assert aug.n == 10
    assert aug.n_mirrored == 2
    assert aug.n_duplicated == 3

    # First 3 records have castling rights: duplicated verbatim.
    for i in range(3):
        np.testing.assert_array_equal(aug.feat_w[2 * i], aug.feat_w[2 * i + 1])
        np.testing.assert_array_equal(aug.feat_b[2 * i], aug.feat_b[2 * i + 1])
        assert aug.score[2 * i] == aug.score[2 * i + 1]
        assert aug.turn[2 * i] == aug.turn[2 * i + 1]
        # The originals are untouched and in base order.
        np.testing.assert_array_equal(aug.feat_w[2 * i], base.feat_w[i])
        assert aug.score[2 * i] == base.score[i]

    # Last 2 records have no castling rights: the second copy is the
    # mirror image.
    records = list(read_records(DATA))
    for j, record in enumerate(records[3:]):
        i = 6 + 2 * j
        expected_w, expected_b = feature_indices(mirror_board(record.board()))
        np.testing.assert_array_equal(aug.feat_w[i + 1, : len(expected_w)], expected_w)
        np.testing.assert_array_equal(aug.feat_b[i + 1, : len(expected_b)], expected_b)
        assert aug.score[i + 1] == record.score
        assert aug.turn[i + 1] == record.turn

    # Every record contributes its score exactly twice.
    np.testing.assert_array_equal(aug.score[::2], base.score)
    np.testing.assert_array_equal(aug.score[1::2], base.score)


def test_augmentation_preserves_score_statistics():
    base = BpdDataset([str(DATA)])
    aug = BpdDataset([str(DATA)], mirror_augment=True)
    assert np.mean(aug.score) == pytest.approx(np.mean(base.score))
    assert np.std(aug.score) == pytest.approx(np.std(base.score))


def test_en_passant_square_does_not_block_mirroring(tmp_path):
    """An EP target square (not a network feature) must not prevent
    mirroring a position without castling rights.  White pawn on e5,
    black just played d7-d5 (pawn on d5, EP target d6), white to move."""
    board = _board_from_dict(
        {
            4: 6,  # K e1
            36: 1,  # P e5
            35: -1,  # p d5
            60: -6,  # k e8
        }
    )
    record = encode_board(board, turn=WHITE, score=1234, castle=0, ep=43)
    pos = position_from_record(record)
    assert pos.castle == 0
    assert pos.ep == 43  # d6

    path = tmp_path / "ep.bin"
    write_records(str(path), [record])

    aug = BpdDataset([str(path)], mirror_augment=True)
    assert aug.n == 2
    assert aug.n_mirrored == 1
    np.testing.assert_array_equal(aug.score, [1234, 1234])
    np.testing.assert_array_equal(aug.turn, [WHITE, WHITE])

    expected_w, expected_b = feature_indices(mirror_board(board))
    np.testing.assert_array_equal(aug.feat_w[1, : len(expected_w)], expected_w)
    np.testing.assert_array_equal(aug.feat_b[1, : len(expected_b)], expected_b)


def test_invalid_records_are_skipped_when_augmenting(tmp_path):
    # No black king: invalid, skipped (also when augmenting).
    bad = encode_board(_board_from_dict({4: 6, 8: 1}), turn=WHITE, score=0, castle=0)
    # Bare kings, black to move, no castling: valid, mirrored.
    good = encode_board(_board_from_dict({4: 6, 60: -6}), turn=1, score=-50)
    path = tmp_path / "mix.bin"
    write_records(str(path), [bad, good])

    ds = BpdDataset([str(path)], mirror_augment=True)
    assert ds.n_invalid == 1
    assert ds.n == 2
    assert ds.n_mirrored == 1
    assert ds.n_duplicated == 0
