"""Bit-exact cross-check against amy's C implementation.

tests/data/crosscheck_reference.json holds the accumulators, scores
and feature index sets (piece, pawn, attack) produced by amy's own
neural_net.c (branch feature/king-half-mirrored, src/nn_crosscheck.c)
for a random in-range network generated with a fixed seed.  The Python
QuantizedNet must reproduce every accumulator value and score,
including int16 wrap-around, the [0, 64] and [0, 8192] clamps, and the
two-step rounded output transform.  The piece/pawn index sets are
checked against the Python oracle (features.py), and the attack index
sets (C-only by design) against the C extraction library.  Regenerate
the reference with tools/make_crosscheck_reference.py (built by make
in the amy checkout).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from amy_training_nnue import cfeatures
from amy_training_nnue.bpd import BLACK, WHITE, encode_board
from amy_training_nnue.features import (
    ACCUMULATOR_SIZE,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    N_FEATURES,
    N_PAWN_FEATURES,
    feature_indices,
    pawn_feature_indices,
)
from amy_training_nnue.quantize import QuantizedNet, quantize_parameters

# Padding widths for the synthetic batches below (real dataset widths
# come from the C feature-extraction constants, see cfeatures.py).
MAX_FEATURES = 32
MAX_PAWN_FEATURES = 128

DATA = Path(__file__).parent / "data" / "crosscheck_reference.json"


def _make_network(seed: int) -> QuantizedNet:
    """Regenerate the deterministic random network from the reference.

    Must mirror tools/make_crosscheck_reference.py exactly (same seed,
    same draw order; input_w and pawn_w are drawn from [-1, 1], the
    range representable by int8 at scale 64).
    """
    rng = np.random.default_rng(seed)
    params = {
        "input_w": rng.uniform(-1, 1, (N_FEATURES, ACCUMULATOR_SIZE)),
        "input_b": rng.uniform(-1, 1, (ACCUMULATOR_SIZE,)),
        "pawn_w": rng.uniform(-1, 1, (N_PAWN_FEATURES, ACCUMULATOR_SIZE)),
        "h1_w": rng.uniform(-2, 2, (2 * ACCUMULATOR_SIZE, HIDDEN_SIZE_1)),
        "h1_b": rng.uniform(-2, 2, (HIDDEN_SIZE_1,)),
        "h2_w": rng.uniform(-2, 2, (HIDDEN_SIZE_1, HIDDEN_SIZE_2)),
        "h2_b": rng.uniform(-2, 2, (HIDDEN_SIZE_2,)),
        "out_w": rng.uniform(-1, 1, (HIDDEN_SIZE_2,)),
        "out_b": rng.uniform(-1, 1, (1,)),
    }
    return QuantizedNet.from_arrays(quantize_parameters(params))


def _board_from_epd(epd: str) -> np.ndarray:
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


def _pad(feat: np.ndarray) -> np.ndarray:
    out = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    out[0, : len(feat)] = feat
    return out


def _pad_pawn(feat: np.ndarray) -> np.ndarray:
    out = np.full((1, MAX_PAWN_FEATURES), -1, dtype=np.int32)
    out[0, : len(feat)] = feat
    return out


@pytest.fixture(scope="module")
def reference():
    return json.loads(DATA.read_text())


def test_accumulators_match_c(reference):
    """Raw (int16 wrap-around) accumulators must equal amy's, including
    the pawn pair rows."""
    ref = reference["positions"][0]
    qnet = _make_network(reference["seed"])
    board = _board_from_epd(ref["epd"])
    fw, fb = feature_indices(board)
    pfw, pfb = pawn_feature_indices(board)
    acc_w = qnet.accumulator(_pad(fw), _pad_pawn(pfw))[0]
    acc_b = qnet.accumulator(_pad(fb), _pad_pawn(pfb))[0]
    assert np.array_equal(acc_w, np.asarray(ref["acc_w"], dtype=np.int32))
    assert np.array_equal(acc_b, np.asarray(ref["acc_b"], dtype=np.int32))


def test_scores_match_c(reference):
    """Scores must equal amy's, for every reference position."""
    qnet = _make_network(reference["seed"])
    for ref in reference["positions"]:
        board = _board_from_epd(ref["epd"])
        turn = WHITE if " w " in ref["epd"] else BLACK
        assert turn == ref["turn"]
        fw, fb = feature_indices(board)
        pfw, pfb = pawn_feature_indices(board)
        score = int(
            qnet.evaluate(
                _pad(fw),
                _pad(fb),
                _pad_pawn(pfw),
                _pad_pawn(pfb),
                np.array([turn], dtype=np.uint8),
            )[0]
        )
        assert score == ref["score"], f"{ref['epd']}: {score} != {ref['score']}"


def test_indices_match_c(reference):
    """The engine's piece/pawn index sets (frozen in the reference)
    must match the Python oracle (features.py).  Pinned via the
    reference, so this test needs no C library: it verifies the C1
    shared index generation against the legacy Python implementation
    for every cross-check position (order-insensitive: accumulator
    sums are commutative)."""
    for ref in reference["positions"]:
        board = _board_from_epd(ref["epd"])
        fw, fb = feature_indices(board)
        pfw, pfb = pawn_feature_indices(board)
        assert sorted(fw.tolist()) == sorted(ref["feat_w"]), ref["epd"]
        assert sorted(fb.tolist()) == sorted(ref["feat_b"]), ref["epd"]
        assert sorted(pfw.tolist()) == sorted(ref["pfeat_w"]), ref["epd"]
        assert sorted(pfb.tolist()) == sorted(ref["pfeat_b"]), ref["epd"]


def test_attack_indices_match_extraction(reference):
    """The attack index sets dumped by the engine (frozen in the
    reference) must be reproduced by the feature extraction library
    on the same positions.  Attacks have no Python oracle by design
    (C is the source of truth); this test pins the reference values
    and cross-checks the two C entry points (the engine's
    am_compute_attack_indices vs the ctypes library's)."""
    for ref in reference["positions"]:
        board = _board_from_epd(ref["epd"])
        turn = WHITE if " w " in ref["epd"] else BLACK
        rec = encode_board(board, turn, 0)
        assert rec.validate() == [], ref["epd"]
        ex = cfeatures.extract(rec.encode(), 1)
        assert ex.valid[0] == 1, ref["epd"]
        active_w = ex.afeat_w[0][ex.afeat_w[0] >= 0].tolist()
        active_b = ex.afeat_b[0][ex.afeat_b[0] >= 0].tolist()
        assert sorted(active_w) == sorted(ref["afeat_w"]), ref["epd"]
        assert sorted(active_b) == sorted(ref["afeat_b"]), ref["epd"]
