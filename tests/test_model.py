"""Tests for the MLX model."""

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from amy_training_nnue.dataset import MAX_FEATURES  # noqa: E402
from amy_training_nnue.features import FLIP, feature_indices  # noqa: E402
from amy_training_nnue.model import WEIGHT_NAMES, NnueNet  # noqa: E402
from tests.test_features import _board_from_epd  # noqa: E402


def _feats(board, batch=1):
    w, b = feature_indices(board)
    fw = np.full((batch, MAX_FEATURES), -1, dtype=np.int32)
    fb = np.full((batch, MAX_FEATURES), -1, dtype=np.int32)
    fw[:, : len(w)] = w
    fb[:, : len(b)] = b
    return fw, fb


def test_forward_shape():
    model = NnueNet(seed=1)
    board = _board_from_epd("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -")
    fw, fb = _feats(board)
    out = model(mx.array(fw), mx.array(fb), mx.array([0]))
    assert out.shape == (1,)
    assert np.isfinite(out.item())


def test_deterministic():
    m1 = NnueNet(seed=7)
    m2 = NnueNet(seed=7)
    board = _board_from_epd("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -")
    fw, fb = _feats(board)
    o1 = np.array(m1(mx.array(fw), mx.array(fb), mx.array([0])))
    o2 = np.array(m2(mx.array(fw), mx.array(fb), mx.array([0])))
    assert o1 == o2


def test_turn_symmetry():
    """Evaluating a position with white to move must equal evaluating
    the vertically flipped position with black to move: the features
    are identical and the STM/opponent ordering is the same."""
    model = NnueNet(seed=3)
    board = _board_from_epd(
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -"
    )
    fw, fb = _feats(board)
    flipped = np.zeros(64, dtype=np.int8)
    for sq in range(64):
        flipped[sq] = -board[sq ^ FLIP]
    fwf, fbf = _feats(flipped)
    s_w = model(mx.array(fw), mx.array(fb), mx.array([0])).item()
    s_b = model(mx.array(fwf), mx.array(fbf), mx.array([1])).item()
    assert s_w == pytest.approx(s_b, rel=1e-6, abs=1e-6)


def test_batch_matches_single():
    model = NnueNet(seed=5)
    specs = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", 0),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -", 0),
        ("r3k2r/8/8/8/8/8/8/4K3 b - -", 1),
    ]
    fws, fbs, turns = [], [], []
    for epd, turn in specs:
        fw, fb = _feats(_board_from_epd(epd))
        fws.append(fw)
        fbs.append(fb)
        turns.append(turn)
    batch = np.array(
        model(
            mx.array(np.concatenate(fws)),
            mx.array(np.concatenate(fbs)),
            mx.array(turns),
        )
    )
    for i, (epd, turn) in enumerate(specs):
        f1w, f1b = _feats(_board_from_epd(epd))
        single = model(mx.array(f1w), mx.array(f1b), mx.array([turn])).item()
        assert batch[i] == pytest.approx(single, rel=1e-6, abs=1e-6)


def test_clip_weights_only_weights():
    model = NnueNet(seed=1)
    for name in WEIGHT_NAMES:
        setattr(model, name, mx.full(getattr(model, name).shape, 5.0))
    for name, v in (("input_b", 9.0), ("h1_b", -9.0), ("h2_b", 8.0), ("out_b", -8.0)):
        setattr(model, name, mx.full(getattr(model, name).shape, v))
    model.clip_weights(1.0)
    for name in WEIGHT_NAMES:
        w = np.array(getattr(model, name))
        assert np.all((w >= -1.0) & (w <= 1.0))
    # biases are untouched
    for name, v in (("input_b", 9.0), ("h1_b", -9.0), ("h2_b", 8.0), ("out_b", -8.0)):
        assert np.array(getattr(model, name)) == pytest.approx(v)


def test_save_load(tmp_path):
    model = NnueNet(seed=11)
    model.save(tmp_path)
    loaded = NnueNet.load(tmp_path)
    board = _board_from_epd("4k3/8/8/8/8/8/8/R3K3 w - -")
    fw, fb = _feats(board)
    o1 = np.array(model(mx.array(fw), mx.array(fb), mx.array([0])))
    o2 = np.array(loaded(mx.array(fw), mx.array(fb), mx.array([0])))
    assert o1 == o2
