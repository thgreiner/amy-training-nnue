"""Tests for quantization to amy's network.quant format."""

import numpy as np
import pytest

from amy_training_nnue.dataset import MAX_FEATURES
from amy_training_nnue.features import N_FEATURES, feature_indices
from amy_training_nnue.quantize import (
    HEADER_SIZE,
    NETWORK_QUANT_SIZE,
    PAYLOAD_SIZE,
    QuantizedNet,
    quantize_parameters,
    read_network_quant,
    weights_header,
    write_network_quant,
)
from tests.test_features import _board_from_epd


def _feats(board):
    w, b = feature_indices(board)
    fw = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    fb = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    fw[0, : len(w)] = w
    fb[0, : len(b)] = b
    return fw, fb


def test_network_size():
    # 20-byte header + 45056*128*1 + 128*2 + 32*256*1 + 32*2 + 32*32*2
    # + 32*4 + 32*4 + 4
    assert PAYLOAD_SIZE == 5_777_988
    assert HEADER_SIZE == 20
    assert NETWORK_QUANT_SIZE == 5_778_008


def test_header_matches_c_struct():
    """The header must be byte-identical to C's WeightsHeader
    (neural_net.c): magic, sizes 128/45056/32/32, quant codes
    1 2 1 2 2 3 3 3."""
    header = weights_header()
    assert len(header) == HEADER_SIZE == 20
    assert header.hex() == "414d5957800000b0200020000102010202030303"


def _zero_net():
    q = {
        "input_layer_weights_q": np.zeros((N_FEATURES, 128), dtype=np.int8),
        "input_layer_bias_q": np.zeros(128, dtype=np.int16),
        "hidden_layer_1_weights_q": np.zeros((32, 256), dtype=np.int8),
        "hidden_layer_1_bias_q": np.zeros(32, dtype=np.int16),
        "hidden_layer_2_weights_q": np.zeros((32, 32), dtype=np.int16),
        "hidden_layer_2_bias_q": np.zeros(32, dtype=np.int32),
        "output_layer_weights_q": np.zeros(32, dtype=np.int32),
        "output_layer_bias_q": np.zeros(1, dtype=np.int32),
    }
    return q


def test_hand_computed_evaluation():
    """All-zero weights with a known activation chain: every position
    must evaluate to exactly 10000 mp.

    Chain: acc = 0 (clamped from 0); h1 pre = bias = 4096 (1.0 in
    64*64 units) -> act (4096 + 31) / 64 = 64; h2 pre = 64 * 8192
    (w=1.0) = 524288 -> act 8192 (the >= clamp boundary); out =
    8192 * 8192 (w=1.0) -> score = (8192 * 10000 + 4095) / 8192 = 10000.
    """
    q = _zero_net()
    q["hidden_layer_1_bias_q"][0] = 4096
    q["hidden_layer_2_weights_q"][0, 0] = 8192  # 1.0
    q["output_layer_weights_q"][0] = 8192  # 1.0
    net = QuantizedNet.from_arrays(q)
    board = _board_from_epd("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -")
    fw, fb = _feats(board)
    score = net.evaluate(fw, fb, np.array([0]))
    assert score[0] == 10000
    # independent of the board
    board2 = _board_from_epd("4k3/8/8/8/8/8/8/R3K3 w - -")
    fw2, fb2 = _feats(board2)
    assert net.evaluate(fw2, fb2, np.array([0]))[0] == 10000


def test_hand_computed_accumulator():
    """Single active feature with int8 weight 100 per neuron, bias 5
    (both scale 64): accumulator = 5 + 100 = 105.  105 exceeds the
    [0, 64] evaluation clamp, but clamping only happens in
    evaluate(), so the raw accumulator must report 105."""
    q = _zero_net()
    q["input_layer_weights_q"][7] = 100  # some feature
    q["input_layer_bias_q"][:] = 5
    net = QuantizedNet.from_arrays(q)
    fw = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    fb = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    fw[0, 0] = 7  # feature active on the white accumulator only
    acc = net.accumulator(fw)
    assert (acc[0] == 5 + 100).all()
    acc_b = net.accumulator(fb)
    assert (acc_b[0] == 5).all()


def test_write_read_roundtrip(tmp_path):
    q = _zero_net()
    q["input_layer_bias_q"] = np.arange(128, dtype=np.int16)
    q["output_layer_bias_q"] = np.array([12345], dtype=np.int32)
    path = tmp_path / "network.quant"
    n = write_network_quant(path, q)
    assert n == NETWORK_QUANT_SIZE
    back = read_network_quant(path)
    for key in q:
        assert (back[key] == q[key]).all(), key


def test_read_rejects_bad_size(tmp_path):
    path = tmp_path / "bad.quant"
    path.write_bytes(b"\x00" * 100)
    with pytest.raises(ValueError, match="size"):
        read_network_quant(path)


def test_read_rejects_bad_header(tmp_path):
    q = _zero_net()
    path = tmp_path / "badheader.quant"
    write_network_quant(path, q)
    for offset, (what, match) in {
        0: ("magic", "magic"),
        4: ("accumulator size", "sizes"),
        12: ("quant codes", "quantization"),  # codes start after 4 x u16 sizes
    }.items():
        data = bytearray(path.read_bytes())
        data[offset] ^= 0xFF
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match=match):
            read_network_quant(path)
        # restore
        data[offset] ^= 0xFF
        path.write_bytes(bytes(data))
    read_network_quant(path)  # uncorrupted file still reads fine


def test_quantize_matches_int_arrays():
    """Quantizing float values v gives round(v * scale) ints."""
    rng = np.random.default_rng(0)
    float_params = {
        "input_w": rng.normal(0, 0.02, (N_FEATURES, 128)),
        "input_b": rng.normal(0, 0.02, 128),
        "h1_w": rng.normal(0, 0.06, (256, 32)),
        "h1_b": rng.normal(0, 0.1, 32),
        "h2_w": rng.normal(0, 0.25, (32, 32)),
        "h2_b": rng.normal(0, 0.1, 32),
        "out_w": rng.normal(0, 0.05, 32),
        "out_b": rng.normal(0, 0.05, 1),
    }
    q = quantize_parameters(float_params)
    assert (
        q["input_layer_weights_q"]
        == np.round(float_params["input_w"] * 64).astype(np.int8)
    ).all()
    assert (
        q["input_layer_bias_q"]
        == np.round(float_params["input_b"] * 64).astype(np.int16)
    ).all()
    assert (
        q["hidden_layer_1_weights_q"]
        == np.round(float_params["h1_w"].T * 64).astype(np.int8)
    ).all()
    assert (
        q["hidden_layer_1_bias_q"]
        == np.round(float_params["h1_b"] * 64 * 64).astype(np.int16)
    ).all()
    assert (
        q["hidden_layer_2_bias_q"]
        == np.round(float_params["h2_b"] * 8192 * 64).astype(np.int32)
    ).all()
    assert (
        q["output_layer_bias_q"]
        == np.round(float_params["out_b"] * 8192 * 8192).astype(np.int32)
    ).all()


def test_float_vs_int_close():
    """A randomly initialized float net and its quantization must
    agree within quantization error on real positions."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    from amy_training_nnue.model import NnueNet

    model = NnueNet(seed=42)
    params = {k: np.array(v) for k, v in model.parameters.items()}
    q = quantize_parameters(params)
    net = QuantizedNet.from_arrays(q)

    total = 0
    for epd in (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -",
        "r3k2r/8/8/8/8/8/8/4K3 b - -",
        "4k3/3P4/8/8/8/8/8/4K3 w - -",
    ):
        board = _board_from_epd(epd)
        fw, fb = _feats(board)
        turn = 0 if epd.split()[1] == "w" else 1
        f = model(mx.array(fw), mx.array(fb), mx.array([turn])).item() * 10000
        i = float(net.evaluate(fw, fb, np.array([turn]))[0])
        assert abs(f - i) < 200, f"{epd}: float {f:.1f} vs int {i:.1f}"
        total += abs(f - i)
    assert total / 4 < 100
