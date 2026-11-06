"""Tests for quantization to amy's network.quant format.

Shapes and scales come from the architecture constants (features.py,
quantize.py), not from hardcoded numbers, so the tests keep working
when the network structure changes for experiments.
"""

import struct

import numpy as np
import pytest

from amy_training_nnue import cfeatures
from amy_training_nnue.bpd import encode_board
from amy_training_nnue.features import (
    ACCUMULATOR_SIZE,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    N_ATTACK_FEATURES,
    N_FEATURES,
    N_PAWN_FEATURES,
    feature_indices,
    pawn_feature_indices,
)
from amy_training_nnue.quantize import (
    FIELD_ORDER,
    HEADER_SIZE,
    NETWORK_QUANT_SIZE,
    SCALE,
    SCALE_INPUT,
    SCALE_OUTPUT,
    WEIGHTS_HEADER_MAGIC,
    QuantizedNet,
    quantize_parameters,
    read_network_quant,
    weights_header,
    write_network_quant,
)
from tests.test_features import _board_from_epd

# Padding widths for the synthetic batches below (real dataset widths
# come from the C feature-extraction constants, see cfeatures.py).
MAX_FEATURES = 32
MAX_PAWN_FEATURES = 128
MAX_ATTACK_FEATURES = 32


def _feats(board):
    """Oracle piece/pawn indices plus an empty attack block (the
    synthetic tests below don't exercise attack rows; the cross-check
    suite covers those against C)."""
    w, b = feature_indices(board)
    pw, pb = pawn_feature_indices(board)
    fw = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    fb = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    pfw = np.full((1, MAX_PAWN_FEATURES), -1, dtype=np.int32)
    pfb = np.full((1, MAX_PAWN_FEATURES), -1, dtype=np.int32)
    afw = np.full((1, MAX_ATTACK_FEATURES), -1, dtype=np.int32)
    afb = np.full((1, MAX_ATTACK_FEATURES), -1, dtype=np.int32)
    fw[0, : len(w)] = w
    fb[0, : len(b)] = b
    pfw[0, : len(pw)] = pw
    pfb[0, : len(pb)] = pb
    return fw, fb, pfw, pfb, afw, afb


def test_header_matches_c_struct():
    """The header must be byte-identical to C's WeightsHeader
    (neural_net.c): u32 magic, 3 x u16 architecture sizes (accumulator,
    piece features, pawn features), 2 bytes of alignment padding, the
    u32 attack feature count (98304 doesn't fit in a u16), 2 x u16
    hidden layer sizes, then one u8 quantization code per piece field
    (1 = int8, 2 = int16, 3 = int32) in FIELD_ORDER order (the pawn
    and attack blocks have no code): 1 2 2 3 2 3 3 3.  28 bytes
    total, matching the C struct layout exactly."""
    header = weights_header()
    assert len(header) == HEADER_SIZE == 28
    assert header == struct.pack(
        "<IHHH2xIHH8B",
        WEIGHTS_HEADER_MAGIC,
        ACCUMULATOR_SIZE,
        N_FEATURES,
        N_PAWN_FEATURES,
        N_ATTACK_FEATURES,
        HIDDEN_SIZE_1,
        HIDDEN_SIZE_2,
        1,
        2,
        2,
        3,
        2,
        3,
        3,
        3,
    )


def _zero_net():
    """Zeroed quantized field arrays, shaped per FIELD_ORDER."""
    return {name: np.zeros(shape, dtype=dtype) for name, dtype, shape in FIELD_ORDER}


def test_hand_computed_evaluation():
    """All-zero weights with a known activation chain: every position
    must evaluate to exactly SCALE_OUTPUT mp.

    Chain: acc = 0 (clamped from 0); h1 pre = bias = 1.0 (i.e.
    SCALE_H1 in 64*8192 units) -> act (SCALE_H1 + 4095) / 8192
    = SCALE_INPUT; h2 pre = SCALE_INPUT * SCALE (w=1.0) -> act SCALE
    (the >= clamp boundary); out = SCALE * SCALE (w=1.0) -> score =
    (SCALE * SCALE_OUTPUT + 4095) / SCALE = SCALE_OUTPUT.
    """
    q = _zero_net()
    q["hidden_layer_1_bias_q"][0] = SCALE_INPUT * SCALE  # 1.0
    q["hidden_layer_2_weights_q"][0, 0] = SCALE  # 1.0
    q["output_layer_weights_q"][0] = SCALE  # 1.0
    net = QuantizedNet.from_arrays(q)
    board = _board_from_epd("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -")
    fw, fb, pfw, pfb, afw, afb = _feats(board)
    score = net.evaluate(fw, fb, pfw, pfb, afw, afb, np.array([0]))
    assert score[0] == SCALE_OUTPUT
    # independent of the board
    board2 = _board_from_epd("4k3/8/8/8/8/8/8/R3K3 w - -")
    fw2, fb2, pfw2, pfb2, afw2, afb2 = _feats(board2)
    assert (
        net.evaluate(fw2, fb2, pfw2, pfb2, afw2, afb2, np.array([0]))[0] == SCALE_OUTPUT
    )


def test_hand_computed_accumulator():
    """Single active piece feature with int8 weight 100 per neuron,
    single active pawn pair row with weight 10, single active attack
    row with weight 7, bias 5 (all scale 64): accumulator = 5 + 100 +
    10 + 7 = 122.  122 exceeds the [0, 64] evaluation clamp, but
    clamping only happens in evaluate(), so the raw accumulator must
    report 122."""
    q = _zero_net()
    q["input_layer_weights_q"][7] = 100  # some piece feature
    q["input_pawn_weights_q"][27] = 10  # some pawn pair row
    q["input_attack_weights_q"][5] = 7  # some attack row
    q["input_layer_bias_q"][:] = 5
    net = QuantizedNet.from_arrays(q)
    fw = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    fb = np.full((1, MAX_FEATURES), -1, dtype=np.int32)
    pfw = np.full((1, MAX_PAWN_FEATURES), -1, dtype=np.int32)
    pfb = np.full((1, MAX_PAWN_FEATURES), -1, dtype=np.int32)
    afw = np.full((1, MAX_ATTACK_FEATURES), -1, dtype=np.int32)
    afb = np.full((1, MAX_ATTACK_FEATURES), -1, dtype=np.int32)
    fw[0, 0] = 7  # feature active on the white accumulator only
    pfw[0, 0] = 27  # pawn pair active on the white accumulator only
    afw[0, 0] = 5  # attack row active on the white accumulator only
    acc = net.accumulator(fw, pfw, afw)
    assert (acc[0] == 5 + 100 + 10 + 7).all()
    acc_b = net.accumulator(fb, pfb, afb)
    assert (acc_b[0] == 5).all()


def test_write_read_roundtrip(tmp_path):
    q = _zero_net()
    q["input_layer_bias_q"] = np.arange(ACCUMULATOR_SIZE, dtype=np.int16)
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
        8: ("pawn features size", "sizes"),
        12: ("attack features size", "sizes"),
        20: ("quant codes", "quantization"),  # codes start after the sizes
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
        "input_w": rng.normal(0, 0.02, (N_FEATURES, ACCUMULATOR_SIZE)),
        "input_b": rng.normal(0, 0.02, ACCUMULATOR_SIZE),
        "pawn_w": rng.normal(0, 0.02, (N_PAWN_FEATURES, ACCUMULATOR_SIZE)),
        "attack_w": rng.normal(0, 0.02, (N_ATTACK_FEATURES, ACCUMULATOR_SIZE)),
        "h1_w": rng.normal(0, 0.06, (2 * ACCUMULATOR_SIZE, HIDDEN_SIZE_1)),
        "h1_b": rng.normal(0, 0.1, HIDDEN_SIZE_1),
        "h2_w": rng.normal(0, 0.25, (HIDDEN_SIZE_1, HIDDEN_SIZE_2)),
        "h2_b": rng.normal(0, 0.1, HIDDEN_SIZE_2),
        "out_w": rng.normal(0, 0.05, HIDDEN_SIZE_2),
        "out_b": rng.normal(0, 0.05, 1),
    }
    q = quantize_parameters(float_params)
    assert (
        q["input_layer_weights_q"]
        == np.round(float_params["input_w"] * SCALE_INPUT).astype(np.int8)
    ).all()
    assert (
        q["input_pawn_weights_q"]
        == np.round(float_params["pawn_w"] * SCALE_INPUT).astype(np.int8)
    ).all()
    assert (
        q["input_attack_weights_q"]
        == np.round(float_params["attack_w"] * SCALE_INPUT).astype(np.int8)
    ).all()
    assert (
        q["input_layer_bias_q"]
        == np.round(float_params["input_b"] * SCALE_INPUT).astype(np.int16)
    ).all()
    assert (
        q["hidden_layer_1_weights_q"]
        == np.round(float_params["h1_w"].T * SCALE).astype(np.int16)
    ).all()
    assert (
        q["hidden_layer_1_bias_q"]
        == np.round(float_params["h1_b"] * SCALE_INPUT * SCALE).astype(np.int32)
    ).all()
    assert (
        q["hidden_layer_2_weights_q"]
        == np.round(float_params["h2_w"].T * SCALE).astype(np.int16)
    ).all()
    assert (
        q["hidden_layer_2_bias_q"]
        == np.round(float_params["h2_b"] * SCALE * SCALE_INPUT).astype(np.int32)
    ).all()
    assert (
        q["output_layer_weights_q"]
        == np.round(float_params["out_w"] * SCALE).astype(np.int32)
    ).all()
    assert (
        q["output_layer_bias_q"]
        == np.round(float_params["out_b"] * SCALE * SCALE).astype(np.int32)
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
        fw, fb, pfw, pfb, _, _ = _feats(board)
        turn = 0 if epd.split()[1] == "w" else 1
        # Real attack indices from the C extraction library, so the
        # comparison also covers the attack block (the same indices go
        # into the float and the quantized net).
        rec = encode_board(board, turn, 0)
        assert rec.validate() == []
        ex = cfeatures.extract(rec.encode(), 1)
        assert ex.valid[0] == 1
        f = (
            model(
                mx.array(fw),
                mx.array(fb),
                mx.array(pfw),
                mx.array(pfb),
                mx.array(ex.afeat_w),
                mx.array(ex.afeat_b),
                mx.array([turn]),
            ).item()
            * 10000
        )
        i = float(
            net.evaluate(fw, fb, pfw, pfb, ex.afeat_w, ex.afeat_b, np.array([turn]))[0]
        )
        assert abs(f - i) < 200, f"{epd}: float {f:.1f} vs int {i:.1f}"
        total += abs(f - i)
    assert total / 4 < 100
