"""Quantize a trained float network to amy's network.quant format and
evaluate it with the exact integer arithmetic of neural_net.c.

network.quant is a 20-byte little-endian header (the WeightsHeader
struct of neural_net.c, see weights_header()) followed by, in order:

    input_layer_weights_q  int8   [22528][ACCUMULATOR_SIZE]
    input_layer_bias_q     int16  [ACCUMULATOR_SIZE]
    hidden_layer_1_weights_q int8 [HIDDEN_SIZE_1][512]
    hidden_layer_1_bias_q  int16  [HIDDEN_SIZE_1]
    hidden_layer_2_weights_q int16 [HIDDEN_SIZE_2][HIDDEN_SIZE_1]
    hidden_layer_2_bias_q  int32  [HIDDEN_SIZE_2]
    output_layer_weights_q int32  [HIDDEN_SIZE_2]
    output_layer_bias_q    int32  [1]

Quantization convention (float v -> fixed point):

    input weights:  int8(v * 64); the accumulator is kept in 1/64 units
    input bias:     int16(v * 64)
    hidden 1:       int8(v * 64) weights, int16(v * 64 * 64) bias;
                    sum in 64*64 units, +31 then /64 -> acts in [0, 64]
    hidden 2:       int16(v * 8192) weights, int32(v * 8192 * 64) bias;
                    sum in 8192*64 units, +31 then /64 -> acts in [0, 8192]
    output:         int32(v * 8192) weights, int32(v * 8192 * 8192) bias

The int8 layers cover float magnitude 2.0 at scale 64 (weights are kept
within [-1, 1] during training, so no clipping in practice); the int16
hidden 1 bias at scale 4096 covers +-8.0.  Out-of-range values are
clipped (with a warning) so the file stays loadable by amy.

The header (little-endian, 20 bytes, must match neural_net.c's
WeightsHeader byte-for-byte):

    u32 magic 0x57594D41 (file bytes read as "AMYW")
    u16 accumulator size, u16 number of input features (22528),
    u16 hidden layer 1 size, u16 hidden layer 2 size
    8 x u8 quantization codes (1 = int8, 2 = int16, 3 = int32), one per
    field, in the same order as the fields above
"""

from __future__ import annotations

import json
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import ACCUMULATOR_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, N_FEATURES

SCALE = 8192
SCALE_INPUT = 64  # input/hidden-1 scale; accumulator in 1/64 units
SCALE_INPUT_HALF = SCALE_INPUT // 2 - 1  # 31, as in neural_net.c
SCALE_HALF = SCALE // 2 - 1  # 4095, as in neural_net.c
SCALE_OUTPUT = 10000  # score_mp = round(output * SCALE_OUTPUT)

FIELD_ORDER = [
    # (name, numpy dtype, shape)
    ("input_layer_weights_q", np.int8, (N_FEATURES, ACCUMULATOR_SIZE)),
    ("input_layer_bias_q", np.int16, (ACCUMULATOR_SIZE,)),
    ("hidden_layer_1_weights_q", np.int8, (HIDDEN_SIZE_1, 2 * ACCUMULATOR_SIZE)),
    ("hidden_layer_1_bias_q", np.int16, (HIDDEN_SIZE_1,)),
    ("hidden_layer_2_weights_q", np.int16, (HIDDEN_SIZE_2, HIDDEN_SIZE_1)),
    ("hidden_layer_2_bias_q", np.int32, (HIDDEN_SIZE_2,)),
    ("output_layer_weights_q", np.int32, (HIDDEN_SIZE_2,)),
    ("output_layer_bias_q", np.int32, (1,)),
]

# WeightsHeader of neural_net.c (little-endian, 20 bytes, no padding).
WEIGHTS_HEADER_MAGIC = 0x57594D41  # file bytes read as "AMYW"
QUANT_INT8 = 1
QUANT_INT16 = 2
QUANT_INT32 = 3
# u32 magic, 4 x u16 sizes (accumulator, num features, h1, h2),
# 8 x u8 quant codes.
HEADER_FORMAT = "<IHHHH8B"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

_QUANT_CODES = {
    "int8": QUANT_INT8,
    "int16": QUANT_INT16,
    "int32": QUANT_INT32,
}


def _header_quant_codes() -> list[int]:
    """The 8 header quantization codes, one per FIELD_ORDER entry."""
    return [_QUANT_CODES[np.dtype(dtype).name] for _, dtype, _ in FIELD_ORDER]


def weights_header() -> bytes:
    """The 20-byte WeightsHeader of neural_net.c for this architecture."""
    return struct.pack(
        HEADER_FORMAT,
        WEIGHTS_HEADER_MAGIC,
        ACCUMULATOR_SIZE,
        N_FEATURES,
        HIDDEN_SIZE_1,
        HIDDEN_SIZE_2,
        *_header_quant_codes(),
    )


# Byte size of the quantized field arrays alone (without the header).
PAYLOAD_SIZE = sum(
    np.dtype(dtype).itemsize * int(np.prod(shape)) for _, dtype, shape in FIELD_ORDER
)

# Total network.quant file size: header + fields.
NETWORK_QUANT_SIZE = HEADER_SIZE + PAYLOAD_SIZE


def _quantize(value: np.ndarray, scale: float, dtype: type) -> np.ndarray:
    q = np.round(value.astype(np.float64) * scale)
    info = np.iinfo(np.dtype(dtype))
    if np.any(q < info.min) or np.any(q > info.max):
        lo, hi = info.min, info.max
        n = int(np.sum((q < lo) | (q > hi)))
        warnings.warn(
            f"{n} value(s) outside int range [{lo}, {hi}], clipped "
            f"(max |q| = {np.max(np.abs(q)):.1f}); retrain with smaller weights "
            "or check the target scale",
            stacklevel=2,
        )
        q = np.clip(q, lo, hi)
    return q.astype(dtype)


def quantize_parameters(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map float model parameters to the quantized field arrays.

    The model stores h1_w in matmul layout (2*ACCUMULATOR_SIZE,
    HIDDEN_SIZE_1) and h2_w as (HIDDEN_SIZE_1, HIDDEN_SIZE_2) with
    h2_w[j, i] = weight from j to i; the file layout is the C layout
    [new][old], i.e. the transposes.
    """
    return {
        "input_layer_weights_q": _quantize(params["input_w"], SCALE_INPUT, np.int8),
        "input_layer_bias_q": _quantize(params["input_b"], SCALE_INPUT, np.int16),
        "hidden_layer_1_weights_q": _quantize(params["h1_w"].T, SCALE_INPUT, np.int8),
        "hidden_layer_1_bias_q": _quantize(
            params["h1_b"], SCALE_INPUT * SCALE_INPUT, np.int16
        ),
        "hidden_layer_2_weights_q": _quantize(params["h2_w"].T, SCALE, np.int16),
        "hidden_layer_2_bias_q": _quantize(
            params["h2_b"], SCALE_INPUT * SCALE, np.int32
        ),
        "output_layer_weights_q": _quantize(params["out_w"], SCALE, np.int32),
        "output_layer_bias_q": _quantize(params["out_b"], SCALE * SCALE, np.int32),
    }


def write_network_quant(path: str | Path, quantized: dict[str, np.ndarray]) -> int:
    """Write the quantized network: WeightsHeader followed by amy's raw
    field layout.  Returns the total byte count."""
    path = Path(path)
    total = HEADER_SIZE
    with open(path, "wb") as f:
        f.write(weights_header())
        for name, dtype, shape in FIELD_ORDER:
            array = quantized[name].astype(dtype).reshape(shape)
            f.write(array.tobytes())
            total += array.nbytes

    return total


def read_network_quant(path: str | Path) -> dict[str, np.ndarray]:
    """Read network.quant back into a dict of named numpy arrays."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) != NETWORK_QUANT_SIZE:
        raise ValueError(
            f"{path}: size {len(data)} != expected {NETWORK_QUANT_SIZE} bytes"
        )
    magic, acc_size, num_features, h1_size, h2_size, *codes = struct.unpack_from(
        HEADER_FORMAT, data
    )
    if magic != WEIGHTS_HEADER_MAGIC:
        raise ValueError(
            f"{path}: bad magic {magic:#010x}, expected {WEIGHTS_HEADER_MAGIC:#010x}"
        )
    if (acc_size, num_features, h1_size, h2_size) != (
        ACCUMULATOR_SIZE,
        N_FEATURES,
        HIDDEN_SIZE_1,
        HIDDEN_SIZE_2,
    ):
        raise ValueError(
            f"{path}: header sizes {(acc_size, num_features, h1_size, h2_size)} != "
            f"({ACCUMULATOR_SIZE}, {N_FEATURES}, {HIDDEN_SIZE_1}, {HIDDEN_SIZE_2}) "
            "(stale network.quant from an older architecture?)"
        )
    expected_codes = _header_quant_codes()
    if codes != expected_codes:
        raise ValueError(
            f"{path}: header quantization codes {codes} != {expected_codes}"
        )
    out = {}
    offset = HEADER_SIZE
    for name, dtype, shape in FIELD_ORDER:
        count = int(np.prod(shape))
        out[name] = np.frombuffer(
            data, dtype=dtype, count=count, offset=offset
        ).reshape(shape)
        offset += np.dtype(dtype).itemsize * count
    return out


def _trunc_div(n: np.ndarray, d: int) -> np.ndarray:
    """Integer division truncating toward zero, like C for all signs."""
    return np.sign(n) * (np.abs(n) // d)


@dataclass
class QuantizedNet:
    """Integer simulation of amy's quantized network (neural_net.c)."""

    input_w: np.ndarray  # int8 (N_FEATURES, ACCUMULATOR_SIZE), scale 64
    input_b: np.ndarray  # int16 (ACCUMULATOR_SIZE,), scale 64
    h1_w: np.ndarray  # int8 (HIDDEN_SIZE_1, 2*ACC), scale 64
    h1_b: np.ndarray  # int16 (HIDDEN_SIZE_1,), scale 64*64
    h2_w: np.ndarray  # int16 (HIDDEN_SIZE_2, HIDDEN_SIZE_1), scale 8192
    h2_b: np.ndarray  # int32 (HIDDEN_SIZE_2,), scale 8192*64
    out_w: np.ndarray  # int32 (HIDDEN_SIZE_2,)
    out_b: np.ndarray  # int32 (1,)

    @staticmethod
    def from_arrays(quantized: dict[str, np.ndarray]) -> "QuantizedNet":
        shapes = {name: shape for name, _, shape in FIELD_ORDER}

        def shaped(name: str) -> np.ndarray:
            return np.asarray(quantized[name]).reshape(shapes[name])

        return QuantizedNet(
            input_w=shaped("input_layer_weights_q"),
            input_b=shaped("input_layer_bias_q"),
            h1_w=shaped("hidden_layer_1_weights_q"),
            h1_b=shaped("hidden_layer_1_bias_q"),
            h2_w=shaped("hidden_layer_2_weights_q"),
            h2_b=shaped("hidden_layer_2_bias_q"),
            out_w=shaped("output_layer_weights_q"),
            out_b=shaped("output_layer_bias_q"),
        )

    @staticmethod
    def from_file(path: str | Path) -> "QuantizedNet":
        return QuantizedNet.from_arrays(read_network_quant(path))

    # -- integer pipeline, matching neural_net.c step by step ----------

    def accumulator(self, feat: np.ndarray) -> np.ndarray:
        """(B, K) int32 indices (padded -1) -> raw int16 accumulator (B, ACCUMULATOR_SIZE).

        Like amy: the accumulator is maintained in int16 with wrap-around
        in units of 1/SCALE_INPUT, starting from the int16 input bias
        (same scale).  The clamping to [0, SCALE_INPUT] happens only at
        evaluation time (see evaluate()).
        """
        acc = np.zeros((feat.shape[0], ACCUMULATOR_SIZE), dtype=np.int32)
        acc += self.input_b.astype(np.int32)
        for k in range(feat.shape[1]):
            idx = feat[:, k]
            valid = idx >= 0
            if not np.any(valid):
                continue
            acc[valid] = (
                (acc[valid] + self.input_w[idx[valid]].astype(np.int32))
                .astype(np.int16)
                .astype(np.int32)
            )
        return acc

    def evaluate(
        self, feat_w: np.ndarray, feat_b: np.ndarray, turn: np.ndarray
    ) -> np.ndarray:
        """Scores in millipawns, side to move. turn: (B,) 0/1.

        C computes hidden_layer_i_q[i] = bias_i + sum_j acc[j] * W[i][j]
        with W in [new][old] file layout; in matrix form that is
        (W @ z.T).T.
        """
        acc_w = self.accumulator(feat_w)
        acc_b = self.accumulator(feat_b)
        n = feat_w.shape[0]
        acc = np.empty((n, 2 * ACCUMULATOR_SIZE), dtype=np.int32)
        wm = turn == 0
        acc[wm] = np.concatenate([acc_w[wm], acc_b[wm]], axis=1)
        acc[~wm] = np.concatenate([acc_b[~wm], acc_w[~wm]], axis=1)
        # neural_net.c clamps the local 2*ACCUMULATOR_SIZE-dim accumulator copy
        # stored accumulators stay raw int16), in 1/64 units.
        acc = np.clip(acc, 0, SCALE_INPUT)

        h1 = (self.h1_w @ acc.T).T + self.h1_b  # int32 (n, HIDDEN_SIZE_1), 64*64 units
        h1 = np.where(h1 <= 0, 0, h1)
        h1 = np.where(
            h1 > SCALE_INPUT * SCALE_INPUT,
            SCALE_INPUT,
            (h1 + SCALE_INPUT_HALF) // SCALE_INPUT,
        )
        h1 = h1.astype(np.int32)

        h2 = (self.h2_w @ h1.T).T + self.h2_b  # int32 (n, HIDDEN_SIZE_2), 8192*64 units
        h2 = np.where(h2 <= 0, 0, h2)
        # neural_net.c clamps with >= here (vs > for hidden layer 1).
        h2 = np.where(
            h2 >= SCALE_INPUT * SCALE,
            SCALE,
            (h2 + SCALE_INPUT_HALF) // SCALE_INPUT,
        )
        h2 = h2.astype(np.int32)

        out_q = h2 @ self.out_w + self.out_b[0]
        # C integer division truncates toward zero (not floor).
        r = _trunc_div(out_q + SCALE_HALF, SCALE)
        score = _trunc_div(r * SCALE_OUTPUT + SCALE_HALF, SCALE)
        return score.astype(np.int32)


def checkpoint_meta(checkpoint_dir: str | Path) -> dict:
    meta_file = Path(checkpoint_dir) / "meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())
    return {}
