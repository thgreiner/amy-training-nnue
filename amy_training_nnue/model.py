"""The NNUE model in floating point, implemented with MLX.

Architecture (float counterpart of amy's quantized network):

    input (shared, both sides):
        acc_side = clip(input_b + sum_{f in side} input_w[f], 0, 1)
        x = [acc_stm, acc_opponent]              # 2*ACCUMULATOR_SIZE values in [0, 1]
    hidden1 = clip(h1_w @ x + h1_b, 0, 1)        # HIDDEN_SIZE_1
    hidden2 = clip(h2_w @ hidden1 + h2_b, 0, 1)  # HIDDEN_SIZE_2
    output  = h2 @ out_w + out_b                 # 1, in units of
                                                    score_millipawns / 10000
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .features import ACCUMULATOR_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, N_FEATURES

# The scale amy uses to convert the raw network output to millipawns
# (SCALE_OUTPUT in neural_net.c): score_mp = round(output * 10000).
DEFAULT_TARGET_SCALE = 10000.0

# Weight tensors kept within [-1, 1] during training (biases are exempt).
WEIGHT_NAMES = ("input_w", "h1_w", "h2_w", "out_w")


class NnueNet(nn.Module):
    def __init__(
        self,
        init_input_std: float = 0.02,
        init_hidden_gains: tuple[float, float] = (1.0, 1.0),
        seed: int = 0,
    ):
        super().__init__()
        mx.random.seed(seed)
        fan_in = ACCUMULATOR_SIZE
        self.input_w = mx.random.normal((N_FEATURES, ACCUMULATOR_SIZE)) * init_input_std
        self.input_b = mx.zeros((ACCUMULATOR_SIZE,))
        # Kaiming (uniform-equivalent via normal) init for the clamped
        # ReLU layers: variance 2 / fan_in.
        self.h1_w = mx.random.normal((2 * ACCUMULATOR_SIZE, HIDDEN_SIZE_1)) * (
            init_hidden_gains[0] * (2.0 / fan_in) ** 0.5
        )
        self.h1_b = mx.zeros((HIDDEN_SIZE_1,))
        self.h2_w = mx.random.normal((HIDDEN_SIZE_1, HIDDEN_SIZE_2)) * (
            init_hidden_gains[1] * (2.0 / HIDDEN_SIZE_1) ** 0.5
        )
        self.h2_b = mx.zeros((HIDDEN_SIZE_2,))
        self.out_w = mx.random.normal((HIDDEN_SIZE_2,)) * 0.05
        self.out_b = mx.zeros((1,))

    def accumulator(self, feat: mx.array) -> mx.array:
        """(B, K) int32 feature indices (padded with -1) -> (B, ACCUMULATOR_SIZE)."""
        valid = feat >= 0
        idx = mx.where(valid, feat, 0)
        gathered = mx.take(self.input_w, idx, axis=0)  # (B, K, 256)
        gathered = mx.where(valid[:, :, None], gathered, 0.0)
        acc = gathered.sum(axis=1) + self.input_b
        return mx.clip(acc, 0.0, 1.0)

    def __call__(self, feat_w: mx.array, feat_b: mx.array, turn: mx.array) -> mx.array:
        acc_w = self.accumulator(feat_w)
        acc_b = self.accumulator(feat_b)
        white_to_move = turn == 0
        x = mx.where(
            white_to_move[:, None],
            mx.concat([acc_w, acc_b], axis=1),
            mx.concat([acc_b, acc_w], axis=1),
        )
        h1 = mx.clip(x @ self.h1_w + self.h1_b, 0.0, 1.0)
        h2 = mx.clip(h1 @ self.h2_w + self.h2_b, 0.0, 1.0)
        return h2 @ self.out_w + self.out_b[0]

    def clip_weights(self, bound: float = 1.0) -> None:
        """Clip the weight tensors (not the biases) to [-bound, bound]."""
        for name in WEIGHT_NAMES:
            setattr(self, name, mx.clip(getattr(self, name), -bound, bound))

    # ------------------------------------------------------------------
    # (de)serialization
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> dict:
        return {
            "input_w": self.input_w,
            "input_b": self.input_b,
            "h1_w": self.h1_w,
            "h1_b": self.h1_b,
            "h2_w": self.h2_w,
            "h2_b": self.h2_b,
            "out_w": self.out_w,
            "out_b": self.out_b,
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        import numpy as np

        # Materialize the (possibly compiled-graph) parameter arrays.
        mx.eval(self.parameters)
        np.savez(
            directory / "weights.npz",
            **{name: np.array(value) for name, value in self.parameters.items()},
        )
        meta = {
            "n_features": N_FEATURES,
            "accumulator_size": ACCUMULATOR_SIZE,
            "hidden_size_1": HIDDEN_SIZE_1,
            "hidden_size_2": HIDDEN_SIZE_2,
            "target_scale": DEFAULT_TARGET_SCALE,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load(directory: str | Path) -> "NnueNet":
        import numpy as np

        model = NnueNet.__new__(NnueNet)
        nn.Module.__init__(model)
        weights = np.load(Path(directory) / "weights.npz", allow_pickle=False)
        for name, value in weights.items():
            setattr(model, name, mx.array(value))
        return model
