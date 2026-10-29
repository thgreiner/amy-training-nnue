"""Load BPD self-play files into a training dataset.

``mirror_augment=True`` adds the horizontal mirror image (file flip,
``sq ^ 0x07``) of every position without castling rights, and
duplicates every position with castling rights verbatim, so that each
original position keeps the same weight in the (doubled) dataset.
The train command enables this by default; other callers (e.g. the
evaluate command) use the plain dataset.
"""

from __future__ import annotations

import numpy as np

from .bpd import read_records
from .features import feature_indices
from .position import mirror_board, position_from_record

# Maximum number of active input features per accumulator.  A position
# has at most 15 own + 15 opponent non-king pieces (30 for a full
# board) plus one opponent-king plane: 31.  33 leaves headroom.
MAX_FEATURES = 33


class BpdDataset:
    """All positions of one or more BPD files as packed tensors.

    Attributes:
        feat_w: (N, K) int32, white accumulator feature indices,
            padded with -1.
        feat_b: (N, K) int32, black accumulator feature indices.
        turn:   (N,) uint8, 0 = white to move, 1 = black.
        target: (N,) float32, stored score / target_scale.
        score:  (N,) int32, stored score in millipawns (side to move).
        n_mirrored: number of rows added as mirror images (0 if
            mirror_augment is off).
        n_duplicated: number of rows added as verbatim duplicates
            (0 if mirror_augment is off).
    """

    def __init__(
        self,
        paths: list[str],
        target_scale: float = 10000.0,
        mirror_augment: bool = False,
    ):
        if not paths:
            raise ValueError("no input files given")
        boards: list[np.ndarray] = []
        turns: list[int] = []
        scores: list[int] = []
        invalid = 0
        n_mirrored = 0
        n_duplicated = 0
        for path in paths:
            for record in read_records(path):
                problems = record.validate()
                if problems:
                    invalid += 1
                    continue
                board = record.board()
                boards.append(board)
                turns.append(record.turn)
                scores.append(record.score)
                if mirror_augment:
                    if position_from_record(record).castle:
                        # Duplicate verbatim so this position keeps the
                        # same weight as the (original, mirror) pairs.
                        boards.append(board)
                        turns.append(record.turn)
                        scores.append(record.score)
                        n_duplicated += 1
                    else:
                        # Neither side can castle: the mirror image is a
                        # valid position with the same value, so train
                        # on it as well (same score and turn).
                        boards.append(mirror_board(board))
                        turns.append(record.turn)
                        scores.append(record.score)
                        n_mirrored += 1
        if not boards:
            raise ValueError(f"no valid records found in {paths}")

        feat_w = np.full((len(boards), MAX_FEATURES), -1, dtype=np.int32)
        feat_b = np.full((len(boards), MAX_FEATURES), -1, dtype=np.int32)
        for i, board in enumerate(boards):
            w, b = feature_indices(board)
            if len(w) > MAX_FEATURES or len(b) > MAX_FEATURES:
                raise RuntimeError(
                    f"position {i} has {len(w)}/{len(b)} features > {MAX_FEATURES}"
                )
            feat_w[i, : len(w)] = w
            feat_b[i, : len(b)] = b

        self.feat_w = feat_w
        self.feat_b = feat_b
        self.turn = np.asarray(turns, dtype=np.uint8)
        self.score = np.asarray(scores, dtype=np.int32)
        self.target = self.score.astype(np.float32) / float(target_scale)
        self.n = len(boards)
        self.n_invalid = invalid
        self.n_mirrored = n_mirrored
        self.n_duplicated = n_duplicated

    def split(self, val_fraction: float = 0.1, seed: int = 0):
        """Shuffle and return (train_idx, val_idx) as int64 arrays."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n)
        n_val = int(self.n * val_fraction)
        if self.n >= 2:
            n_val = min(max(n_val, 1), self.n - 1)
        return idx[n_val:], idx[:n_val]
