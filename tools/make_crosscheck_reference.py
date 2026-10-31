"""Regenerate the cross-check reference (tests/data/crosscheck_reference.json).

Re-runs amy's C implementation (src/nn_crosscheck.c in the amy NNUE checkout)
with a random in-range network generated with a fixed seed, and saves its
output.  Usage:

    python tools/make_crosscheck_reference.py [AMY_DIR] [CROSSCHECK_BIN]

Defaults: AMY_DIR=/Users/thorsten/Projects/amy-eval-nnue
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amy_training_nnue.features import (  # noqa: E402
    ACCUMULATOR_SIZE,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    N_FEATURES,
)
from amy_training_nnue.quantize import (  # noqa: E402
    quantize_parameters,
    write_network_quant,
)

AMY = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("/Users/thorsten/Projects/amy-eval-nnue")
)
BIN = Path(sys.argv[2]) if len(sys.argv) > 2 else AMY / "src" / "nn_crosscheck"
OUT = ROOT / "tests" / "data" / "crosscheck_reference.json"

EPS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -",
    "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq -",
    "3r1k2/p3qppP/8/3p4/3r4/5QP1/P4PB1/4R1K1 w - -",
    "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6",
    # Both kings on the left half (non-mirrored frame): the horizontal
    # mirror of the previous EPD.
    "2k1r3/Pppq3p/8/4p3/4r3/1PQ5/1BP4P/1K1R4 w - -",
]

SEED = 1234

# input_w is drawn from [-1, 1]: the range representable by int8 at
# scale 64 (and the range trained weights are kept in).
rng = np.random.default_rng(SEED)
params = {
    "input_w": rng.uniform(-1, 1, (N_FEATURES, ACCUMULATOR_SIZE)),
    "input_b": rng.uniform(-1, 1, (ACCUMULATOR_SIZE,)),
    "h1_w": rng.uniform(-2, 2, (2 * ACCUMULATOR_SIZE, HIDDEN_SIZE_1)),
    "h1_b": rng.uniform(-2, 2, (HIDDEN_SIZE_1,)),
    "h2_w": rng.uniform(-2, 2, (HIDDEN_SIZE_1, HIDDEN_SIZE_2)),
    "h2_b": rng.uniform(-2, 2, (HIDDEN_SIZE_2,)),
    "out_w": rng.uniform(-1, 1, (HIDDEN_SIZE_2,)),
    "out_b": rng.uniform(-1, 1, (1,)),
}
quantized = quantize_parameters(params)

tmp = Path(tempfile.mkdtemp(prefix="nnue_ref_"))
write_network_quant(tmp / "network.quant", quantized)

proc = subprocess.run([str(BIN)], cwd=tmp, capture_output=True, text=True)
if proc.returncode != 0:
    print(proc.stderr)
    sys.exit(1)

lines = [ln for ln in proc.stdout.splitlines() if ln != ""]
ref = {"seed": SEED, "positions": []}
i = 0
while i < len(lines):
    if lines[i].startswith("POS "):
        n = int(lines[i].split()[1])
        turn = int(lines[i].split()[-1])
        acc_w = np.array(lines[i + 1].split(), dtype=np.int32)
        acc_b = np.array(lines[i + 2].split(), dtype=np.int32)
        i += 3
        assert lines[i].startswith("SCORE ")
        score = int(lines[i].split()[-1])
        i += 1
        while i < len(lines) and lines[i].startswith("I "):
            i += 1
        assert lines[i].startswith("SCOREI ")
        assert int(lines[i].split()[-1]) == score
        i += 1
        ref["positions"].append(
            {
                "epd": EPS[n],
                "turn": turn,
                "score": score,
                "acc_w": acc_w.tolist(),
                "acc_b": acc_b.tolist(),
            }
        )
    else:
        i += 1

assert len(ref["positions"]) == len(EPS)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(ref))
print(f"wrote {OUT}")
