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
from amy_training_nnue.quantize import quantize_parameters, write_network_quant  # noqa: E402

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
]

SEED = 1234

# input_w is drawn from [-1, 1]: the range representable by int8 at
# scale 64 (and the range trained weights are kept in).
rng = np.random.default_rng(SEED)
params = {
    "input_w": rng.uniform(-1, 1, (40960, 256)),
    "input_b": rng.uniform(-1, 1, (256,)),
    "h1_w": rng.uniform(-2, 2, (512, 32)),
    "h1_b": rng.uniform(-2, 2, (32,)),
    "h2_w": rng.uniform(-2, 2, (32, 32)),
    "h2_b": rng.uniform(-2, 2, (32,)),
    "out_w": rng.uniform(-1, 1, (32,)),
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
