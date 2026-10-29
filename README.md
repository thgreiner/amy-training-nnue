# amy-training-nnue

Training application for amy's NNUE evaluation function.

Trains the weights of the neural network used by amy's
`feature/neural-net-with-clamped-relu` branch from amy selfplay data in
[BPD format](https://github.com/thgreiner/amy), and exports the raw
`network.quant` weight file that amy loads at startup.

## Architecture

The network mirrors amy's C implementation (`src/neural_net.c`) exactly:

| Stage | amy C | Python |
|---|---|---|
| Features | 40960 king-relative bit-planes (6 piece types x own/opponent x 64 king squares x ... ), index `4096*tp + 64*king_sq + sq`, black side rank-flipped with `sq ^ 0x38` | `amy_training_nnue/features.py` |
| Accumulator | per-side int16 vector of 256 (units of 1/64), clamped to `[0, 64]` at evaluation | `model.py` (float), `quantize.py` (int16) |
| Network | 2x256 -> 32 -> 32 -> 1; h1 clamped ReLU `(h + 31) / 64` in `[0, 64]`, h2 `(h + 31) / 64` in `[0, 8192]` | `model.py` (float, `SCALE_INPUT_HALF` clamps), `quantize.py` (integer) |
| Output | `((out + 4095) / 8192) * 10000` then `(r + 4095) / 8192` -> millipawns, C truncating division | same, `quantize.py` |
| Constants | `SCALE = 8192`, `SCALE_INPUT = 64`, `SCALE_INPUT_HALF = 31`, `SCALE_HALF = 4095`, `SCALE_OUTPUT = 10000` | identical |

Training happens in floating point (MLX, `NnueNet`), then the parameters are
quantized into the exact integer pipeline amy runs: the accumulator is kept
in 1/64 units (input weights `int8` x 64, input bias `int16` x 64), hidden 1
uses `int8` x 64 weights with an `int16` x 64 x 64 bias (activations in
[0, 64]), hidden 2 uses `int16` x 8192 weights with an `int32` x 8192 x 64
bias (activations in [0, 8192]), and the output layer `int32` x 8192 /
`int32` x 8192^2. `QuantizedNet` in `quantize.py` simulates that integer
pipeline bit-exactly and is used to report quantized metrics. The file is
10,505,028 bytes.

## Setup

```bash
conda env create -f environment.yml   # or: conda env update
conda activate amy-training-nnue
pip install -e .
```

## Usage

```bash
# Dataset statistics
python -m amy_training_nnue stats ../amy/evaluation_log.bin

# Inspect records (EPD, score, validity)
python -m amy_training_nnue inspect ../amy/evaluation_log.bin --count 10

# Train (float) + export quantized network.quant
python -m amy_training_nnue train ../amy/evaluation_log.bin \
    --output checkpoints/default --epochs 30 --batch-size 256 --seed 0

# Re-export a network from a saved float checkpoint
python -m amy_training_nnue export checkpoints/default/final --out network.quant

# Evaluate a quantized network on a BPD file
python -m amy_training_nnue evaluate checkpoints/default/network.quant \
    ../amy/evaluation_log.bin --show 5
```

Copy the produced `network.quant` into amy's working directory (next to the
`Amy` binary) and it is picked up at startup.

Training options of note: `--lr`, `--weight-decay`, `--val-fraction`,
`--save-every N` (checkpoint cadence; every checkpoint is ~42 MB),
`--resume <dir>` (continue from a float checkpoint).

## Data

Input is amy's selfplay evaluation log: 40-byte BPD records
(position + turn + score in millipawns, filtered by amy's
`bpd_log_eval_diff_threshold`). Labels are `score / 10000` in model units.

## Verification

The Python pipeline is verified **bit-exact** against amy's C code:

- `tests/test_crosscheck.py` + `tests/data/crosscheck_reference.json`:
  a random in-range network (fixed seed) is evaluated with amy's own
  `neural_net.c` (via `src/nn_crosscheck.c` in the amy checkout) and with
  `QuantizedNet`; raw int16 accumulators and final scores must be identical.
- A production-trained `network.quant` loads in amy (`nn_read_weights`),
  amy's full test suite (`Amy --test`, including `test_all_neural_net`)
  passes with it, and amy's scores match `QuantizedNet` for the same
  positions.

Regenerate the cross-check reference (needs the amy NNUE checkout):

```bash
# in the amy checkout (feature/neural-net-with-clamped-relu):
cd src && gcc -std=gnu23 -g -O3 -mcpu=apple-m1 -I../include -I.. \
    nn_crosscheck.c $(ls *.o | grep -v '^main.o$') -o nn_crosscheck
# then:
python tools/make_crosscheck_reference.py [AMY_DIR]
```

Run the full suite:

```bash
python -m pytest tests/ -q   # 40 tests
```

## Current results

On the ~1M-record selfplay log (`../amy/evaluation_log.bin`), 30 epochs,
AdamW lr=1e-3, batch 256, seed 0, on an M1 Pro (about 40 min):

```
val: rmse 485 mp, mae 176 mp, corr 0.9869, r2 0.9735, within 25cp 0.995
float and quantized metrics agree (rmse 485 vs 490 mp)
```

`tests/data/selfplay_sample.bin` is a frozen 5% snapshot of the selfplay
log (seed 42, ~60k records) used by the training smoke test; the live log
keeps growing and is never read by the tests. Refresh the snapshot with
`python tools/make_sample.py` if it becomes stale, and commit the result.

The residual is close to the noise floor of the labels (engine scores from
selfplay). More/better data (e.g. deeper search scores, more games) is the
main path to a stronger net.
