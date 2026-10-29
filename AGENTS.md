# amy-training-nnue — agent guide

Training app for amy's NNUE (branch `feature/relative-king-positions`,
checked out at `../amy-eval-nnue`). Reads amy selfplay BPD logs, trains a
float network with MLX, quantizes it into amy's exact integer pipeline, and
emits the `network.quant` file amy loads at startup.

## Hard invariant: bit-exact with amy's C

`QuantizedNet` (`amy_training_nnue/quantize.py`) must reproduce amy's C
`neural_net.c` **bit-for-bit**, not approximately:

- int16 accumulators with **wrap-around** (numpy `.astype(np.int16)` matches
  arm64 C; the raw accumulator is stored unclamped, clamping to
  `[0, SCALE_INPUT]` happens only on the 2*ACCUMULATOR_SIZE (256)-dim
  local copy at evaluation time).
- The pipeline works in **1/64 units** from the accumulator through
  hidden layer 1: input bias int16 @ 64, input weights int8 @ 64 (the hot
  loop is a plain `int16 += int8`, no per-feature rescale), h1 weights
  int8 @ 64 with an int16 @ 64*64 bias (sum in 64*64 units), h2 weights
  int16 @ 8192 with an int32 @ 8192*64 bias (sum in 8192*64 units).  The
  conversion to the 8192 units the output transform expects is two `/64`
  divisions with `+31` rounding.
- Clamped ReLU: h1: `h = 0` if `q <= 0`, else `64` if `q > 64*64`
  (strict `>`), else `trunc((q + 31) / 64)`.  h2: `h = 0` if `q <= 0`,
  else `8192` if `q >= 8192*64` (`>=`), else `trunc((q + 31) / 64)` —
  match the C per layer.
- Integer division **truncates toward zero** (C), not floor (Python).
  Use `_trunc_div`, never `//` on possibly-negative int values.
- Output transform is two steps:
  `r = trunc((out_q + 4095) / 8192); score = trunc((r * 10000 + 4095) / 8192)`.
  Both `+ 4095` terms are required.
- Black-side feature squares are rank-flipped: `sq ^ 0x38`, including the
  black king square when indexing the black feature plane.
- Feature index: `4096 * piece_type + 64 * king_sq + sq`,
  `piece_type` 0..4 = own P N B R Q, 5..9 = opponent P N B R Q,
  plus plane 10: `40960 + 64 * king_sq + opp_king_sq` (opponent king
  relative to own king, always active; see
  `nn_build_accumulator()` in neural_net.c).  N_FEATURES = 11 * 64 * 64
  = 45056.

Any change to the pipeline must be re-validated:

1. `python -m pytest tests/ -q` — 52 tests, including
   `tests/test_crosscheck.py`, which compares against C ground truth stored
   in `tests/data/crosscheck_reference.json` (regenerate with
   `python tools/make_crosscheck_reference.py [AMY_DIR]`, which needs
   `src/nn_crosscheck` built by a plain `make` in the amy checkout).
2. For export changes: copy `network.quant` into a scratch dir, run
   `Amy --test` there (all tests must pass), and compare
   `nn_crosscheck` scores with `QuantizedNet.evaluate`.

## BPD format

40-byte records: 4 x u64 board (64 nibbles; square `sq` = nibble `sq & 15`
of `pos[sq >> 4]`), u64 castling/EP specials, i32 score (millipawns,
white perspective), u8 turn, padding. Nibbles: 0 empty, 1-7 white
P..K, 8-14 black P..K, 5/12 also mark castling rooks (position-dependent),
15 = EP target square. Use `Position.epd()` / `position_from_record()` —
do not hand-roll nibble extraction.

## MLX notes (version-specific gotchas)

- `import mlx.optimizers as mxopt` (top-level module);
  `mxopt.AdamW(learning_rate=..., weight_decay=...)`.
- Do **not** use `mx.compile` with `outputs=model.state` around
  `nn.value_and_grad` + optimizer update: parameter arrays end up
  "without a primitive" and `mx.eval`/`np.array` fail. Plain (uncompiled)
  steps are fine — an epoch is ~2 s on M1 Pro.
- `NnueNet.save()` calls `mx.eval` first; keep that if you add compiled
  code paths.
- Model checkpoints are `weights.npz` (numpy), not safetensors.

## Conventions

- Scores are millipawns everywhere on the outside; the float model works in
  units of `1/10000` pawn (`target = score_mp / 10000`, `SCALE_OUTPUT=10000`).
- `train` augments the dataset by default (`--mirror-augment` /
  `--no-mirror-augment`): positions where neither side has castling rights
  are additionally trained on their horizontal mirror image (file flip
  `sq ^ 0x07`; same score and turn — the mirror is a legal position with the
  same value), and positions with castling rights are duplicated verbatim so
  every original position keeps the same weight (dataset doubles in size).
  A 10-epoch A/B (batch 2048, seed 0) on the full log reduced quantized val
  rmse from 375 to 265 mp and the mirror asymmetry from 584 to 206 mp mean.
  `BpdDataset` itself keeps `mirror_augment=False` as its default — the
  `evaluate` command uses the plain dataset for diagnostics.
- Keep the `network.quant` export byte-identical to amy's layout: a
  20-byte little-endian `WeightsHeader` (magic `0x57594D41`, whose file
  bytes read as "AMYW",
  then accumulator size 128, number of input features 45056, hidden
  layer sizes 32/32, per-field quantization codes 1/2/3 =
  int8/16/32; see `weights_header()`) followed by 5,777,988 bytes of
  fields in `write_network_quant`'s order (5,778,008 bytes total).
- Style: `black` + `isort` (configured in `pyproject.toml`).
- Data lives in `../amy/evaluation_log.bin` (grows with every selfplay
  session — tests must not read it). `tests/data/selfplay_sample.bin` is
  a frozen 5% snapshot (seed 42) used by the training smoke test;
  deliberately refresh it with `python tools/make_sample.py` and commit
  the result. `tests/data/bpd_test.bin` is a 5-record
  cross-implementation fixture (produced by amy's `bpd-test`).

## Layout

```
amy_training_nnue/
  bpd.py       40-byte record parse/encode, validation
  position.py  Position dataclass, EPD, nibble helpers
  features.py  king-relative feature indices incl. opponent-king plane
               (must match neural_net.c)
  dataset.py   BPD -> padded feature tensors + targets, mirror
               augmentation (on by default in train)
  model.py     NnueNet (MLX float)
  quantize.py  QuantizedNet (int simulation), network.quant I/O
  train.py     training loop, checkpoints, metrics, export
  cli.py       stats / inspect / train / export / evaluate
tests/         pytest suite incl. bit-exact C cross-check
tools/         make_crosscheck_reference.py
```
