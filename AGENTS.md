# amy-training-nnue — agent guide

Training app for amy's NNUE (branch `feature/king-half-mirrored`,
checked out at `../amy-eval-nnue`). Reads amy selfplay BPD logs, trains a
float network with MLX, quantizes it into amy's exact integer pipeline, and
emits the `network.quant` file amy loads at startup.

## Hard invariant: bit-exact with amy's C

`QuantizedNet` (`amy_training_nnue/quantize.py`) must reproduce amy's C
`neural_net.c` **bit-for-bit**, not approximately:

- int16 accumulators with **wrap-around** (numpy `.astype(np.int16)` matches
  arm64 C; the raw accumulator is stored unclamped, clamping to
  `[0, SCALE_INPUT]` happens only on the 2*ACCUMULATOR_SIZE (512)-dim
  local copy at evaluation time).
- The pipeline works in **1/64 units** from the accumulator through
  hidden layer 1: input bias int16 @ 64, input weights int8 @ 64 (the hot
  loop is a plain `int16 += int8`, no per-feature rescale), plus — for
  each active pawn pair — the shared pawn weight row int8 @ 64 (same
  `int16 += int8` add, one row for each side's accumulator); h1 weights
  int16 @ 8192 with an int32 @ 64*8192 bias (sum in 64*8192 units), h2
  weights int16 @ 8192 with an int32 @ 8192*64 bias (sum in 8192*64
  units).  The sums are converted to the 64 / 8192 units of the
  activations (h1 in `[0, 64]`, h2 in `[0, 8192]`) by
  `trunc((q + 4095) / 8192)` for h1 and `trunc((q + 31) / 64)` for h2.
- Clamped ReLU: h1: `h = 0` if `q <= 0`, else `64` if `q > 64*8192`
  (strict `>`), else `trunc((q + 4095) / 8192)`.  h2: `h = 0` if `q <= 0`,
  else `8192` if `q >= 8192*64` (`>=`), else `trunc((q + 31) / 64)` —
  match the C per layer.
- Integer division **truncates toward zero** (C), not floor (Python).
  Use `_trunc_div`, never `//` on possibly-negative int values.
- Output transform is two steps:
  `r = trunc((out_q + 4095) / 8192); score = trunc((r * 10000 + 4095) / 8192)`.
  Both `+ 4095` terms are required.
- Black-side feature squares are rank-flipped: `sq ^ 0x38`, and if the
  side's own king (in its own frame) stands on files e..h, the *whole*
  frame is additionally mirrored horizontally (`sq ^ 0x07`), so the king
  is always canonicalized to the left half (files a..d).  The two masks
  commute.  A position and its horizontal mirror image therefore produce
  identical feature vectors.
- Feature index: `2048 * plane + 64 * king_index + sq'`, planes 0..4 =
  own P N B R Q, 5..9 = opponent P N B R Q, 10 = the opponent king in
  the side's own canonical frame (the own king is not a feature; the
  opponent king is plane 10; see `nn_build_accumulator()` in
  neural_net.c).  `king_index` = file + 4 * rank of the canonical
  own king (0..31: a1 = 0 ... d1 = 3, ... a8 = 28 ... d8 = 31);
  `sq'` is the piece square (for plane 10: the opponent king's
  square) in the same frame (pieces are not
  restricted to the left half, only the king).  N_FEATURES =
  11 * 32 * 64 = 22528.
- Pawn pair features (see `nn_init_pawns()` in neural_net.c): every
  unordered pair of pawns within one file of distance (same file, any
  rank, or adjacent files) contributes one row of a *shared*
  N_PAWN_FEATURES x ACCUMULATOR_SIZE int8 matrix (4560 x 256, one row
  per unordered pair of the 96 color-square ids; both sides read the
  same matrix, but each pair is indexed in that side's own canonical
  frame, where white pawns have color 0 and black pawns 1 — swapped for
  black).  Id per pawn: `color * 48 + (sq' - 8)` (ranks 2..7 only —
  positions with pawns on rank 1 or 8 must not be fed to the pipeline,
  and bpd.py rejects them); pair index: combadic `hi * (hi - 1) / 2 +
  lo` (hi = max, lo = min), in 0..4559.  Only 1488 of the 4560 rows are
  ever active (the rest may hold zeros); both sides get the same pair
  *set*, mapped through their own frame.  Worked example (starting
  position): (b2, a2) is index 27 in the white frame (ids 6/7) and
  index 4559 in the black frame (ids 94/95, colors swapped).

Any change to the pipeline must be re-validated:

1. `python -m pytest tests/ -q` — 59 tests, including
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
do not hand-roll nibble extraction. `BpdRecord.validate()` rejects
positions with pawns on rank 1 or 8: the pawn pair feature index
(`sq' - 8`) would run out of bounds, as in the engine.

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

- The network structure must stay changeable for experiments: all
  shapes and sizes derive from the constants in `features.py`
  (`N_FEATURES`, `ACCUMULATOR_SIZE`, `HIDDEN_SIZE_1/2`, `PLANE_SIZE`)
  and `quantize.py` (`SCALE*`, `FIELD_ORDER`). Never hardcode
  architecture numbers (256, 32, 512, 8192, ...) in library code or
  tests; hand-computed test expectations are expressed in terms of
  those constants. Note that `tests/data/crosscheck_reference.json`
  is pinned to amy's C architecture — if the structure changes, the
  C side must be updated and the reference regenerated.
- Scores are millipawns everywhere on the outside; the float model works in
  units of `1/10000` pawn (`target = score_mp / 10000`, `SCALE_OUTPUT=10000`),
  and the float forward pass and the quantized pipeline stay linear in
  those units.  The *loss* is the exception: it is MSE on `tanh(out)` vs
  `tanh(target)` (see `_loss` in train.py), so the trained outputs are
  squeezed into (-1, 1); float-val metrics are measured in that same
  squeezed space.
- No mirror augmentation: with the king-half canonical frame, a
  position and its horizontal mirror image produce *identical* feature
  vectors (and the same target), so augmenting would only duplicate
  rows.  The old `--mirror-augment` flag (pre-canonicalization pipeline)
  has been removed.  `mirror_board` in position.py remains available for
  tests (e.g. the mirror-folding invariant in test_features.py).
- Keep the `network.quant` export byte-identical to amy's layout: a
  24-byte little-endian `WeightsHeader` (magic `0x57594D41`, whose file
  bytes read as "AMYW",
  then accumulator size 256, number of piece input features 22528,
  number of pawn input features 4560, hidden layer sizes 32/32,
  per-field quantization codes 1/2/3 = int8/16/32 — input weights 1,
  input bias 2, h1 weights 2, h1 bias 3, h2 weights 2, h2 bias 3,
  output weights 3, output bias 3; the pawn weights have *no* code and
  are read unconditionally as int8.  See `weights_header()`.  The C
  struct is 22 field bytes + 2 trailing padding — pack with
  `struct.pack("<IHHHHH8B2x", ...)`, a 22-byte pack silently
  misaligns the whole file) followed by 6,970,244 bytes of fields in
  `write_network_quant`'s order (6,970,268 bytes total): piece input
  weights, the shared 4560 x 256 int8 pawn block, input bias, hidden
  and output layers.
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
  features.py  king-relative feature indices + pawn pair indices
               (must match neural_net.c)
  dataset.py   BPD -> padded feature tensors + targets
  model.py     NnueNet (MLX float)
  quantize.py  QuantizedNet (int simulation), network.quant I/O
  train.py     training loop, checkpoints, metrics, export
  cli.py       stats / inspect / train / export / evaluate
tests/         pytest suite incl. bit-exact C cross-check
tools/         make_crosscheck_reference.py
```
