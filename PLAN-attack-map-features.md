# Plan: attack-map features + C feature generation for training

Status: **in progress** — T0 done (plus engine integration, see Progress
log); T1–T5 open.
Target branch: `feature/neural-network-with-pawns` (repo `amy-eval-nnue`; the
original `feature/king-half-mirrored` target was superseded).
Date: 2026-02-25
Note: after T1 this document moves to the amy repo root (PLAN-*.md convention).

## Goals

1. Add attack-map features to the NNUE (planes update the accumulator like
   piece/pawn features, but with a different indexing scheme).
2. Make the C code the **single source of truth** for feature generation
   (accumulator indices, pawn-pair indices, attack indices) so C (engine
   evaluation) and Python (training) cannot drift.
3. Move feature generation (including BPD parsing) into C and call it from
   Python via ctypes, fixing the slow per-record Python loop in
   `dataset.py`.
4. Merge the training code into the amy repository (NNUE branch).

## Key decisions (agreed)

- D1. Training code moves **into** the amy repo (`amy-eval-nnue`), as a
  subdirectory (proposed: `training/`). Autotools owns C; pip/conda owns
  Python; no Python build integration into autotools.
- D2. **No pure-Python fallback**: after the C feature library exists, the
  C build is a hard dependency of training. `features.py` leaves the
  runtime path; its tests are ported to the C path or removed (golden files
  are the guard).
- D3. BPD record parsing + validation happens in C (same rules as
  `BpdRecord.validate`: one king per side, no pawns on rank 1/8).
- D4. Feature reuse from C = **indices only** (accumulator indices, pawn
  indices, attack indices). The integer pipeline in `quantize.py`
  (`QuantizedNet`) stays in Python as the cross-check oracle.
- D5. ~~**Incremental** engine accumulator updates for attack features are
  **out of scope** here — separate task (see Follow-ups).~~
  **Superseded (2026-11-06)**: the incremental engine update was
  implemented as part of the "use the attack weights" work — it is a
  set-diff re-sync at evaluation time, so F1 is absorbed into this plan.
- D6. Attack features are a **separate int8 weight matrix** block (same
  pattern as the pawn block: `int16 += int8` row sums, own `.quant` header
  field), not merged into the 22,528-dim piece input.

## Out of scope

- Any change to the hidden/output layers or quantization scales.
- Mirror augmentation (still none; see open item O3).

## Prerequisite state (updated 2026-11-06)

- Attack-map index code is **committed** in `amy-eval-nnue` (`34c4cef`
  "Add attack_map generation": `src/attack_map.c`, `src/intset_sorted.c`,
  …). The API is `am_compute_attack_maps()`; per-side indices are
  `64*64*plane + from*64 + to` over **24 planes** (attacker P..Q ×
  victim P..Q, excluding pawn→pawn; no attacks on kings; queen attacks
  on bishops/rooks restricted to the straight/diagonal rays the other
  piece type does not already cover; bishop/queen attacks on pawns
  exclude squares already attacked by pawns) = `N_ATTACK_FEATURES` =
  98,304 rows per side, computed off a full `struct Position` (`atkTo`,
  `mask`) with the side's canonical frame applied to both `from` and `to`
  (folds under mirroring, O3).
- The engine **uses** the attack block in evaluation: `nn_update_attack_maps()`
  re-syncs the stored per-side attack maps by set-diff and applies the
  rows to the accumulator (int16 wrap-around, same pattern as the pawn
  block); maps are reset on king moves and position init (committed
  2026-11-06, `8c98046` "Add attack features to the NNUE input").
  The block is not loaded from `network.quant` yet (T5), so in
  production it is all-zero and the contribution is a no-op — old
  network files remain loadable with unchanged behaviour.

## Tasks

### T0 — Finalize attack-map index generation (C)

- [x] Review and finish `attack_map.c` / `intset_sorted` /
      `test_attack_map` (committed `34c4cef`; engine integration on top,
      `8c98046`, see Progress log).
- [x] Resolve O3 (frame normalization of attack indices) — option (b):
      the side's canonical frame is applied to `from`/`to`, so attacks
      fold under mirroring like piece features.
- [x] `make` + `Amy --test` green in `amy-eval-nnue`.
- [x] Commit to the working branch (`feature/neural-network-with-pawns`).

Gate: met (clean build, `Amy --test` passes incl. the new attack tests in
`test_neural_net.c`, committed).  Extra: `intset_sorted_add` overflow in
`am_compute_attacks` now exits(1) instead of truncating (R1).

### T1 — Repo merge: training code into amy repo

- [ ] Move the training tree into `amy-eval-nnue/training/` (package
      `amy_training_nnue`, `tests/`, `tools/`, `conftest.py`,
      `pyproject.toml`, `environment.yml`, `tests/data/` fixtures).
- [ ] Resolve O1 (location of `evaluation_log.bin`).
- [ ] Update `AGENTS.md` / README in both repos; move this plan document
      to the amy repo root.
- [ ] Retire/archive the standalone `amy-training-nnue` repo.

Gate: `python -m pytest tests/ -q` green from the new location; no
references to the old repo remain.

### T2 — C feature-extraction library + dataset via ctypes

- [ ] New `src/nn_extract.c` (name TBD) exposing a batch API, roughly:
      `nn_extract_features(records, n, out_feat_w/b, out_pfeat_w/b,
      out_afeat_w/b, out_turn/score, out_counts, out_invalid)`.
      In C: decode BPD nibbles, validate, build `struct Position`, then
      compute piece accumulator indices (refactored out of
      `nn_build_accumulator`), pawn-pair indices (refactored from
      `nn_init_pawns`/`nn_pp_index`), attack indices
      (`am_show_attack_map`).
- [ ] Refactor rule: the engine's own position-init path calls the **same**
      index helpers (single source of truth).
- [ ] API takes a *range* of records and preallocated output slices, so the
      dataset can be built lazily/chunked (see risk R2).
- [ ] Overflow guard: C returns an error if a record exceeds the output
      capacity (esp. attacks, see O2) — never silently truncate.
- [ ] Constants export function (N_FEATURES, N_PAWN_FEATURES,
      N_ATTACK_FEATURES, …) for the sync test (T3).
- [ ] `Makefile.am`: library target reusing `amy_core_sources` (the
      `nn_crosscheck` recipe proves the source set links).
- [ ] Python: small ctypes wrapper module; `dataset.py` rewritten around
      one C call per file/chunk; `features.py` removed from the runtime
      path (D2); its tests ported to the C path.
- [ ] C/Python validation-rule cross-test with hand-built invalid records
      (kings, rank 1/8 pawns).
- [ ] Golden-file test: frozen index outputs for `tests/data/bpd_test.bin`
      and the selfplay sample (regenerated by the C library).

Gate: dataset for the 5% sample builds with C and matches golden files;
throughput measured and recorded (target: order-of-magnitude faster than
the current Python loop); full pytest green.

### T3 — Sync & validation machinery

- [ ] Extend `nn_crosscheck.c` to dump all index sets (piece, pawn,
      attack); regenerate `tests/data/crosscheck_reference.json`;
      `test_crosscheck.py` covers the new sets.
- [ ] Constants sync test: C-exported constants vs Python constants
      (machine-checked invariant, replaces doc-level "must match").
- [ ] Update `AGENTS.md` invariants to reflect C-as-source-of-truth.

Gate: full pytest; cross-check against a freshly built `nn_crosscheck`.

### T4 — Training pipeline for attack features

- [ ] `model.py`: second sparse input block per side (attack indices →
      attack weight matrix), per D6.
- [ ] `quantize.py`: `QuantizedNet` sums attack rows into the accumulator
      (same wrap-around int16 semantics as the pawn block); new
      `N_ATTACK_FEATURES` constant; `weights_header()` + export write the
      new block and header fields.
- [ ] `.quant` header: new field(s) for attack-feature count + quant code;
      C `WeightsHeader` struct and `struct.pack` format change in the
      **same commit** as the C loader (T5) — one repo now, so one commit.
- [ ] Training smoke test on `tests/data/selfplay_sample.bin` (targets and
      squeezed-space metrics unchanged).

Gate: full pytest; exported `network.quant` loads in the new C build;
**training/experiment-only** — do not deploy to the engine until T5 lands.

### T5 — C engine: load attack block (paired with T4)

- [ ] C `WeightsHeader` + loader read the attack block (size assert +
      fread), matching the Python export byte-for-byte.
- Note: the original second bullet ("matrix is loaded but **not yet
      used** in evaluation") is obsolete — evaluation **already
      consumes** the block (`nn_update_attack_maps`).  Until a `.quant`
      carrying the block is loaded the matrix is all-zero and the
      contribution is a no-op.

Gate: `Amy --test` green; a network exported by T4 loads without error
and changes evaluation (non-zero contribution).

## Follow-ups (separate tasks, not in this plan)

- F1. ~~C engine attack integration: accumulator contribution at position
      init **plus per-move incremental updates** (D5).~~ **Absorbed
      (2026-11-06)**: the incremental update is implemented in the engine
      (`nn_update_attack_maps`, see Prerequisite state).  An
      attack-enabled network is deployable to the engine once T5 loads
      the block.
- F2. Refresh `tests/data/selfplay_sample.bin` once a network trained with
      attack features is in self-play use.

## Cross-cutting notes

- **Format pairing**: any `.quant` layout change touches C struct + loader,
  Python `weights_header()`, and the export — must land together (easy now
  that both sides live in one repo/branch).
- **Reference pinning**: `crosscheck_reference.json` is pinned to the C
  architecture; regenerate whenever C feature logic changes (T0/T2).
- **Cross-check pinning of the attack block**: `nn_crosscheck` loads the
  Python-exported `network.quant`, which has no attack block until T4/T5
  land — the C-side attack matrix is zero there, so the existing
  bit-exact comparison is unaffected.  T3 must extend the dump (attack
  indices, and a variant with random attack weights) once the format
  pairs up.
- **Pawn-rank validation**: `BpdRecord.validate` rules are duplicated in C
  (T2) — keep the cross-test, it is the drift guard.

## Risks & open items

- R1. **Resolved (2026-11-06)**: measured max active attack features is
  13 across many positions (the map stores only attacks on opponent
  pieces/pawns, and some redundant attacker/victim pairs are excluded);
  `INTSET_SORTED_MAX = 32` is kept.  `am_compute_attacks` checks the
  `intset_sorted_add` result and exits(1) on overflow instead of
  truncating silently.
- R2. **Dataset memory**: attack blocks add 2×K int32 per position
      (~+2 KB/position at K=256, ~3 GB per 1 M records total). The
      range-based C API (T2) enables chunked/lazy dataset construction if
      it doesn't fit in RAM; decide at T2 based on measured sizes.
- O1. **Data location**: does `evaluation_log.bin` stay external
      (`../amy/…`) or move with the repo? (T1 needs an answer; default
      proposal: stay external, path via CLI arg/env var.)
- O2. **Resolved with R1**: C-side maps hold ≤32 entries with a fatal
      overflow guard; the dataset (T2) can use the same bound for its
      padded tensors.
- O3. **Resolved (2026-11-06, option b)**: `am_compute_attack_maps`
  applies the side's canonical frame flip to both `from` and `to`, so a
  position and its horizontal mirror produce identical attack features
  (folds, like piece features).
- O4. **Resolved (2026-11-06)**: own-attacks only — each side's map
  (own pieces attacking opponent pieces/pawns) feeds that side's own
  accumulator.

## Progress log

- 2026-02-25: Plan written; decisions D1–D6 agreed; tasks T0–T5 defined.
- 2026-11-06: T0 done. Attack-map index generation committed (`34c4cef`);
  the engine now uses the attack block in evaluation (`8c98046` "Add
  attack features to the NNUE input"): `nn_update_attack_maps()`
  set-diff re-sync at evaluation, map resets on king moves and position
  init, `nn_validate_weights` includes the stored maps, debug printing in
  `dbase_show_position`, 14 new tests in `test_neural_net.c` (incl. a
  random-move consistency test); `make` + `Amy --test` green.  O3
  resolved (folded, option b), O4 resolved (own attacks only).  Plane
  design settled at 24 planes / 98,304 rows (attacker P..Q × victim
  P..Q minus pawn→pawn, with ray restrictions — see Prerequisite
  state).  R1 resolved: measured max 13 active features, capacity 32
  kept, overflow now fatal.  D5 superseded, F1 absorbed.
  Side note: the opponent-king plane 10 (separate plan,
  `PLAN-opponent-king-plane.md`) landed on both sides around the same
  time (C `ee0bd09`, training repo `8463e80`); `crosscheck_reference.json`
  was regenerated for it.
