"""Training loop for the NNUE (MLX)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as mxopt
import numpy as np
from mlx.nn.losses import huber_loss

from .dataset import BpdDataset
from .model import DEFAULT_TARGET_SCALE, WEIGHT_NAMES, NnueNet
from .quantize import (
    NETWORK_QUANT_SIZE,
    QuantizedNet,
    quantize_parameters,
    write_network_quant,
)


@dataclass
class TrainConfig:
    paths: list[str]
    output_dir: str = "checkpoints"
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    weight_bound: float | None = (
        1.0  # clip weight tensors (not biases) to [-B, B]; None disables
    )
    val_fraction: float = 0.1
    seed: int = 0
    target_scale: float = DEFAULT_TARGET_SCALE
    # Mirror dataset augmentation (see BpdDataset): on by default.
    mirror_augment: bool = True
    log_every: int = 20
    save_every: int = 10
    resume: str | None = None
    history: list = field(default_factory=list)


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Metrics in millipawns; pred/target in the model's output units."""
    if len(pred) == 0:
        return {
            "rmse_mp": float("nan"),
            "mae_mp": float("nan"),
            "corr": float("nan"),
            "r2": float("nan"),
            "within_25cp": float("nan"),
        }
    scale = float(DEFAULT_TARGET_SCALE)
    pred_mp = pred.astype(np.float64) * scale
    target_mp = target.astype(np.float64) * scale
    diff = pred_mp - target_mp
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    pc = float(np.corrcoef(pred_mp, target_mp)[0, 1]) if len(pred) > 1 else float("nan")
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((target_mp - target_mp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    within_25cp = float(np.mean(np.abs(diff) < 2500))
    return {
        "rmse_mp": rmse,
        "mae_mp": mae,
        "corr": pc,
        "r2": r2,
        "within_25cp": within_25cp,
    }


def _fmt_metrics(m: dict) -> str:
    return (
        f"rmse {m['rmse_mp']:.0f} mp, mae {m['mae_mp']:.0f} mp, "
        f"corr {m['corr']:.4f}, r2 {m['r2']:.4f}, "
        f"within 25cp {m['within_25cp']:.3f}"
    )


def model_predictions(
    model: NnueNet, ds: BpdDataset, idx: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    preds = []
    for i in range(0, len(idx), batch_size):
        sel = idx[i : i + batch_size]
        out = model(
            mx.array(ds.feat_w[sel]), mx.array(ds.feat_b[sel]), mx.array(ds.turn[sel])
        )
        preds.append(np.array(out))
    return np.concatenate(preds)


def _float_metrics(model: NnueNet, ds: BpdDataset, idx: np.ndarray) -> dict:
    pred = model_predictions(model, ds, idx)
    m = _metrics(pred, ds.target[idx])
    m["loss"] = float(np.mean((pred - ds.target[idx]) ** 2))
    return m


def _quant_metrics(
    qnet: QuantizedNet, ds: BpdDataset, idx: np.ndarray, batch_size: int = 1024
) -> dict:
    preds = []
    for i in range(0, len(idx), batch_size):
        sel = idx[i : i + batch_size]
        preds.append(qnet.evaluate(ds.feat_w[sel], ds.feat_b[sel], ds.turn[sel]))
    pred_mp = np.concatenate(preds).astype(np.float64)
    target_mp = ds.score[idx].astype(np.float64)
    diff = pred_mp - target_mp
    pc = (
        float(np.corrcoef(pred_mp, target_mp)[0, 1])
        if len(pred_mp) > 1
        else float("nan")
    )
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((target_mp - target_mp.mean()) ** 2))
    return {
        "rmse_mp": float(np.sqrt(np.mean(diff**2))),
        "mae_mp": float(np.mean(np.abs(diff))),
        "corr": pc,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "within_25cp": float(np.mean(np.abs(diff) < 2500)),
    }


def weight_max_abs(model: NnueNet) -> dict:
    """Max |w| per weight tensor, for monitoring the weight bound."""
    return {
        name: float(mx.max(mx.abs(getattr(model, name))).item())
        for name in WEIGHT_NAMES
    }


def _loss(model, fw, fb, turn, target):
    out = model(fw, fb, turn)
    # return mx.mean((out - target) ** 2)
    return mx.mean(huber_loss(out, target, 1.5))


def train(config: TrainConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = BpdDataset(
        config.paths,
        target_scale=config.target_scale,
        mirror_augment=config.mirror_augment,
    )
    if ds.n_invalid:
        print(f"skipped {ds.n_invalid} invalid records")
    augment_info = ""
    if config.mirror_augment:
        augment_info = f" ({ds.n_mirrored} mirrored, {ds.n_duplicated} duplicated)"
    print(
        f"loaded {ds.n} positions{augment_info}: "
        f"score {ds.score.min()}..{ds.score.max()} mp, "
        f"mean {ds.score.mean():.0f} mp"
    )

    if config.resume:
        model = NnueNet.load(config.resume)
        print(f"resumed from {config.resume}")
    else:
        model = NnueNet(seed=config.seed)
    if config.weight_bound is not None:
        model.clip_weights(config.weight_bound)

    train_idx, val_idx = ds.split(config.val_fraction, config.seed)
    print(f"train {len(train_idx)} / val {len(val_idx)}")

    opt = mxopt.AdamW(learning_rate=config.lr, weight_decay=config.weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, _loss)

    bound = config.weight_bound

    def step(fw, fb, turn, target):
        loss, grads = loss_and_grad_fn(model, fw, fb, turn, target)
        opt.update(model, grads)
        if bound is not None:
            model.clip_weights(bound)
        return loss

    losses = config.history
    for epoch in range(config.epochs):
        rng = np.random.default_rng(config.seed + epoch)
        order = rng.permutation(train_idx)
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(order), config.batch_size):
            sel = order[i : i + config.batch_size]
            fw = mx.array(ds.feat_w[sel])
            fb = mx.array(ds.feat_b[sel])
            turn = mx.array(ds.turn[sel])
            target = mx.array(ds.target[sel])
            loss = step(fw, fb, turn, target)
            loss_val = float(loss.item())
            epoch_loss += loss_val
            n_batches += 1
            if config.log_every and n_batches % config.log_every == 0:
                total_batches = (
                    len(order) + config.batch_size - 1
                ) // config.batch_size
                wm = weight_max_abs(model)
                print(
                    f"epoch {epoch + 1}/{config.epochs} "
                    f"step {n_batches}/{total_batches} "
                    f"loss {loss_val:.6f} "
                    f"max|w| in={wm['input_w']:.3f} h1={wm['h1_w']:.3f} "
                    f"h2={wm['h2_w']:.3f} out={wm['out_w']:.3f}",
                    flush=True,
                )
        dt = time.time() - t0
        avg = epoch_loss / max(n_batches, 1)
        losses.append({"epoch": epoch + 1, "loss": avg, "seconds": dt})
        print(
            f"epoch {epoch + 1}/{config.epochs} done in {dt:.1f}s, "
            f"avg loss {avg:.6f}",
            flush=True,
        )

        val = _float_metrics(model, ds, val_idx)

        if (epoch + 1) % config.save_every == 0 or epoch + 1 == config.epochs:
            ckpt = output_dir / f"epoch_{epoch + 1:03d}"
            model.save(ckpt)
            meta = {"val": val, "history": losses, "target_scale": config.target_scale}
            (ckpt / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  saved {ckpt.name}")
        print(f"  val: {_fmt_metrics(val)}")

    final = output_dir / "final"
    model.save(final)
    val = _float_metrics(model, ds, val_idx)
    meta = {
        "n_positions": ds.n,
        "mirror_augment": config.mirror_augment,
        "n_mirrored": ds.n_mirrored,
        "n_duplicated": ds.n_duplicated,
        "target_scale": config.target_scale,
        "epochs": config.epochs,
        "lr": config.lr,
        "seed": config.seed,
        "val": val,
        "history": losses,
    }
    (final / "meta.json").write_text(json.dumps(meta, indent=2))

    # Export the quantized network amy can load.
    params = {k: np.array(v) for k, v in model.parameters.items()}
    quantized = quantize_parameters(params)
    quant_path = output_dir / "network.quant"
    n_bytes = write_network_quant(quant_path, quantized)
    assert n_bytes == NETWORK_QUANT_SIZE
    print(f"wrote {quant_path} ({n_bytes} bytes)")

    qnet = QuantizedNet.from_arrays(quantized)
    qval = _quant_metrics(qnet, ds, val_idx)
    meta["val_quantized"] = qval
    (final / "meta.json").write_text(json.dumps(meta, indent=2))
    print("float     val:", _fmt_metrics(val))
    print("quantized val:", _fmt_metrics(qval))
    return final
