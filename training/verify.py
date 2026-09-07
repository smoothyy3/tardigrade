"""
This test file is AI generated.

Three questions, pass/fail:
  1. does it grow from a single seed cell into the target?
  2. does it still hold that shape after a realistic idle?
  3. does it regrow after half the grid is deleted?

The idle check is long on purpose.

Usage:
    python training/verify.py --run training/runs/tardigrade
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from export_weights import load_run  # noqa: E402
from train import CreatureDataset  # noqa: E402

GROW_STEPS = 64
IDLE_STEPS = 20000
HEAL_STEPS = 200

GROW_MSE_MAX = 0.010
IDLE_MSE_MAX = 0.020
HEAL_MSE_MAX = 0.020


def to_rgb(state):
    """Alpha-composite the visible channels over white, as the terminal will."""
    rgb, alpha = state[0, :3], state[0, 3:4].clamp(0, 1)
    return (1.0 - alpha + rgb).clamp(0, 1).permute(1, 2, 0).numpy()


def mse(state, target):
    return torch.mean((state[0, :4].clamp(0, 1) - target[:4]) ** 2).item()


def rollout(model, state, steps, target, snapshot_every=None):
    """Run `steps` CA steps, returning the final state, the MSE trace and snapshots."""
    trace, snapshots = [], []
    with torch.no_grad():
        for step in range(steps):
            state, _ = model(state)
            trace.append(mse(state, target))
            if snapshot_every and (step + 1) % snapshot_every == 0:
                snapshots.append(to_rgb(state))
    return state, trace, snapshots


def check(name, value, limit):
    ok = value <= limit
    print(f"{'PASS' if ok else 'FAIL'}  {name}: mse {value:.5f} (limit {limit})")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "tardigrade")
    parser.add_argument("--idle-steps", type=int, default=IDLE_STEPS)
    parser.add_argument(
        "--target",
        type=Path,
        default=HERE / "target_40x30.png",
        help="the sprite this run was trained against",
    )
    args = parser.parse_args()

    model, config = load_run(args.run)
    canvas, channel_n = config.IM_HEIGHT, config.MODEL.CHANNEL_N
    target = CreatureDataset._load_target(args.target, canvas)
    seed = CreatureDataset._make_seed(canvas, channel_n).unsqueeze(0)

    # 1. growth
    grown, grow_trace, grow_snaps = rollout(
        model, seed, GROW_STEPS, target, snapshot_every=GROW_STEPS // 4
    )
    grew = check("grows from seed", grow_trace[-1], GROW_MSE_MAX)

    # 2. persistence
    idled, idle_trace, idle_snaps = rollout(
        model, grown, args.idle_steps, target, snapshot_every=args.idle_steps // 4
    )
    persists = check(f"holds for {args.idle_steps} idle steps", idle_trace[-1], IDLE_MSE_MAX)

    # 3. regrowth after losing the left half of the grid
    wounded = idled.clone()
    wounded[:, :, :, : canvas // 2] = 0.0
    healed, heal_trace, heal_snaps = rollout(
        model, wounded, HEAL_STEPS, target, snapshot_every=HEAL_STEPS // 4
    )
    heals = check(f"heals within {HEAL_STEPS} steps", heal_trace[-1], HEAL_MSE_MAX)

    rows = [
        ("grow", [to_rgb(seed)] + grow_snaps),
        ("idle", [to_rgb(grown)] + idle_snaps),
        ("heal", [to_rgb(wounded)] + heal_snaps),
    ]
    fig, axes = plt.subplots(len(rows), 5, figsize=(10, 6))
    for (label, frames), row in zip(rows, axes):
        for ax, frame in zip(row, frames):
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(label)
    fig.suptitle(f"grow {grow_trace[-1]:.4f} | idle {idle_trace[-1]:.4f} | heal {heal_trace[-1]:.4f}")
    fig.tight_layout()
    out = args.run / "verify.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    sys.exit(0 if (grew and persists and heals) else 1)


if __name__ == "__main__":
    main()
