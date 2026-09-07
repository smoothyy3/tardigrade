"""
Train a creature NCA from a sprite.

Thin wrapper around the NCAtorch fork

The creature is the --target, not
the code:

    python training/train.py --steps 60000
    python training/train.py --steps 60000 --name gecko --target training/target_gecko_40x30.png
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


from framework import NCATORCH  # noqa: E402

from nca.core.models.model_factory import create_model  # noqa: E402
from nca.data.dataset_factory import DATASET_REGISTRY, create_dataset  # noqa: E402
from nca.data.datasets.base_dataset import NCADataset  # noqa: E402
from nca.training.sample_pool import SamplePool  # noqa: E402
from nca.training.trainer_factory import create_trainer  # noqa: E402
from nca.utils.config import Config  # noqa: E402
from nca.utils.image_utils import make_circle_damage_mask  # noqa: E402

DEFAULT_TARGET = HERE / "target_40x30.png"


def _damage_a_fraction(self, states: torch.Tensor) -> torch.Tensor:
    """
    Wound only POOL_DMG_RATIO of the drawn states, instead of all of them.
    """
    count = round(states.shape[0] * self.current_damage_ratio)
    if count == 0:
        return states
    chosen = torch.randperm(states.shape[0], device=states.device)[:count]
    mask = make_circle_damage_mask(count, states.size(-1)).to(states.device)
    wounded = states.clone()
    wounded[chosen] = states[chosen] * mask
    return wounded


SamplePool.apply_damage = _damage_a_fraction

CANVAS = 56
CHANNEL_N = 16


class CreatureDataset(NCADataset):
    """
    One fixed target, one fixed single-pixel seed, no conditioning.
    """

    def __init__(self, target_png: Path, canvas: int, channel_n: int, total_samples: int):
        self.total_samples = total_samples
        self.target = self._load_target(target_png, canvas)
        self.seed = self._make_seed(canvas, channel_n)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, _idx):
        # cond is empty: COND_DIM == 0 makes the trainer drop it before the model.
        return self.seed.clone(), torch.zeros(0), self.target.clone()

    @staticmethod
    def _load_target(path: Path, canvas: int) -> torch.Tensor:
        img = np.float32(Image.open(path).convert("RGBA")) / 255.0
        img[..., :3] *= img[..., 3:]
        target = torch.from_numpy(img).permute(2, 0, 1)
        _, h, w = target.shape
        if h > canvas or w > canvas:
            raise SystemExit(
                f"{path.name} is {w}x{h}, larger than the {canvas}x{canvas} canvas. "
                "Shrink the sprite, or raise CANVAS here and --size in the binary to match"
            )
        top, left = (canvas - h) // 2, (canvas - w) // 2
        return F.pad(target, (left, canvas - w - left, top, canvas - h - top))

    @staticmethod
    def _make_seed(canvas: int, channel_n: int) -> torch.Tensor:
        seed = torch.zeros((channel_n, canvas, canvas), dtype=torch.float32)
        seed[:4, canvas // 2, canvas // 2] = 1.0
        return seed


def register_dataset(target: Path):

    def create(config: Config, _train: bool):
        total = (
            config.TRAINING.STEPS * config.TRAINING.BATCH_SIZE
            if config.TRAINING.STEPS != -1
            else float("inf")
        )
        dataset = CreatureDataset(target, CANVAS, config.MODEL.CHANNEL_N, total)
        return dataset, 0, CANVAS, CANVAS

    DATASET_REGISTRY["creature"] = create


def build_config(args) -> Config:
    return Config(
        SEED=args.seed,
        DEVICE=args.device,
        LOGGING={
            "WANDB": False,
            "PROJECT_NAME": "tardigrade",
            "TRAIN_NAME": args.name,
            "FOLDER_NAME": str(HERE / "runs" / args.name),
            "LOG_INTERVAL": 500,
            "SAVE_INTERVAL": 5000,
            "INTERMEDIATE_LOGGING_STEPS": [5, 20, 35],
        },
        MODEL={
            "NAME": "MLP",
            "HIDDEN_CHANNELS": [128],
            "CHANNEL_N": CHANNEL_N,
            "LIVING_MASK": True,
            "LIVING_MASK_INDEX": 3,
            "NOISE_INJECTION": 0.01,
            "FINAL_ACTIVATION": True,
            "CLAMP_OUTPUT": True,
            "FIRE_RATE": 0.5,
            "PERCEPTIONS": [{"MODE": "conv", "KERNEL_SIZE": 3, "OUT_CHANNEL": 80}],
        },
        TRAINING={
            "BATCH_SIZE": args.batch_size,
            "STEPS": args.steps,
            "LOSS_FN": "mse",
            "OVERFLOW_LOSS": True,
            "OVERFLOW_WEIGHT": 1.0,
            "LEARNING_RATE": 5e-4,
            "LR_SCHEDULE_MODE": "cosine",
            "OPTIMIZER_BETAS": [0.5, 0.95],
            "WARMUP_STEPS": min(1000, args.steps // 10),
            "ITER_N_MIN": 48,
            "ITER_N_MAX": 64,
            "GRADIENT_CLIPPING_NORM": 1.0,
            "MIXED_PRECISION": False,
        },
        DATASET={"NAME": "creature", "TARGET_SIZE": CANVAS, "NUM_WORKERS": 0},
        PATTERN_POOL={
            "ENABLED": True,
            "POOL_SIZE": 256,
            "POOL_DELAY": 1,
            "POOL_START_RATIO": 0.5,
            "POOL_END_RATIO": 0.5,
            "POOL_DMG_RATIO": 0.5,
            "POOL_DMG_DELAY": 1,
        },
    )


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="tardigrade")
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="target sprite; anything up to the canvas size works",
    )
    parser.add_argument("--device", type=str, default=default_device())
    return parser.parse_args()


def framework_revision() -> str:
    """
    Which NCAtorch produced these weights.
    """
    try:
        revision = subprocess.run(
            ["git", "-C", str(NCATORCH), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(NCATORCH), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return revision + (" (modified)" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    args = parse_args()
    register_dataset(args.target.resolve())
    print(f"NCAtorch: {NCATORCH} @ {framework_revision()}")
    print(f"Target:   {args.target.resolve()}")
    print(f"Device:   {args.device}")
    if args.seed != -1:
        import random

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    config = build_config(args)
    dataloader, cond_dim, height, width = create_dataset(config)
    config.set_cond_dim(cond_dim)
    config.set_im_height(height)
    config.set_im_width(width)

    model = create_model(config, cond_dim, height, width)
    trainer = create_trainer(config, model, dataloader, config_path=None)
    if trainer.train() == -1:
        print("Training diverged.")
        sys.exit(1)
    print(f"Checkpoints in {config.LOGGING.FOLDER_NAME}")


if __name__ == "__main__":
    main()
