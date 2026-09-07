"""
Dump a trained checkpoint to assets/tardigrade.nca for the Go runtime.

Also writes a fixed input/output pair for the Go forward pass to be tested
against.

Usage:
    python training/export_weights.py --run training/runs/tardigrade
"""

import argparse
import struct
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

sys.path.insert(0, str(HERE))
from framework import NCATORCH  # noqa: E402,F401

from nca.core.models.model_factory import create_model  # noqa: E402
from nca.utils.config import load_config  # noqa: E402

FLAG_TANH = 1 << 0
FLAG_CLAMP = 1 << 1

FIXTURE_SIZE = 8
FIXTURE_SEED = 7


def load_run(run: Path, deterministic=False):
    """Rebuild the CA from a run folder. Returns (model, config)."""
    config = load_config(run / "config.yaml")
    config.DEVICE = "cpu"
    if deterministic:
        config.MODEL.NOISE_INJECTION = 0.0
        config.MODEL.FIRE_RATE = 1.0
    model = create_model(config, config.COND_DIM, config.IM_HEIGHT, config.IM_WIDTH)
    model.load_state_dict(torch.load(run / "ca_final.pt", map_location="cpu"))
    model.eval()
    return model, config


def _floats(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().cpu().contiguous().flatten().tolist()
    return struct.pack(f"<{len(flat)}f", *flat)


def write_weights(model, config, out: Path):
    sd = model.state_dict()
    perception_w = sd["perception.conv.weight"]
    hidden_w = sd["update_model.model.0.weight"]
    output_w = sd["update_model.model.2.weight"]

    channel_n = perception_w.shape[1]
    percept_n, kernel = perception_w.shape[0], perception_w.shape[2]
    hidden_n = hidden_w.shape[0]

    flags = 0
    if config.MODEL.FINAL_ACTIVATION:
        flags |= FLAG_TANH
    if config.MODEL.CLAMP_OUTPUT:
        flags |= FLAG_CLAMP

    header = b"NCA1" + struct.pack(
        "<5I3fII",
        1,
        channel_n,
        percept_n,
        kernel,
        hidden_n,
        config.MODEL.FIRE_RATE,
        config.MODEL.NOISE_INJECTION,
        0.1,  # LivingMask threshold
        config.MODEL.LIVING_MASK_INDEX,
        flags,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(header)
        f.write(_floats(perception_w))
        f.write(_floats(sd["perception.conv.bias"]))
        f.write(_floats(hidden_w.squeeze(-1).squeeze(-1)))
        f.write(_floats(sd["update_model.model.0.bias"]))
        f.write(_floats(output_w.squeeze(-1).squeeze(-1)))
        f.write(_floats(sd["update_model.model.2.bias"]))
    print(f"{out} ({out.stat().st_size} bytes)  C={channel_n} P={percept_n} H={hidden_n}")


def write_fixture(run: Path, out: Path):
    model, config = load_run(run, deterministic=True)
    channel_n = config.MODEL.CHANNEL_N
    generator = torch.Generator().manual_seed(FIXTURE_SEED)
    state = (
        torch.rand((1, channel_n, FIXTURE_SIZE, FIXTURE_SIZE), generator=generator,) * 2 - 1
    )
    with torch.no_grad():
        stepped, _ = model(state)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"NCAF")
        f.write(struct.pack("<4I", 1, channel_n, FIXTURE_SIZE, FIXTURE_SIZE))
        f.write(_floats(state))
        f.write(_floats(stepped))
    print(f"{out} ({out.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs" / "tardigrade")
    parser.add_argument("--out", type=Path, default=REPO / "assets" / "tardigrade.nca")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=REPO / "internal" / "nca" / "testdata" / "forward_case.bin",
    )
    args = parser.parse_args()

    model, config = load_run(args.run)
    write_weights(model, config, args.out)
    write_fixture(args.run, args.fixture)


if __name__ == "__main__":
    main()
