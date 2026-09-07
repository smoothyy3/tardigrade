"""
Find the NCAtorch checkout and put it on the import path.

Only training uses any of this. The shipped binary is pure Go.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _find() -> Path:
    if "NCATORCH_PATH" in os.environ:
        explicit = Path(os.environ["NCATORCH_PATH"]).resolve()
        if not (explicit / "nca").is_dir():
            raise SystemExit(f"NCATORCH_PATH={explicit} has no nca/ in it.")
        return explicit
    if (HERE / "nca").is_dir():
        return HERE
    sibling = (REPO.parent / "NCAtorch").resolve()
    if (sibling / "nca").is_dir():
        return sibling
    raise SystemExit(
        "NCAtorch not found. Clone https://github.com/mspitzna/NCAtorch next to "
        "this repo, or set NCATORCH_PATH to an existing checkout."
    )


sys.path.insert(0, str(_find()))

import nca  # noqa: E402

NCATORCH = Path(nca.__file__).resolve().parent.parent
