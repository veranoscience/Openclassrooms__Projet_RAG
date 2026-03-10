from __future__ import annotations

import subprocess
from typing import Literal


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def rebuild(mode: Literal["index_only", "full"] = "index_only", force: bool = True) -> None:
    """
    Rebuild pipeline.
    - index_only : rebuild FAISS à partir du CSV clean
    - full       : fetch + preprocess + build index
    """
    if mode not in ("index_only", "full"):
        raise ValueError("mode must be 'index_only' or 'full'")

    if mode == "full":
        run_cmd(["python", "scripts/fetch_events.py"])
        run_cmd(["python", "scripts/preprocess.py"])

    cmd = ["python", "scripts/build_faiss_index.py"]
    if force:
        cmd.append("--force")
    run_cmd(cmd)