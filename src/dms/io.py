from __future__ import annotations

import json
import os
import platform as py_platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"ERROR: {e}"


def write_metadata_json(path: Path, extra: Dict[str, Any] | None = None) -> None:
    """Write environment + platform provenance to metadata.json."""
    extra = extra or {}
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "platform": py_platform.platform(),
        "hostname": py_platform.node(),
        "cwd": str(Path.cwd()),
        "env": {
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "packages": {
            "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"]),
            "conda_list": _run(["conda", "list"]),
        },
        **extra,
    }
    path.write_text(json.dumps(meta, indent=2))


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def mutation_id(chain: str, resid: str, wt: str, mut: str, system: str = "SYSTEM") -> str:
    return f"{system}_{chain}_{resid}_{wt[0]}_{mut[0]}"

