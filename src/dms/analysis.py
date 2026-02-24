from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml
import pymbar


@dataclass
class WorkSet:
    w_fwd: np.ndarray  # reduced work (dimensionless): beta * W
    w_rev: np.ndarray
    beta: float        # 1/(kJ/mol)
    temperature_k: float


def kT_kjmol(T: float) -> float:
    # kB = 0.008314462618 kJ/mol/K
    return 0.008314462618 * T


def load_resolved_config(run_dir: Path) -> Dict[str, Any]:
    return yaml.safe_load((run_dir / "config_resolved.yaml").read_text())


def load_workset(run_dir: Path) -> WorkSet:
    cfg = load_resolved_config(run_dir)
    T = float(cfg["protocol"]["dynamics"]["temperature_k"])

    # These are already reduced works in kT units (dimensionless)
    w_fwd = np.load(run_dir / "forward_final_works_kT.npy").astype(float)
    w_rev = np.load(run_dir / "reverse_final_works_kT.npy").astype(float)

    mask = np.isfinite(w_fwd) & np.isfinite(w_rev)
    w_fwd = w_fwd[mask]
    w_rev = w_rev[mask]

    # beta is not used anymore, but keep field for compatibility
    beta = 1.0
    return WorkSet(w_fwd=w_fwd, w_rev=w_rev, beta=beta, temperature_k=T)


def kT_to_kcalmol(x_kT: float, T: float) -> float:
    # kB*T in kcal/mol = 0.001987204258 * T
    return float(x_kT) * (0.001987204258 * T)


def kT_to_kjmol(x_kT: float, T: float) -> float:
    return float(x_kT) * (0.008314462618 * T)


def bar_dG_kT(w_fwd: np.ndarray, w_rev: np.ndarray) -> tuple[float, float]:
    
    

    w_fwd = np.asarray(w_fwd, dtype=float)
    w_rev = np.asarray(w_rev, dtype=float)

    # filter non-finite (defensive)
    mask = np.isfinite(w_fwd) & np.isfinite(w_rev)
    w_fwd = w_fwd[mask]
    w_rev = w_rev[mask]
    if len(w_fwd) < 2 or len(w_rev) < 2:
        return float("nan"), float("nan")

    res = pymbar.bar(w_fwd, w_rev)  # PyMBAR 4 returns dict
    return float(res["Delta_f"]), float(res.get("dDelta_f", np.nan))



def bootstrap_bar_kT(w_fwd: np.ndarray, w_rev: np.ndarray, nboot: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    w_fwd = np.asarray(w_fwd, dtype=float)
    w_rev = np.asarray(w_rev, dtype=float)

    Nf, Nr = len(w_fwd), len(w_rev)
    dGs = []

    for _ in range(nboot):
        wf = w_fwd[rng.integers(0, Nf, size=Nf)]
        wr = w_rev[rng.integers(0, Nr, size=Nr)]
        dG, _ = bar_dG_kT(wf, wr)
        if np.isfinite(dG):
            dGs.append(dG)

    if len(dGs) < 20:
        return float("nan"), float("nan")

    dGs = np.array(dGs, dtype=float)
    return float(dGs.mean()), float(dGs.std(ddof=1))




def qc_basic(w_fwd: np.ndarray, w_rev: np.ndarray) -> Dict[str, Any]:
    w_fwd = np.asarray(w_fwd, dtype=float)
    w_rev = np.asarray(w_rev, dtype=float)

    mf = float(np.mean(w_fwd)) if w_fwd.size else float("nan")
    mr = float(np.mean(w_rev)) if w_rev.size else float("nan")

    sf = float(np.std(w_fwd, ddof=1)) if w_fwd.size > 1 else 0.0
    sr = float(np.std(w_rev, ddof=1)) if w_rev.size > 1 else 0.0

    pooled = max(1e-8, (sf + sr) / 2.0)

    # heuristic overlap metric (higher is worse)
    z = float(abs(mf - (-mr)) / pooled) if np.isfinite(mf) and np.isfinite(mr) else float("nan")

    return {
        "Nf": int(w_fwd.size),
        "Nr": int(w_rev.size),
        "mean_fwd_kT": mf,
        "std_fwd_kT": sf,
        "mean_rev_kT": mr,
        "std_rev_kT": sr,
        "overlap_z_heuristic": z,
        "flag_low_overlap": bool(z > 10.0) if np.isfinite(z) else True,
    }

