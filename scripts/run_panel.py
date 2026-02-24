#export PYTHONPATH="$PWD/src"
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from dms.run_one import run_one_mutation, MutationSpec


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def dump_yaml(obj: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def build_unbound_system_cfg(bound_system_cfg: dict) -> dict:
    """
    Given a system.yaml dict in our system-agnostic format, produce an unbound-leg system
    that contains only the mutated entity (target_pdb) and no partner_pdb.

    Assumes system.yaml format:
      system_name: ...
      inputs:
        target_pdb: ...
        partner_pdb: ...   # optional
      ... (other keys preserved)
    """
    cfg = dict(bound_system_cfg)  # shallow copy
    inputs = dict(cfg.get("inputs", {}))
    if "target_pdb" not in inputs:
        raise KeyError("system.yaml must contain inputs.target_pdb")

    # Drop partner for unbound leg (apo target)
    inputs.pop("partner_pdb", None)
    cfg["inputs"] = inputs

    # Optional: annotate name so run dirs are clearer
    if "system_name" in cfg and cfg["system_name"]:
        cfg["system_name"] = f"{cfg['system_name']}_unbound"

    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_panel.py", description="Run a mutation panel (bound + unbound legs).")

    p.add_argument("--protocol", required=True, type=Path, help="Path to protocol.yaml")
    p.add_argument("--system", required=True, type=Path, help="Path to system.yaml (bound definition)")
    p.add_argument("--out-dir", required=True, type=Path, help="Output root directory for runs/")
    p.add_argument("--csv", required=True, type=Path, help="Mutations CSV: chain,resid,mut_aa3")

    # Minimal useful overrides (optional)
    p.add_argument("--device-index", default="0", help="CUDA DeviceIndex (default: 0)")
    p.add_argument("--platform", default=None, help="Override platform name (e.g., CUDA/CPU). Default from protocol.yaml")
    p.add_argument("--smoke", action="store_true", help="Run tiny smoke test (very few steps/reps)")

    # Optional: override replicate/seeding from protocol.yaml
    p.add_argument("--n-reps", type=int, default=None, help="Override replicates.n_reps")
    p.add_argument("--base-seed", type=int, default=None, help="Override replicates.base_seed")
    p.add_argument(
        "--seed-stride-per-mutation",
        type=int,
        default=None,
        help="Override replicates.seed_stride_per_mutation (offset added per mutation index)",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    proto = load_yaml(args.protocol)
    reps = proto.get("replicates", {})

    n_reps = args.n_reps if args.n_reps is not None else int(reps.get("n_reps", 20))
    base_seed = args.base_seed if args.base_seed is not None else int(reps.get("base_seed", 1000))
    stride = (
        args.seed_stride_per_mutation
        if args.seed_stride_per_mutation is not None
        else int(reps.get("seed_stride_per_mutation", 10000))
    )

    platform_name = args.platform if args.platform is not None else proto.get("platform", {}).get("name", "CUDA")

    df = pd.read_csv(args.csv)

    # outputs into bound/ and unbound/ under out-dir
    out_bound = args.out_dir / "bound"
    out_unbound = args.out_dir / "unbound"
    out_bound.mkdir(parents=True, exist_ok=True)
    out_unbound.mkdir(parents=True, exist_ok=True)

    # Load the single bound system.yaml
    bound_system_cfg = load_yaml(args.system)

    # Write a temporary unbound system.yaml (target-only)
    unbound_system_cfg = build_unbound_system_cfg(bound_system_cfg)

    with tempfile.TemporaryDirectory() as td:
        tmp_unbound_path = Path(td) / "system_unbound.yaml"
        dump_yaml(unbound_system_cfg, tmp_unbound_path)

        for i, row in df.iterrows():
            mut = MutationSpec(chain=str(row.chain), resid=str(row.resid), mut_aa3=str(row.mut_aa3))

            # keep seed offset identical for both legs for this mutation index
            mut_base_seed = base_seed + stride * i

            # ---- bound leg ----
            run_dir_bound = run_one_mutation(
                system_cfg_path=args.system,
                protocol_cfg_path=args.protocol,
                mutation=mut,
                out_root=out_bound,
                n_reps=n_reps,
                base_seed=mut_base_seed,
                device_index=args.device_index,
                platform_name=str(platform_name),
                smoke=bool(args.smoke),
            )
            print("Wrote (bound):", run_dir_bound)

            # ---- unbound leg (target-only) ----
            run_dir_unbound = run_one_mutation(
                system_cfg_path=tmp_unbound_path,
                protocol_cfg_path=args.protocol,
                mutation=mut,
                out_root=out_unbound,
                n_reps=n_reps,
                base_seed=mut_base_seed,
                device_index=args.device_index,
                platform_name=str(platform_name),
                smoke=bool(args.smoke),
            )
            print("Wrote (unbound):", run_dir_unbound)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# command
"""

python scripts/run_panel.py \
  --protocol configs/protocol.yaml \
  --system configs/system.yaml \
  --out-dir runs/test_system \
  --csv mutations/test_panel.csv \
  --smoke

"""

"""
python scripts/run_panel.py \
  --protocol configs/protocol_speed_test.yaml \
  --system configs/system.yaml \
  --out-dir runs/speedtest \
  --csv mutations/test_panel.csv \
"""