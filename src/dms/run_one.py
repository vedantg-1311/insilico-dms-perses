from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import time

import openmm

from openmm import unit
from openmmtools.integrators import PeriodicNonequilibriumIntegrator

from perses.app.relative_point_mutation_setup import PointMutationExecutor

from dms.io import ensure_dir, write_metadata_json, mutation_id


# --- Same lambda functions as your original script ---
x = "lambda"
DEFAULT_ALCHEMICAL_FUNCTIONS = {
    "lambda_sterics_core": x,
    "lambda_electrostatics_core": x,
    "lambda_sterics_insert": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    "lambda_sterics_delete": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    "lambda_electrostatics_insert": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    "lambda_electrostatics_delete": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    "lambda_bonds": x,
    "lambda_angles": x,
    "lambda_torsions": x,
}


@dataclass(frozen=True)
class MutationSpec:
    chain: str
    resid: str
    mut_aa3: str  # e.g., "GLY"


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def make_platform(
    name: str = "CUDA",
    precision: str = "mixed",
    deterministic: bool = True,
    device_index: Optional[str] = None,
) -> openmm.Platform:
    plat = openmm.Platform.getPlatformByName(name)
    if name in ["CUDA", "OpenCL"]:
        plat.setPropertyDefaultValue("Precision", precision)
    if name == "CUDA":
        plat.setPropertyDefaultValue("DeterministicForces", "true" if deterministic else "false")
        if device_index is not None:
            plat.setPropertyDefaultValue("DeviceIndex", str(device_index))
    return plat


def run_one_mutation(
    system_cfg_path: Path,
    protocol_cfg_path: Path,
    mutation: MutationSpec,
    out_root: Path,
    n_reps: int,
    base_seed: int,
    device_index: Optional[str] = "0",
    platform_name: str = "CUDA",
    smoke: bool = False,
) -> Path:
    """
    Runs NEQ switching for a single mutation using the same approach as the user's original script:
    - build HTF with PointMutationExecutor
    - use htf.hybrid_system + htf.hybrid_positions
    - use openmmtools PeriodicNonequilibriumIntegrator with DEFAULT_ALCHEMICAL_FUNCTIONS
    - accumulate protocol work via integrator.get_protocol_work(dimensionless=True)

    Outputs per run_dir:
      - forward_works_master.npy (shape: [n_reps, nsteps_neq+1]) dimensionless (kT)
      - reverse_works_master.npy
      - forward_final_works_kT.npy (shape: [n_reps])
      - reverse_final_works_kT.npy
      - optional position traces (old/new) for eq/neq at save frequencies
      - metadata.json, config_resolved.yaml, log.txt
    """
    system_cfg = load_yaml(system_cfg_path)
    proto_cfg = load_yaml(protocol_cfg_path)

    if n_reps is None:
        n_reps = int(proto_cfg["replicates"]["n_reps"])
    if base_seed is None:
        base_seed = int(proto_cfg["replicates"]["base_seed"])


    # barnase_pdb = Path(system_cfg["inputs"]["barnase_pdb"])
    # barstar_pdb = Path(system_cfg["inputs"]["barstar_pdb"])

    target_pdb = Path(system_cfg["inputs"]["target_pdb"])
    partner_pdb = Path(system_cfg["inputs"].get("partner_pdb", "")) if system_cfg["inputs"].get("partner_pdb") else None


    # protocol settings
    nsteps_eq = int(proto_cfg["neq"]["nsteps_eq"])
    nsteps_neq = int(proto_cfg["neq"]["nsteps_neq"])
    save_freq_eq = int(proto_cfg["neq"]["save_freq_eq"])
    save_freq_neq = int(proto_cfg["neq"]["save_freq_neq"])

    temperature = float(proto_cfg["dynamics"]["temperature_k"]) * unit.kelvin
    timestep = float(proto_cfg["dynamics"]["timestep_fs"]) * unit.femtosecond

    # splitting must match what your original script used unless you changed intentionally
    splitting = proto_cfg["dynamics"].get("splitting", "V R H O R V")
    if smoke:
        # keep it tiny and safe
        nsteps_eq = min(nsteps_eq, 2)
        nsteps_neq = min(nsteps_neq, 5)
        save_freq_eq = 1
        save_freq_neq = 1
        n_reps = min(n_reps, 1)

    # Build HTF
    ionic_strength = float(system_cfg["solvent"]["ionic_strength_molar"]) * unit.molar
    box_nm = system_cfg["solvent"]["box_dimensions_nm"]
    complex_box_dimensions = unit.Quantity(tuple(box_nm), unit.nanometer)

    platform = make_platform(
        name=platform_name,
        precision=proto_cfg["platform"].get("precision", "mixed"),
        deterministic=bool(proto_cfg["platform"].get("deterministic_forces", True)),
        device_index=device_index,
    )

    # print("DEBUG protein_filename:", target_pdb)
    # print("DEBUG ligand_input:", partner_pdb)
    # print("DEBUG mutation_chain_id:", mutation.chain)

    executor = PointMutationExecutor(
        protein_filename=str(target_pdb),
        mutation_chain_id=mutation.chain,
        mutation_residue_id=str(mutation.resid),
        proposed_residue=mutation.mut_aa3,
        ligand_input=str(partner_pdb) if partner_pdb else None,
        ionic_strength=ionic_strength,
        flatten_torsions=True,
        flatten_exceptions=True,
        conduct_endstate_validation=False,
        nonbonded_method="PME",
        complex_box_dimensions=complex_box_dimensions,
        forcefield_files=[
            "amber/protein.ff14SB.xml",
            "amber/tip3p_standard.xml",
            "amber/tip3p_HFE_multivalent.xml",
        ],
        platform=platform,
    )
    htf = executor.get_apo_htf()

    # Determine WT residue for naming (optional)
    wt = "WT"
    try:
        # topology proposal exists in many perses builds
        tp = getattr(htf, "_topology_proposal", None)
        if tp is not None:
            for ch in tp.old_topology.chains():
                if ch.id == mutation.chain:
                    for res in ch.residues():
                        if res.id == str(mutation.resid):
                            wt = res.name
                            raise StopIteration
    except StopIteration:
        pass
    except Exception:
        pass

    # mid = mutation_id(mutation.chain, str(mutation.resid), wt, mutation.mut_aa3, protein="BARSTAR")
    system_name = system_cfg.get("system_name", "SYSTEM")
    mid = mutation_id(mutation.chain, str(mutation.resid), wt, mutation.mut_aa3, system=system_name)

    run_dir = ensure_dir(out_root / mid / f"seed_{base_seed}")

    # write resolved config + metadata
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            {"system": system_cfg, "protocol": proto_cfg, "mutation": mutation.__dict__},
            sort_keys=False,
        )
    )
    write_metadata_json(
        run_dir / "metadata.json",
        extra={
            "mutation": mutation.__dict__,
            "platform_name": platform_name,
            "device_index": device_index,
            "smoke": smoke,
        },
    )

    # Hybrid system + positions (critical!)
    system = htf.hybrid_system
    positions = htf.hybrid_positions

    # Master storage
    forward_works_master: List[np.ndarray] = []
    reverse_works_master: List[np.ndarray] = []

    # Optional trajectory storage (saved sparsely)
    forward_eq_old, forward_eq_new = [], []
    forward_neq_old, forward_neq_new = [], []
    reverse_eq_new, reverse_eq_old = [], []
    reverse_neq_old, reverse_neq_new = [], []

    # Run replicates
    for rep in range(n_reps):
        seed = base_seed + rep

        # integrator (same as original)
        integrator = PeriodicNonequilibriumIntegrator(
            DEFAULT_ALCHEMICAL_FUNCTIONS,
            nsteps_eq,
            nsteps_neq,
            splitting,
            timestep=timestep,
            temperature=temperature,
        )

        # context
        context = openmm.Context(system, integrator, platform)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        context.setPositions(positions)
        context.setVelocitiesToTemperature(temperature, seed)

        

        # minimize (same as original; helps avoid NaNs)
        openmm.LocalEnergyMinimizer.minimize(context)

        # record actual platform used (ground truth)
        with (run_dir / "log.txt").open("a") as f:
            f.write(f"rep={rep} seed={seed} platform={context.getPlatform().getName()}\n")

        # --- Equilibrium at lambda=0 ---
        for step in range(nsteps_eq):
            t0 = time.time()
            integrator.step(1)
            if step % save_freq_eq == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                forward_eq_old.append(np.asarray(htf.old_positions(pos)))
                forward_eq_new.append(np.asarray(htf.new_positions(pos)))
                with (run_dir / "log.txt").open("a") as f:
                    f.write(f"rep={rep} eq0 step={step} dt={time.time()-t0:.4f}s\n")

        # --- Forward NEQ (0->1) ---
        forward_works = [integrator.get_protocol_work(dimensionless=True)]
        for fwd_step in range(nsteps_neq):
            t0 = time.time()
            integrator.step(1)
            forward_works.append(integrator.get_protocol_work(dimensionless=True))
            if fwd_step % save_freq_neq == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                forward_neq_old.append(np.asarray(htf.old_positions(pos)))
                forward_neq_new.append(np.asarray(htf.new_positions(pos)))
                with (run_dir / "log.txt").open("a") as f:
                    f.write(f"rep={rep} fwd step={fwd_step} dt={time.time()-t0:.4f}s\n")

        forward_works = np.asarray(forward_works, dtype=float)
        forward_works_master.append(forward_works)

        # --- Equilibrium at lambda=1 ---
        for step in range(nsteps_eq):
            t0 = time.time()
            integrator.step(1)
            if step % save_freq_eq == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                reverse_eq_new.append(np.asarray(htf.new_positions(pos)))
                reverse_eq_old.append(np.asarray(htf.old_positions(pos)))
                with (run_dir / "log.txt").open("a") as f:
                    f.write(f"rep={rep} eq1 step={step} dt={time.time()-t0:.4f}s\n")

        # --- Reverse NEQ (1->0) ---
        reverse_works = [integrator.get_protocol_work(dimensionless=True)]
        for rev_step in range(nsteps_neq):
            t0 = time.time()
            integrator.step(1)
            reverse_works.append(integrator.get_protocol_work(dimensionless=True))
            if rev_step % save_freq_neq == 0:
                pos = context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True)
                reverse_neq_old.append(np.asarray(htf.old_positions(pos)))
                reverse_neq_new.append(np.asarray(htf.new_positions(pos)))
                with (run_dir / "log.txt").open("a") as f:
                    f.write(f"rep={rep} rev step={rev_step} dt={time.time()-t0:.4f}s\n")

        reverse_works = np.asarray(reverse_works, dtype=float)
        reverse_works_master.append(reverse_works)

        # cleanup
        del context, integrator

    # Save works (dimensionless kT units)
    forward_works_master_arr = np.stack(forward_works_master, axis=0)  # (n_reps, nsteps_neq+1)
    reverse_works_master_arr = np.stack(reverse_works_master, axis=0)

    np.save(run_dir / "forward_works_master_kT.npy", forward_works_master_arr)
    np.save(run_dir / "reverse_works_master_kT.npy", reverse_works_master_arr)

    # Final work per replicate = last value
    np.save(run_dir / "forward_final_works_kT.npy", forward_works_master_arr[:, -1])
    np.save(run_dir / "reverse_final_works_kT.npy", reverse_works_master_arr[:, -1])

    # Optional: save sparse trajectories (old/new mapped) if any were collected
    if len(forward_eq_old):
        np.save(run_dir / "forward_eq_old.npy", np.asarray(forward_eq_old))
        np.save(run_dir / "forward_eq_new.npy", np.asarray(forward_eq_new))
        np.save(run_dir / "forward_neq_old.npy", np.asarray(forward_neq_old))
        np.save(run_dir / "forward_neq_new.npy", np.asarray(forward_neq_new))
        np.save(run_dir / "reverse_eq_new.npy", np.asarray(reverse_eq_new))
        np.save(run_dir / "reverse_eq_old.npy", np.asarray(reverse_eq_old))
        np.save(run_dir / "reverse_neq_old.npy", np.asarray(reverse_neq_old))
        np.save(run_dir / "reverse_neq_new.npy", np.asarray(reverse_neq_new))

    return run_dir
