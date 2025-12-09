# -----------------------------
# File: scripts/run_committee_inference.py
# -----------------------------
"""
AUXILIARY SCRIPT
Run committee inference with Nequix models over a trajectory or set of structures.
Usage:
  python scripts/run_committee_inference.py --models models/cycle_00/ --structures data/pool_candidates/ --out ensemble_results.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read, write

# Attempt to import Nequix ASE wrapper; fallback will raise early if missing
try:
    from nequixase import NequixCalculator
except Exception:
    NequixCalculator = None

from active_learning_utils import load_nequix_model, model_to_ase_calculator


def evaluate(models, atoms):
    energies = []
    forces = []
    for m in models:
        calc = model_to_ase_calculator(m)
        atoms.set_calculator(calc)
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
    E = np.array(energies)
    F = np.array(forces)
    return {
        'E_mean': float(E.mean()),
        'E_var': float(E.var()),
        'F_mean': F.mean(axis=0).tolist(),
        'F_var': float(F.var(axis=0).mean())
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', required=True, help='Directory with model files or list comma-separated')
    p.add_argument('--structures', required=True, help='Trajectory or folder of structures')
    p.add_argument('--out', default='ensemble_results.json')
    args = p.parse_args()

    model_paths = []
    mpath = Path(args.models)
    if mpath.is_dir():
        model_paths = sorted([str(p) for p in mpath.iterdir() if p.suffix in ('.pt', '.pth')])
    else:
        model_paths = [s.strip() for s in args.models.split(',')]

    # load models once (may be path or object depending on loader implementation)
    models = []
    for mp in model_paths:
        models.append(load_nequix_model(Path(mp)))

    # read structures
    structures = []
    s = Path(args.structures)
    if s.is_dir():
        # load supported files
        for f in sorted(s.iterdir()):
            if f.suffix in ('.xyz', '.extxyz', '.vasp', '.traj'):
                structures.extend(read(str(f), ':'))
    else:
        structures = read(str(s), ':')

    results = []
    for i, atoms in enumerate(structures):
        info = evaluate(models, atoms)
        info['index'] = i
        results.append(info)

    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)

if __name__ == '__main__':
    main()
