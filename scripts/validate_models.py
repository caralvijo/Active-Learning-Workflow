# -----------------------------
# File: scripts/validate_models.py
# -----------------------------
"""
AUXILIARY SCRIPT
Validate ensemble predictions vs DFT-labeled validation set.
Requires the validation extxyz to contain atoms.info['dft_energy'] and atoms.arrays['forces'] or a separate JSON.
Usage:
  python scripts/validate_models.py --models models/cycle_00/ --testset data/validation_set/ --out validation_report.json
"""

import argparse
import numpy as np
import json
from pathlib import Path
from ase.io import read
from active_learning_utils import load_nequix_model, model_to_ase_calculator


def compute_metrics(pred_energy, pred_forces, ref_energy, ref_forces):
    e_err = abs(pred_energy - ref_energy)
    f_err = np.linalg.norm(np.array(pred_forces) - np.array(ref_forces), axis=1).mean()
    return e_err, f_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', required=True)
    p.add_argument('--testset', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    model_dir = Path(args.models)
    model_paths = sorted([str(p) for p in model_dir.iterdir() if p.suffix in ('.pt', '.pth')])
    models = [load_nequix_model(Path(mp)) for mp in model_paths]

    structures = list(read(args.testset, ':'))
    report = []
    for i, atoms in enumerate(structures):
        # predict ensemble
        energies = []
        forces = []
        for m in models:
            calc = model_to_ase_calculator(m)
            atoms.set_calculator(calc)
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
        pred_E = float(np.mean(energies))
        pred_F = np.mean(forces, axis=0).tolist()
        # reference must be stored in atoms.info or in arrays
        ref_E = atoms.info.get('dft_energy', None)
        ref_F = atoms.arrays.get('forces', None)
        if ref_E is None or ref_F is None:
            raise RuntimeError('Testset atoms must contain dft_energy in atoms.info and forces in atoms.arrays')
        e_err, f_err = compute_metrics(pred_E, pred_F, ref_E, ref_F)
        report.append({'index': i, 'energy_error': float(e_err), 'forces_error': float(f_err)})

    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2)

    print('Validation complete. Wrote', args.out)

if __name__ == '__main__':
    main()
