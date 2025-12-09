# -----------------------------
# File: scripts/select_high_uncertainty.py
# -----------------------------
"""
AUXILIARY SCRIPT
Select frames above a combined uncertainty threshold or top-k.
Usage:
  python scripts/select_high_uncertainty.py --ensemble ensemble_results.json --structures data/pool_candidates/ --out data/selected_structures/ --topk 20
"""

import argparse
import json
from pathlib import Path
from ase.io import read, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ensemble', required=True)
    p.add_argument('--structures', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--topk', type=int, default=None)
    p.add_argument('--threshold', type=float, default=None)
    args = p.parse_args()

    with open(args.ensemble) as fh:
        entries = json.load(fh)

    # entries is list of dicts expected
    scores = []
    for e in entries:
        score = e.get('E_var', 0.0) + e.get('F_var', 0.0)
        scores.append((e['index'], score))
    scores.sort(key=lambda x: x[1], reverse=True)

    if args.topk:
        chosen = [idx for idx,_ in scores[:args.topk]]
    elif args.threshold is not None:
        chosen = [idx for idx,s in scores if s >= args.threshold]
    else:
        raise RuntimeError('Provide topk or threshold')

    structures = list(read(args.structures, ':'))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for idx in chosen:
        atoms = structures[idx]
        write(str(outdir / f'frame_{idx}.extxyz'), atoms)

    print(f'Selected {len(chosen)} frames -> {outdir}')

if __name__ == '__main__':
    main()
