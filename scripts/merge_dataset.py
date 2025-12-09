# -----------------------------
# File: scripts/merge_dataset.py
# -----------------------------
"""
AUXILIARY SCRIPT
Merge new DFT outputs (extxyz + json metadata) into main dataset extxyz.
Usage:
  python scripts/merge_dataset.py --new data/dft_results/ --dataset data/training_set.extxyz
"""

import argparse
from ase.io import read, write
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--new', required=True)
    p.add_argument('--dataset', required=True)
    args = p.parse_args()

    main_set = list(read(args.dataset, ':')) if Path(args.dataset).exists() else []
    new_files = sorted([p for p in Path(args.new).iterdir() if p.suffix in ('.extxyz', '.xyz')])
    for nf in new_files:
        new_atoms = list(read(str(nf), ':'))
        main_set.extend(new_atoms)
    write(args.dataset, main_set)
    print(f'Merged {len(new_files)} files into {args.dataset}')

if __name__ == '__main__':
    main()
