# --------------------------------------------------------------
# File: utils.py
# --------------------------------------------------------------
from ase import Atoms
from ase.io import read, write
import os, json, glob
import numpy as np




def ensure_dirs(*dirs):
for d in dirs:
os.makedirs(d, exist_ok=True)




def read_xyz_or_extxyz(path):
return read(path)




def save_structure_extxyz(atoms, path, properties=None):
# properties: dict to save metadata
write(path, atoms, format='extxyz')
if properties:
meta_path = path + '.json'
with open(meta_path, 'w') as f:
json.dump(properties, f, indent=2)


# small helper for datetime stamps
from datetime import datetime


def now_iso():
return datetime.utcnow().isoformat() + 'Z'
