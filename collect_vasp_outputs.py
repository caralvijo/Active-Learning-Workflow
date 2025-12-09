# --------------------------------------------------------------
# File: collect_vasp_outputs.py
# --------------------------------------------------------------
"""
Parsea los resultados de VASP para cada jobdir y escribe un extxyz + metadata json
conteniendo energía, fuerzas y magmoms (si están).
"""
import os
from ase.io import read, write
import json




def parse_vasp_job(jobdir):
# intento con ase.read('vasprun.xml') o 'OUTCAR' (ASE soporta vasprun.xml)
vasprun = os.path.join(jobdir, 'vasprun.xml')
outcar = os.path.join(jobdir, 'OUTCAR')
atoms = None
meta = {}
try:
if os.path.exists(vasprun):
atoms = read(vasprun, index=-1)
elif os.path.exists(outcar):
atoms = read(outcar, index=-1)
except Exception as e:
print('Error parsing', jobdir, e)
if atoms is not None:
E = atoms.get_potential_energy()
F = atoms.get_forces().tolist()
mag = None
# ASE may store magmom info in atoms.get_magnetic_moments()
try:
mag = atoms.get_magnetic_moments().tolist()
except Exception:
mag = None
meta = {'energy': float(E), 'forces': F, 'magmoms': mag, 'parsed': True}
else:
meta = {'parsed': False}
return atoms, meta




def collect_all_results(dft_submit_dir, labeled_dir):
os.makedirs(labeled_dir, exist_ok=True)
jobdirs = [os.path.join(dft_submit_dir,d) for d in os.listdir(dft_submit_dir) if os.path.isdir(os.path.join(dft_submit_dir,d))]
for jd in jobdirs:
atoms, meta = parse_vasp_job(jd)
base = os.path.basename(jd)
if meta.get('parsed'):
outpath = os.path.join(labeled_dir, base + '.extxyz')
write(outpath, atoms, format='extxyz')
with open(outpath + '.json','w') as fh:
json.dump(meta, fh, indent=2)
