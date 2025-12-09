# --------------------------------------------------------------
# File: generate_vasp_inputs.py
# --------------------------------------------------------------
"""
Genera entradas VASP (POSCAR + INCAR + KPOINTS + POTCAR link) para cada estructura
seleccionada. Debes ajustar las plantillas a tus INCAR/KPOINTS/POTCAR.
"""
import os
from ase.io import read, write




def generate_vasp_single_points(selected_dir, out_dir, dft_config):
os.makedirs(out_dir, exist_ok=True)
files = sorted([f for f in os.listdir(selected_dir) if f.endswith('.xyz') or f.endswith('.extxyz')])
for f in files:
atoms = read(os.path.join(selected_dir,f))
base = os.path.splitext(f)[0]
jobdir = os.path.join(out_dir, base)
os.makedirs(jobdir, exist_ok=True)
# write POSCAR
write(os.path.join(jobdir,'POSCAR'), atoms, format='vasp')
# copy templates
os.system(f'cp {dft_config["vasp"]["incar_template"]} {jobdir}/INCAR')
os.system(f'cp {dft_config["vasp"]["kpoints_template"]} {jobdir}/KPOINTS')
# POTCAR handling: user must create POTCARs or point to directory
# Option: symlink a POTCAR prebuilt
potcar_src = os.path.join(dft_config['vasp']['potcar_dir'], dft_config['vasp']['pseudo'])
if os.path.exists(potcar_src):
os.system(f'cp {potcar_src} {jobdir}/POTCAR')
