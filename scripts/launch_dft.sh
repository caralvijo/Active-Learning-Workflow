# -----------------------------
# File: scripts/launch_dft.sh
# -----------------------------
"""
AUXILIARY SCRIPT (bash)
Simple launcher to convert extxyz to POSCAR and submit VASP jobs via sbatch.
Usage:
  bash scripts/launch_dft.sh data/selected_structures/ 16
Arguments:
  $1 = input folder with .extxyz files
  $2 = nprocs per job
"""

#!/usr/bin/env bash
set -e
STRUCT_DIR=$1
NPROC=${2-16}
OUTROOT=dft_runs
mkdir -p ${OUTROOT}

for f in ${STRUCT_DIR}/*.extxyz; do
  name=$(basename ${f%.*})
  work=${OUTROOT}/${name}
  mkdir -p ${work}
  # convert to POSCAR using ase
  python - <<PY
from ase.io import read, write
atoms = read('${f}')
write('${work}/POSCAR', atoms, format='vasp')
PY
  # copy templates
  cp templates/INCAR ${work}/INCAR || true
  cp templates/KPOINTS ${work}/KPOINTS || true
  cp templates/POTCAR ${work}/POTCAR || true

  # create sbatch script
  cat > ${work}/run_vasp.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vasp_${name}
#SBATCH --nodes=1
#SBATCH --ntasks=${NPROC}
#SBATCH --time=48:00:00
#SBATCH --output=vasp_${name}.out

module load vasp
mpirun -np ${NPROC} vasp_std > vasp.out
EOF

  (cd ${work} && sbatch run_vasp.sbatch)
  echo "Submitted ${name}"
done
