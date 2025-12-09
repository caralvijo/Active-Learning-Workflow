# --------------------------------------------------------------
# File: run_vasp.slurm (plantilla)
# --------------------------------------------------------------
"""
#!/bin/bash
#SBATCH --job-name=vasp_sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --output=slurm-%j.out


module load vasp
srun vasp_std
"""
