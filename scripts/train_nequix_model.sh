# -----------------------------
# File: scripts/train_nequix_model.sh
# -----------------------------
"""
AUXILIARY SCRIPT (bash)
Template to train a Nequix model. Customize based on your Nequix training CLI.
Usage:
  bash scripts/train_nequix_model.sh data/training_set.extxyz models/cycle_01/model_1
"""

#!/usr/bin/env bash
set -e
DATA=$1
OUTDIR=$2
SEED=${3-1234}
mkdir -p ${OUTDIR}

# Example placeholder command - replace with your actual training command
nequix-train --train ${DATA} --out ${OUTDIR} --seed ${SEED} --epochs 200

# End of file
