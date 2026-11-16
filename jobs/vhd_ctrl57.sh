#!/bin/bash
#SBATCH -J vhdc57
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-1 -c 8 --mem=96G -t 24:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# Visium HD partition controls for all 57 encoders (oracle, coordinate, random matched, means from training blocks). Idempotent.
set -e
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
SEC=(Visium_HD_Human_Colon_Cancer_P2 Visium_HD_Human_Colon_Cancer_P5)
python -u vhd_ctrl57.py --name ${SEC[$SLURM_ARRAY_TASK_ID]} --mode cheap
