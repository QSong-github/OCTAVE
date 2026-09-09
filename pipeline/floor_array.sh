#!/bin/bash
#SBATCH --job-name=hfloor
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/hfloor_%A_%a.out
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
C=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cohorts.txt)
python -u src/hest_floor.py --cohort "$C"
