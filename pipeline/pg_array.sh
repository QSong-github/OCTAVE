#!/bin/bash
#SBATCH --job-name=pgxen
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pgxen_%A_%a.out
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
N=$(sed -n "${SLURM_ARRAY_TASK_ID}p" xen15.txt)
python -u src/per_gene_xen.py --name "$N"
