#!/bin/bash
#SBATCH --job-name=pgxen
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pgxen_%A_%a.out
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
N=$(sed -n "${SLURM_ARRAY_TASK_ID}p" xen15.txt)
python -u src/per_gene_xen.py --name "$N"
