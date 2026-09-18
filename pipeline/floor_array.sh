#!/bin/bash
#SBATCH --job-name=hfloor
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/hfloor_%A_%a.out
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
C=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cohorts.txt)
python -u src/hest_floor.py --cohort "$C"
