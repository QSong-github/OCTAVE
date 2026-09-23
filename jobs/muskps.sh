#!/bin/bash
#SBATCH -J muskps
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH -c 4 --mem=48G -t 8:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python -u src/hest_effres_ps.py --encoder musk --out results/hest_effres_ps_musk.json
