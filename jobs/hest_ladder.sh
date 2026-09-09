#!/bin/bash
#SBATCH --job-name=hestladder
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=16
python -u src/hest_ladder.py --tmax 1024 --reps 3 --norm log1p
