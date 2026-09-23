#!/bin/bash
#SBATCH --job-name=hestcoord
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1
python -u src/hest_coords.py
