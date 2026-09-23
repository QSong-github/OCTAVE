#!/bin/bash
#SBATCH -J thitosmoke
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 -c 6 --mem=32G -t 4:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1
python -u src/thitogene_hest.py --cohorts SKCM --fold 0 --out results/thito_smoke.json
