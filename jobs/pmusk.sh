#!/bin/bash
#SBATCH -J pmusk
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1 -c 6 --mem=64G -t 01:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
pip install --no-cache-dir --no-deps einops 2>&1 | tail -1
python -u probe_musk.py
