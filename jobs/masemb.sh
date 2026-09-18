#!/bin/bash
#SBATCH -J masemb
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1 -c 6 --mem=96G -t 24:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
python -u src/hest_embed_v2.py --encoder mascaret --batch 128
