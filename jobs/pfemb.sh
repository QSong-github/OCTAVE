#!/bin/bash
#SBATCH -J pfemb
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 -c 6 --mem=16G -t 12:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate tfpf
cd /path/to/systema4ST; unset HF_HUB_OFFLINE
python -u pathfound_embed.py
