#!/bin/bash
#SBATCH -J probecfg
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 4 --mem=24G -t 3:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_TOKEN=$(cat /path/to/.cache/huggingface/token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; unset HF_HUB_OFFLINE
python -u probe_cfg.py "$@"
