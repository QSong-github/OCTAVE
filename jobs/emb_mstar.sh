#!/bin/bash
#SBATCH -J emb_mstar
#SBATCH --qos=YOUR_QOS --partition=hpg-b200,hpg-rtx6000,hpg-turin --gres=gpu:1 --array=0-0 -c 6 --mem=16G -t 24:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export HF_TOKEN=$(cat /path/to/.cache/huggingface/token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; unset HF_HUB_OFFLINE
A=(mstar); X=${A[$SLURM_ARRAY_TASK_ID]}
[ "$(ls results/hest_emb/*_${X}.npz 2>/dev/null | wc -l)" -ge 72 ] && exit 0
python -u src/hest_embed_v2.py --encoder "$X" --batch 128
