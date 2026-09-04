#!/bin/bash
#SBATCH -J pcount3
#SBATCH --qos=qsong1 --partition=hpg-default -c 4 --mem=32G -t 2:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_TOKEN=$(cat /blue/qsong1/wang.qing/.cache/huggingface/token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; unset HF_HUB_OFFLINE
python -u pcount3.py
# Path Foundation 在 TF 环境里另算（串行，避免两个进程同时写 encoder_params.json）
conda activate tfpf; export CUDA_VISIBLE_DEVICES=""
python -u pcount_pf.py 2>&1 | grep -v "cuda_\|cpu_feature\|rebuild TensorFlow\|TensorRT\|absl\|I0000\|W0000"
