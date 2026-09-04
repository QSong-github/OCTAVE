#!/bin/bash
#SBATCH -J pfembcpu
#SBATCH --qos=qsong1 --partition=hpg-default --array=0-7 -c 8 --mem=24G -t 24:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate tfpf
cd /blue/qsong1/wang.qing/systema4ST; unset HF_HUB_OFFLINE
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=1
export SHARD=$SLURM_ARRAY_TASK_ID NSHARD=8
python -u pathfound_embed.py 2>&1 | grep -v "cuda_\|cpu_feature_guard\|rebuild TensorFlow\|TensorRT\|absl\|I0000\|W0000"
