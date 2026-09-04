#!/bin/bash
#SBATCH -J pfcheck
#SBATCH --qos=qsong1 --partition=hpg-default -c 4 --mem=16G -t 0:30:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate tfpf
cd /blue/qsong1/wang.qing/systema4ST; unset HF_HUB_OFFLINE; export CUDA_VISIBLE_DEVICES=""
python -u pfcheck.py 2>&1 | grep -v "cuda_\|cpu_feature_guard\|rebuild TensorFlow\|TensorRT\|^$"
