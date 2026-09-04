#!/bin/bash
#SBATCH --job-name=probe3
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
python -u probe3.py
