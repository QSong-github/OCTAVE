#!/bin/bash
#SBATCH --job-name=hestphikon
#SBATCH --qos=qsong1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export HF_HOME=/blue/qsong1/wang.qing/systema4ST/.hf
python -u src/hest_embed.py --encoder phikon
python -u src/hest_embed.py --encoder hibou_l
