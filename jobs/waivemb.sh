#!/bin/bash
#SBATCH -J waivemb
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1 --array=0-1 -c 6 --mem=96G -t 24:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -eu
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
E=(phaet mascaret); N=${E[$SLURM_ARRAY_TASK_ID]}
echo "节点 $(hostname)  编码器 $N"
python -u src/hest_embed_v2.py --encoder "$N" --batch 128
