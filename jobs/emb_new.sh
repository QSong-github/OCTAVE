#!/bin/bash
#SBATCH --job-name=embnew
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
# 这六个已在本地 HF 缓存（TRIDENT_ENC），encoder() 早有分支，只是从未为 HEST 抽过嵌入。
# 每个编码器一个任务；嵌入按样本落盘可续跑。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1
ENC=(uni_v1 virchow conch_v1 keep openmidnight hibou_l)
E=${ENC[$SLURM_ARRAY_TASK_ID]}
echo "节点 $(hostname)  编码器 $E"; nvidia-smi --query-gpu=name --format=csv,noheader
python -u src/hest_embed_v2.py --encoder "$E" --batch 128
