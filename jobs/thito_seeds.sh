#!/bin/bash
#SBATCH --job-name=thitoseed
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:b200:1
#SBATCH --array=0-57
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
# 审稿意见：THItoGene 只跑了一次。补两个固定种子（1、2），加上原始运行共三次；协议与 jobs/thito_arr.sh 完全一致。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=$SLURM_ARRAY_TASK_ID; SEED=$((1 + I / 29)); read -r COH FOLD < <(sed -n "$((I % 29 + 1))p" folds.txt)
OUT=results/thitogene_${COH}_f${FOLD}_s${SEED}.json; [ -s "$OUT" ] && exit 0
echo "节点 $(hostname)  队列 $COH  折 $FOLD  seed $SEED"
python -u src/thitogene_hest.py --cohorts "$COH" --fold "$FOLD" --match_steps 3200 --seed $SEED --out "$OUT"
