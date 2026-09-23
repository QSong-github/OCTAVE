#!/bin/bash
#SBATCH --job-name=thitoseed
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --array=0-57
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/path/to/project/logs/%x_%A_%a.out
# 复核意见：THItoGene 只跑了一次。补两个固定种子（1、2），加上原始运行共三次；协议与 jobs/thito_arr.sh 完全一致。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=$SLURM_ARRAY_TASK_ID; SEED=$((1 + I / 29)); read -r COH FOLD < <(sed -n "$((I % 29 + 1))p" folds.txt)
OUT=results/thitogene_${COH}_f${FOLD}_s${SEED}.json; [ -s "$OUT" ] && exit 0
echo "节点 $(hostname)  队列 $COH  折 $FOLD  seed $SEED"
python -u src/thitogene_hest.py --cohorts "$COH" --fold "$FOLD" --match_steps 3200 --seed $SEED --out "$OUT"
