#!/bin/bash
#SBATCH --job-name=hggep
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:b200:1
#SBATCH --array=0-28
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
# 逐折并行：29 折各一个任务，各写各的 JSON，跑完再合并。
# 不加 %N 限流；QOS 的 gres/gpu=15 已是唯一的并发上限。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
read -r COH FOLD < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" folds.txt)
echo "节点 $(hostname)  队列 $COH  折 $FOLD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u src/thitogene_hest.py --cohorts "$COH" --fold "$FOLD" --match_steps 3200 \
    --out "results/thitogene_${COH}_f${FOLD}.json"
