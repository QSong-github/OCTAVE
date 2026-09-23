#!/bin/bash
#SBATCH --job-name=bleep
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
# 先在两个最小队列冒烟(5 epoch), 通过再铺开
python -u src/bleep_hest.py --cohorts SKCM,HCC --epochs 5 --out results/bleep_smoke.json 2>&1 | tail -25
[ -f results/bleep_smoke.json ] || { echo "❌ 冒烟未产出结果, 停止"; exit 1; }
echo "======== 闸门通过, 全队列 40 epoch ========"
python -u src/bleep_hest.py --cohorts all --epochs 40 --out results/bleep_hest.json 2>&1 | tail -40
echo "######## BLEEP DONE ########"
