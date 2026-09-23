#!/bin/bash
#SBATCH --job-name=heclipsmk
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
# 冒烟：只跑两个最小队列。看三件事——
#  ① torch_geometric / HypergraphConv 能否导入
#  ② 是否出现 §32 那种「整片恰好 0.0000」（孤立点 → NaN 的征兆）
#  ③ 单折用时，用来估全量墙钟
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u src/heclip_hest.py --cohorts HCC,LUNG \
    --out results/heclip_smoke.json
echo "######## HECLIP SMOKE DONE ########"
