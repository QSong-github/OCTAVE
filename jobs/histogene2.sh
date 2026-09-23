#!/bin/bash
#SBATCH --job-name=histogene2
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
# 按作者 tutorial.ipynb 的发表配置重跑: n_layers=8, lr=1e-5, epochs=100
# 首轮用了函数签名默认值(4 / 1e-4 / 60), lr 高 10 倍 → 欠拟合, 结果作废。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
mv results/histogene_hest.json results/histogene_wrongcfg.json 2>/dev/null
python -u src/histogene_hest.py --epochs 100 --n_layers 8 --lr 1e-5 --out results/histogene_hest.json
echo "######## HISTOGENE2 DONE ########"
