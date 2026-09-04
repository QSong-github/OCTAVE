#!/bin/bash
#SBATCH --job-name=histogene
#SBATCH --qos=qsong1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 用 hest conda 环境, 不是 venv_np1:
#   适配层只 import vis_model(作者模型), 不碰 dataset/predict/utils —— 那三个才需要 scprep。
#   而 hest.bench.st_dataset 的导入链需要 trident, 只有 hest 环境有。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import sys; sys.path.insert(0,'/blue/qsong1/wang.qing/systema4ST/methods/HisToGene'); from vis_model import HisToGene; print('  ✅ 作者模型可导入')" || exit 1
python -u src/histogene_hest.py --cohorts SKCM,HCC --epochs 20 --out results/histogene_smoke.json
[ -f results/histogene_smoke.json ] || { echo "❌ 冒烟未产出, 停止"; exit 1; }
echo "======== 闸门通过 ========"
python -u src/histogene_hest.py --epochs 60 --out results/histogene_hest.json
echo "######## HISTOGENE DONE ########"
