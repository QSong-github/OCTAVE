#!/bin/bash
#SBATCH --job-name=histogene3
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 第三轮: 保持发表的 n_layers=8 / lr=1e-5, 但按队列调 epoch 使梯度步数统一到 3200
# (= 作者 HER2ST 的 32片×100ep)。这是针对"步数"这一已识别混杂的单一调整, 不是调参。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
echo "节点 $(hostname)"
mv results/histogene_hest.json results/histogene_pubcfg.json 2>/dev/null
python -u src/histogene_hest.py --n_layers 8 --lr 1e-5 --match_steps 3200 \
    --out results/histogene_matched.json
echo "######## HISTOGENE3 DONE ########"
