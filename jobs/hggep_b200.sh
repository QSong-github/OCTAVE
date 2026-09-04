#!/bin/bash
#SBATCH --job-name=hggepb2
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# L4(22GB) 溢出：作者的 build_adj_hypergraph 对每个节点算全图欧氏距离，
# 他们在 HER2ST 上每片 300-700 点，本协议是 MAXSPOT=4000。改 MAXSPOT 会破坏
# 与其它方法的可比性，所以换大卡而不是改协议。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u src/hggep_hest.py --cohorts HCC,LUNG --match_steps 3200 \
    --out results/hggep_smoke.json
echo "######## HGGEP B200 SMOKE DONE ########"
