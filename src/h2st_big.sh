#!/bin/bash
# 在 B200(180GB) 上重跑 L4(22GB) 上 OOM 的队列。
# 不降 MAXSPOT —— 那会让 Hist2ST 与 HisToGene 看到不同的数据，破坏同协议比较。
# Hist2ST 对整片做一次前向：4000 token 的 ViT 注意力(4000²×16头×8层) + 4000×4000 稠密 GNN，
# 大队列的大切片必然超出 22GB。换更大的卡是唯一不改变方法的解法。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=$1; ARM=$2
if [ "$ARM" = "matched" ]; then
  python -u src/hist2st_hest.py --cohorts "$C" --match_steps 3200 --out "results/hist2st_matched_${C}.json"
else
  python -u src/hist2st_hest.py --cohorts "$C" --match_steps 0 --epochs 100 --out "results/hist2st_pubcfg_${C}.json"
fi
