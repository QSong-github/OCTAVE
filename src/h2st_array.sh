#!/bin/bash
# Hist2ST 按队列并行。两个配置各跑一遍（与 HisToGene 同等待遇）：
#   arm=matched : match_steps=3200，步数对齐作者 HER2ST 的 32片×100ep
#   arm=pubcfg  : epochs=100，作者原始 epoch 数（步数 = 片数×100）
# 两臂之差本身有信息：若 pubcfg（步数少得多）反而更好，说明瓶颈是片数不是步数。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
COH=(SKCM HCC LUNG PAAD COAD READ IDC LYMPH_IDC PRAD CCRCC)
N=${#COH[@]}
CI=$((SLURM_ARRAY_TASK_ID % N))
AI=$((SLURM_ARRAY_TASK_ID / N))
C=${COH[$CI]}
if [ "$AI" -eq 0 ]; then
  python -u src/hist2st_hest.py --cohorts "$C" --match_steps 3200 \
    --out "results/hist2st_matched_${C}.json"
else
  python -u src/hist2st_hest.py --cohorts "$C" --match_steps 0 --epochs 100 \
    --out "results/hist2st_pubcfg_${C}.json"
fi
