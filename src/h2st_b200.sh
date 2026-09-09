#!/bin/bash
# 忠实实现作者配置（mse + 0.25×ZINB + 0.5×自蒸馏）后显存需求约 6 倍：
# bake=5 要额外 5 次前向并保留激活。22GB 的 L4 连最小队列都装不下 ⇒ 全部走 B200。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COH=(SKCM HCC LUNG PAAD COAD READ IDC LYMPH_IDC PRAD CCRCC)
N=${#COH[@]}
C=${COH[$((SLURM_ARRAY_TASK_ID % N))]}
if [ $((SLURM_ARRAY_TASK_ID / N)) -eq 0 ]; then
  python -u src/hist2st_hest.py --cohorts "$C" --match_steps 3200 --out "results/h2st2_matched_${C}.json"
else
  python -u src/hist2st_hest.py --cohorts "$C" --match_steps 0 --epochs 100 --out "results/h2st2_pubcfg_${C}.json"
fi
