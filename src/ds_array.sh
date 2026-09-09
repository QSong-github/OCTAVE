#!/bin/bash
# 固定评测栅格（16µm）下的下游后果，全部 16 片。
# 关键控制：四个预测分箱都映射回同一批 16µm bin，故 bin 数/邻接/真值完全相同。
# §22 让评测栅格跟着预测栅格走，得到「调粗涨 48%」；本设计固定栅格后结论反向。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
FILES=($(ls data/prepped_xen/*_bin16.h5ad))
F=${FILES[$SLURM_ARRAY_TASK_ID]}
N=$(basename "$F" _bin16.h5ad)
[ -s "results/downstream_${N}.json" ] && { echo "已存"; exit 0; }
python -u src/downstream.py --name "$N"
