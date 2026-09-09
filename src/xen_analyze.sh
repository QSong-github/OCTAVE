#!/bin/bash
# Xenium 的等价 σ —— 与 9 张 Visium HD 严格同协议（片内空间块 16×16 + 汇总打分）
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
FILES=($(ls data/prepped_xen/*_bin16.h5ad))
F=${FILES[$SLURM_ARRAY_TASK_ID]}
N=$(basename "$F" _bin16.h5ad)
python -u src/nine.py --name "$N" --h5ad "$F" --emb results/emb_xen/emb_hibou_l_${N}.npy \
       --outdir xenium --label "$N"
