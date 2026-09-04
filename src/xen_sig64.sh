#!/bin/bash
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
FILES=($(ls data/prepped_xen/*_bin64.h5ad))
F=${FILES[$SLURM_ARRAY_TASK_ID]}
N=$(basename "$F" _bin64.h5ad)
python -u src/nine.py --name "${N}_bin64" --h5ad "$F" \
  --emb results/emb_xen/emb_hibou_l_${N}_bin64.npy \
  --outdir xenium_multi --label "${N} @ 64um" --pitch_um 64
