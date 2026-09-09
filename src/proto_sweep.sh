#!/bin/bash
# 协议效应扫描：同一片、同一方法、同一编码器，只改空间划分的粗细。
# 块栅格 g 决定训练/测试块的物理尺寸：g 越小，块越大、训练与测试的空间隔离越强。
# FINDINGS §2.4 在 Visium HD 两片上测到协议效应 0.20–0.30 PCC，是方法间差异(≈0.03)的 10 倍；
# 本扫描把它扩到 Xenium 16 片 × 4 种划分，检验该结论是否稳健。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
FILES=($(ls data/prepped_xen/*_bin16.h5ad))
GRIDS=(16 8 4 2)
N=${#FILES[@]}
FI=$((SLURM_ARRAY_TASK_ID % N))
GI=$((SLURM_ARRAY_TASK_ID / N))
F=${FILES[$FI]}; G=${GRIDS[$GI]}
NAME=$(basename "$F" _bin16.h5ad)
python -u src/nine.py --name "${NAME}" --h5ad "$F" \
  --emb results/emb_xen/emb_hibou_l_${NAME}.npy \
  --outdir proto_g${G} --label "${NAME} g=${G}" --grid "$G" --tag "_g${G}"
