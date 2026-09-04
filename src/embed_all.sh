#!/bin/bash
# 9 张片的 hibou_l 嵌入 —— 全部来自 10x 原始数据，统一管线（ctx_px=224, grid=1, level 0）
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
NAMES=(Visium_HD_Human_Colon_Cancer_P1 Visium_HD_Human_Colon_Cancer_P2 Visium_HD_Human_Colon_Cancer_P5 \
       Visium_HD_Human_Colon_Normal_P3 Visium_HD_Human_Pancreas Visium_HD_Mouse_Brain \
       Visium_HD_Mouse_Kidney Visium_HD_Mouse_Embryo Visium_HD_Mouse_Small_Intestine)
N=${NAMES[$SLURM_ARRAY_TASK_ID]}
ENC=${ENC:-hibou_l}
OUT=results/emb9/emb_${ENC}_${N}.npy
[ -s "$OUT" ] && { echo "[$N] 已存在，跳过"; exit 0; }
TIF=data/visiumhd/$N/${N}_PYRAMIDAL.tif
[ -s "$TIF" ] || { echo "[$N] ✘ 缺金字塔图"; exit 1; }
mkdir -p results/emb9
python -u src/hd_embed.py --h5ad data/prepped/${N}_16um.h5ad --tiff "$TIF" \
       --out "$OUT" --encoder "$ENC" --ctx_px 224 --batch 256
