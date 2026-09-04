#!/bin/bash
# Xenium 嵌入：按样本换算 ctx_px 使【物理视野】锁定在 61.4 µm
# —— 与 9 张 Visium HD 完全一致。FOV 律（σ = 63 + 0.91·FOV）说 σ 由物理视野决定，
#    视野不锁死，跨平台/跨样本的 σ 就不可比。
#    Xenium_V1_Human_Ovary_Cancer_FF 的 H&E 是双倍放大扫描（7.30 vs 3.65 px/µm），
#    照搬 224 会得到 30.7µm 视野，跑完不报错，只会给出一个偏细的 σ 被误读成组织学差异。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
FILES=($(ls data/prepped_xen/*_bin16.h5ad))
F=${FILES[$SLURM_ARRAY_TASK_ID]}
N=$(basename "$F" _bin16.h5ad)
OUT=results/emb_xen/emb_hibou_l_${N}.npy
[ -s "$OUT" ] && { echo "[$N] 已存在"; exit 0; }
CTX=$(python -c "
import anndata as ad
a = ad.read_h5ad('$F', backed='r')
p = float(a.uns['px_per_um'])
print(int(round(61.4 * p)))")
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif)
echo "[$N] ctx_px=$CTX（物理视野 61.4 µm）  图=$(basename $TIF)"
mkdir -p results/emb_xen
python -u src/hd_embed.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder hibou_l --ctx_px "$CTX" --batch 256
