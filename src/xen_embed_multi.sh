#!/bin/bash
# Xenium 多尺度分箱的嵌入：物理视野仍锁 61.4 µm（与 bin 尺度无关）
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
FILES=($(ls data/prepped_xen/*_bin8.h5ad data/prepped_xen/*_bin32.h5ad data/prepped_xen/*_bin64.h5ad))
F=${FILES[$SLURM_ARRAY_TASK_ID]}
BASE=$(basename "$F" .h5ad); N=${BASE%_bin*}; BIN=${BASE##*_bin}
OUT=results/emb_xen/emb_hibou_l_${N}_bin${BIN}.npy
[ -s "$OUT" ] && { echo "已存在"; exit 0; }
CTX=$(python -c "
import anndata as ad; a=ad.read_h5ad('$F',backed='r'); print(int(round(61.4*float(a.uns['px_per_um']))))")
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif)
echo "[$N @ ${BIN}µm] ctx_px=$CTX"
python -u src/hd_embed.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder hibou_l --ctx_px "$CTX" --batch 256
