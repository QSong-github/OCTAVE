#!/bin/bash
# 多尺度分箱的等价 σ：邻接半径随分箱边长走，其余协议与 16µm 完全一致
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
FILES=($(ls data/prepped_xen/*_bin8.h5ad data/prepped_xen/*_bin32.h5ad data/prepped_xen/*_bin64.h5ad))
F=${FILES[$SLURM_ARRAY_TASK_ID]}
BASE=$(basename "$F" .h5ad); N=${BASE%_bin*}; BIN=${BASE##*_bin}
E=results/emb_xen/emb_hibou_l_${N}_bin${BIN}.npy
[ -s "$E" ] || { echo "缺嵌入 $E"; exit 0; }
python -u src/nine.py --name "${N}_bin${BIN}" --h5ad "$F" --emb "$E" \
       --outdir xenium_multi --label "${N} @ ${BIN}um" --pitch_um "$BIN"
