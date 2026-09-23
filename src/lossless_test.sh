#!/bin/bash
# 检验无损压缩能否恢复与上游项目管线的等价性。
# JPEG Q=95 只带来 1.2% 像素 MAE，却使 hibou-L 嵌入余弦掉到 0.889（已诊断）。
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
D=data/visiumhd/Visium_HD_Human_Colon_Cancer_P2
SRC=$D/Visium_HD_Human_Colon_Cancer_P2_tissue_image.btf
OUT=$D/Visium_HD_Human_Colon_Cancer_P2_LOSSLESS.tif
if [ ! -s "$OUT" ]; then
  echo "vips 无损转换 (deflate + horizontal predictor)"
  time vips tiffsave "$SRC" "$OUT" --tile --tile-width 512 --tile-height 512 \
       --pyramid --compression deflate --predictor horizontal --bigtiff
fi
ls -la "$OUT"
python -u src/hd_embed.py --validate --encoder hibou_l --val_tiff "$OUT"
