#!/bin/bash
# 补齐 P2 的 16µm 数据 + 把 P5 图转金字塔，使 9 张片全部走统一管线
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
B=https://cf.10xgenomics.com/samples/spatial-exp/3.0.0
R=/path/to/systema4ST/data/visiumhd
cd /path/to/systema4ST

N=Visium_HD_Human_Colon_Cancer_P2
if [ ! -d "$R/$N/binned_outputs/square_016um" ]; then
  echo "下载 P2 binned_outputs"
  curl -sL --retry 5 --max-time 14400 "$B/$N/${N}_binned_outputs.tar.gz" \
    | tar -xzf - -C "$R/$N" --wildcards '*square_016um*' || true
fi
ls "$R/$N/binned_outputs/square_016um/" 2>/dev/null

for N in Visium_HD_Human_Colon_Cancer_P5; do
  SRC=$R/$N/${N}_tissue_image.btf; OUT=$R/$N/${N}_PYRAMIDAL.tif
  if [ ! -s "$OUT" ]; then
    echo "vips 转金字塔 $N"
    vips tiffsave "$SRC" "$OUT" --tile --tile-width 512 --tile-height 512 \
         --pyramid --compression jpeg --Q 95 --bigtiff
  fi
  python - <<PY
import openslide
s = openslide.OpenSlide("$OUT")
print(f"[$N] openslide OK 尺寸={s.dimensions} 层={s.level_count}")
PY
done

echo "=== 构建 P2/P5 的 h5ad ==="
python -u src/hd_prep.py --name Visium_HD_Human_Colon_Cancer_P2
python -u src/hd_prep.py --name Visium_HD_Human_Colon_Cancer_P5
ls -la data/prepped/
