#!/bin/bash
# 下载全分辨率 H&E 并转成 openslide 可读的金字塔 tiff
# 首轮 tar 因 square_016um/spatial 里几张预览图是指向 square_002um 的硬链接而退出，
# 但三个必需文件（矩阵/坐标/scalefactors）已全部落盘 —— 故本轮只补图像，且不以 tar 退出码为准。
set -u
B=https://cf.10xgenomics.com/samples/spatial-exp/3.0.0
ROOT=/path/to/systema4ST/data/visiumhd
NAMES=(Visium_HD_Human_Colon_Cancer_P1 Visium_HD_Human_Colon_Normal_P3 \
       Visium_HD_Human_Pancreas Visium_HD_Mouse_Brain Visium_HD_Mouse_Kidney \
       Visium_HD_Mouse_Embryo Visium_HD_Mouse_Small_Intestine \
       Visium_HD_Human_Colon_Cancer_P2)
N=${NAMES[$SLURM_ARRAY_TASK_ID]}
D=$ROOT/$N; mkdir -p "$D"
echo "[$N] $(date '+%F %T')"

SRC=""
for ext in btf tif tiff; do
  f="$D/${N}_tissue_image.$ext"
  [ -s "$f" ] && { SRC="$f"; echo "[$N] 已有 $ext"; break; }
  if curl -sIL -o /dev/null -w '%{http_code}' --max-time 30 "$B/$N/${N}_tissue_image.$ext" | grep -q 200; then
    echo "[$N] 下载 tissue_image.$ext"
    curl -L --retry 5 --retry-delay 10 --max-time 21600 -o "$f" "$B/$N/${N}_tissue_image.$ext" \
      && SRC="$f" && break
  fi
done
[ -z "$SRC" ] && { echo "[$N] ✘ 无全分辨率图像"; exit 1; }
ls -la "$SRC"

# 转金字塔（母项目的 P2 也是 *PYRAMIDAL* 形态；openslide 读不了扁平 BigTIFF）
OUT="$D/${N}_PYRAMIDAL.tif"
if [ -s "$OUT" ]; then
  echo "[$N] 金字塔已存在"
else
  echo "[$N] vips 转金字塔 (jpeg Q=95, tile 512)"
  vips tiffsave "$SRC" "$OUT" --tile --tile-width 512 --tile-height 512 \
       --pyramid --compression jpeg --Q 95 --bigtiff || { echo "[$N] ✘ 转换失败"; exit 1; }
fi
ls -la "$OUT"
python - <<PY
import openslide, sys
s = openslide.OpenSlide("$OUT")
print(f"[$N] openslide OK  尺寸={s.dimensions}  层数={s.level_count}  降采样={[round(d,1) for d in s.level_downsamples]}")
PY
echo "[$N] 完成 $(date '+%F %T')"
