#!/bin/bash
# 下载新增 Visium HD 样本（10x 官方直链 v3.0.0，无需账号）
# 流式解包只保留 square_016um —— binned_outputs.tar.gz 含 2/8/16µm 三档，我们只用 16µm
set -u
B=https://cf.10xgenomics.com/samples/spatial-exp/3.0.0
ROOT=/path/to/systema4ST/data/visiumhd
NAMES=(Visium_HD_Human_Colon_Cancer_P1 Visium_HD_Human_Colon_Normal_P3 \
       Visium_HD_Human_Pancreas Visium_HD_Mouse_Brain Visium_HD_Mouse_Kidney \
       Visium_HD_Mouse_Embryo Visium_HD_Mouse_Small_Intestine)
N=${NAMES[$SLURM_ARRAY_TASK_ID]}
D=$ROOT/$N
mkdir -p "$D"
echo "[$N] 开始 $(date '+%F %T')"

# ── 1. 16µm 分箱数据（流式，只留 square_016um）──
if [ -d "$D/binned_outputs/square_016um" ]; then
  echo "[$N] 16µm 已存在，跳过"
else
  echo "[$N] 流式解包 binned_outputs → square_016um"
  curl -sL --retry 5 --retry-delay 10 --max-time 14400 \
       "$B/$N/${N}_binned_outputs.tar.gz" \
    | tar -xzf - -C "$D" --wildcards '*square_016um*' || { echo "[$N] 分箱数据失败"; exit 1; }
fi
du -sh "$D/binned_outputs" 2>/dev/null

# ── 2. 全分辨率 H&E（btf 或 tif，逐个试）──
IMG=""
for ext in tissue_image.btf tissue_image.tif tissue_image.tiff; do
  if [ -s "$D/${N}_$ext" ]; then IMG="$D/${N}_$ext"; echo "[$N] 图像已存在 $ext"; break; fi
  if curl -sIL -o /dev/null -w '%{http_code}' --max-time 30 "$B/$N/${N}_$ext" | grep -q 200; then
    echo "[$N] 下载 $ext"
    curl -sL --retry 5 --retry-delay 10 --max-time 14400 -o "$D/${N}_$ext" "$B/$N/${N}_$ext" \
      && IMG="$D/${N}_$ext" && break
  fi
done
[ -z "$IMG" ] && { echo "[$N] ⚠ 未找到全分辨率图像"; }

# ── 3. spatial（含 scalefactors，用于像素↔µm 换算）──
[ -s "$D/spatial.tar.gz" ] || curl -sL --retry 5 --max-time 1800 -o "$D/spatial.tar.gz" "$B/$N/${N}_spatial.tar.gz"
tar -xzf "$D/spatial.tar.gz" -C "$D" 2>/dev/null

echo "[$N] 完成 $(date '+%F %T')"
ls -la "$D" | head -20
find "$D/binned_outputs/square_016um" -maxdepth 2 2>/dev/null | head -12
