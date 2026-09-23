#!/bin/bash
# Xenium 数据获取：配对明场 H&E（post-Xenium，同一物理切片）+ 单细胞表达
# 选样依据：① 有 <SAMPLE>_he_image.ome.tif（63/约100 个公开切片才有）
#           ② 未出现在 HEST v1.1.0（按下载链接精确映射核对，见 内部记录）
# 12 张乳腺连续切片（S1-S4 × Top/Mid/Bot）可做「同组织不同切面的 σ 重复性」——
# 这是任何单张切片都给不了的对照。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
B=https://cf.10xgenomics.com/samples/xenium
R=/path/to/project/data/xenium
NAMES=(Human_Breast_Biomarkers_S1_Top:4.0.0 Human_Breast_Biomarkers_S1_Mid:4.0.0 Human_Breast_Biomarkers_S1_Bot:4.0.0 \
       Human_Breast_Biomarkers_S2_Top:4.0.0 Human_Breast_Biomarkers_S2_Mid:4.0.0 Human_Breast_Biomarkers_S2_Bot:4.0.0 \
       Human_Breast_Biomarkers_S3_Top:4.0.0 Human_Breast_Biomarkers_S3_Mid:4.0.0 Human_Breast_Biomarkers_S3_Bot:4.0.0 \
       Human_Breast_Biomarkers_S4_Top:4.0.0 Human_Breast_Biomarkers_S4_Mid:4.0.0 Human_Breast_Biomarkers_S4_Bot:4.0.0 \
       Xenium_Prime_Cervical_Cancer_FFPE:3.0.0 Xenium_Prime_Ovarian_Cancer_FFPE_XRrun:3.0.0 \
       Xenium_Prime_Human_Ovary_FF:3.0.0 Xenium_V1_Human_Ovary_Cancer_FF:4.0.0 \
       Xenium_V1_Human_Kidney_FFPE_Protein_updated:4.0.0 Xenium_Prime_Mouse_Pup_FFPE:3.0.0)
E=${NAMES[$SLURM_ARRAY_TASK_ID]}; N=${E%%:*}; V=${E##*:}
D=$R/$N; mkdir -p "$D"; cd "$D"
echo "[$N] v$V  $(date '+%F %T')"

# ── H&E + 对齐矩阵 ──
[ -s "${N}_he_image.ome.tif" ] || curl -L --retry 5 --retry-delay 10 --max-time 21600 -o "${N}_he_image.ome.tif" "$B/$V/$N/${N}_he_image.ome.tif"
[ -s "${N}_he_imagealignment.csv" ] || curl -sL --retry 5 --max-time 600 -o "${N}_he_imagealignment.csv" "$B/$V/$N/${N}_he_imagealignment.csv"
cat "${N}_he_imagealignment.csv"

# ── 表达：只留需要的成员，随后删掉 zip（zip 8-34 GB，morphology 荧光图我们不需要）──
if [ ! -s cell_feature_matrix.h5 ]; then
  curl -L --retry 5 --retry-delay 10 --max-time 28800 -o outs.zip "$B/$V/$N/${N}_outs.zip"
  unzip -o -j outs.zip "*cell_feature_matrix.h5" "*cells.parquet" "*experiment.xenium" "*gene_panel.json" "*transcripts.parquet" 2>/dev/null
  rm -f outs.zip
fi
ls -la

# ── H&E 是否可直接被 openslide 读；不行则转金字塔 ──
python - <<PY
import openslide, subprocess, os, json
f = "${N}_he_image.ome.tif"
try:
    s = openslide.OpenSlide(f)
    print(f"[${N}] openslide 直读 OK 尺寸={s.dimensions} 层={s.level_count}")
except Exception as e:
    print(f"[${N}] openslide 读不了 ({type(e).__name__})，转金字塔")
    out = "${N}_PYRAMIDAL.tif"
    if not os.path.exists(out):
        subprocess.run(["vips","tiffsave",f,out,"--tile","--tile-width","512","--tile-height","512",
                        "--pyramid","--compression","jpeg","--Q","95","--bigtiff"], check=True)
    s = openslide.OpenSlide(out)
    print(f"[${N}] 转换后 OK 尺寸={s.dimensions} 层={s.level_count}")
PY
echo "[$N] 完成 $(date '+%F %T')"
