#!/bin/bash
#SBATCH -J fetchxen22
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-21 -c 1 --mem=16G --nice=50000 -t 10:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# re-fetch the 22 Xenium regions from the 10x public CDN, then bin to 16 um with the unchanged src/xen_prep.py.
# Same files and steps as src/fetch_xenium.sh, except transcripts.parquet (not used).
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
B=https://cf.10xgenomics.com/samples/xenium
R=/path/to/project/data/xenium
NAMES=(Human_Breast_Biomarkers_S1_Top:4.0.0 Human_Breast_Biomarkers_S1_Mid:4.0.0 Human_Breast_Biomarkers_S1_Bot:4.0.0 \
       Human_Breast_Biomarkers_S2_Top:4.0.0 Human_Breast_Biomarkers_S2_Mid:4.0.0 Human_Breast_Biomarkers_S2_Bot:4.0.0 \
       Human_Breast_Biomarkers_S3_Top:4.0.0 Human_Breast_Biomarkers_S3_Mid:4.0.0 Human_Breast_Biomarkers_S3_Bot:4.0.0 \
       Human_Breast_Biomarkers_S4_Top:4.0.0 Human_Breast_Biomarkers_S4_Mid:4.0.0 Human_Breast_Biomarkers_S4_Bot:4.0.0 \
       Xenium_Prime_Cervical_Cancer_FFPE:3.0.0 Xenium_Prime_Ovarian_Cancer_FFPE_XRrun:3.0.0 \
       Xenium_V1_Human_Ovary_Cancer_FF:4.0.0 Xenium_V1_Human_Kidney_FFPE_Protein_updated:4.0.0 \
       Xenium_V1_Human_Lung_Cancer_FFPE:3.0.0 Xenium_Prime_Human_Lung_Cancer_FFPE:3.0.0 Xenium_Prime_Breast_Cancer_FFPE:3.0.0 \
       Xenium_Prime_Human_Prostate_FFPE:3.0.0 Xenium_Prime_Human_Skin_FFPE:3.0.0 Xenium_Prime_Human_Lymph_Node_Reactive_FFPE:3.0.0)
E=${NAMES[$SLURM_ARRAY_TASK_ID]}; N=${E%%:*}; V=${E##*:}
D=$R/$N; mkdir -p "$D"; cd "$D"
echo "[$N] v$V  $(date '+%F %T')"
[ -s "${N}_he_image.ome.tif" ] || curl -L --fail --retry 5 --retry-delay 10 --max-time 21600 -o "${N}_he_image.ome.tif" "$B/$V/$N/${N}_he_image.ome.tif"
[ -s "${N}_he_imagealignment.csv" ] || curl -sL --fail --retry 5 --max-time 600 -o "${N}_he_imagealignment.csv" "$B/$V/$N/${N}_he_imagealignment.csv"
if [ ! -s cell_feature_matrix.h5 ]; then
  # resume a cut transfer (curl error 18 left one region empty on the first run); check the zip directory before extracting
  for k in 1 2 3 4 5 6; do
    curl -L --fail -C - --retry 5 --retry-all-errors --retry-delay 20 --max-time 28800 -o outs.zip "$B/$V/$N/${N}_outs.zip" && unzip -l outs.zip >/dev/null 2>&1 && break
    echo "[$N] download attempt $k incomplete, resuming"; sleep 30
  done
  unzip -o -j outs.zip "*cell_feature_matrix.h5" "*cells.parquet" "*experiment.xenium" "*gene_panel.json" 2>/dev/null
  [ -s cell_feature_matrix.h5 ] && rm -f outs.zip
fi
ls -la
python - <<PY
import openslide, subprocess, os
f = "${N}_he_image.ome.tif"
try:
    s = openslide.OpenSlide(f)
    print(f"[${N}] openslide reads it directly, size={s.dimensions} levels={s.level_count}")
except Exception as e:
    print(f"[${N}] openslide cannot read it ({type(e).__name__}); converting to a pyramid")
    out = "${N}_PYRAMIDAL.tif"
    if not os.path.exists(out):
        subprocess.run(["vips","tiffsave",f,out,"--tile","--tile-width","512","--tile-height","512",
                        "--pyramid","--compression","jpeg","--Q","95","--bigtiff"], check=True)
    s = openslide.OpenSlide(out)
    print(f"[${N}] converted, size={s.dimensions} levels={s.level_count}")
PY
cd /path/to/project
OUT=data/prepped_xen/${N}_bin16.h5ad
[ -s "$OUT" ] || python -u src/xen_prep.py --name "$N"
python - <<PY
import anndata as ad, numpy as np, glob
a = ad.read_h5ad("$OUT", backed="r"); n = a.n_obs
fs = sorted(glob.glob("results/emb_xen/emb_*_${N}.npy"))
bad = [f for f in fs if np.load(f, mmap_mode="r").shape[0] != n]
print(f"[${N}] CHECK bins={n} embeddings_checked={len(fs)} row_mismatches={len(bad)}", bad[:3])
PY
echo "[$N] done $(date '+%F %T')"
