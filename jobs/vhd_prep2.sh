#!/bin/bash
#SBATCH -J vhdprep
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-1 -c 4 --mem=64G -t 8:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# Visium HD P2/P5 -> pyramidal TIFF (as src/fetch_img.sh) and 16 um h5ad (src/hd_prep.py). Idempotent.
# module load libvips puts its own python 3.5 first on PATH, so vips runs in a subshell. Conda activation goes through jobs/act.sh.
# The TIFFs from the first run are kept, so the check also reads level-0 tiles.
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
NAMES=(Visium_HD_Human_Colon_Cancer_P2 Visium_HD_Human_Colon_Cancer_P5)
N=${NAMES[$SLURM_ARRAY_TASK_ID]}; D=data/visiumhd/$N
[ -s $D/${N}_PYRAMIDAL.tif ] || ( module load libvips; vips tiffsave $D/${N}_tissue_image.btf $D/${N}_PYRAMIDAL.tif --tile --tile-width 512 --tile-height 512 --pyramid --compression jpeg --Q 95 --bigtiff )
python - <<PY
import openslide; s = openslide.OpenSlide("$D/${N}_PYRAMIDAL.tif"); W, H = s.dimensions
for x, y in ((W // 2, H // 2), (W - 1024, H - 1024)): s.read_region((x, y), 0, (512, 512)).load()
s.read_region((0, 0), s.level_count - 1, s.level_dimensions[-1]).load()
print("[$N] openslide OK", s.dimensions, s.level_count, s.level_dimensions[-1])
PY
[ -s data/prepped/${N}_16um.h5ad ] || python -u src/hd_prep.py --name $N
ls -la data/prepped/${N}_16um.h5ad
