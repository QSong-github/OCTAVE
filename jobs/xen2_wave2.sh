#!/bin/bash
#SBATCH -J xen2w2
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-5 -c 8 --mem=96G -t 16:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# 第二批 Xenium 第二波：对照、可靠性上限、归纳式、归因、分割敏感、K-means 种子、消融、逐基因、算子敏感（与 16 区域同参数）。幂等。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
NAMES=(Xenium_V1_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE)
N=${NAMES[$SLURM_ARRAY_TASK_ID]}
[ -s results/blocks_xen_bands_base/${N}.json ] || { echo "[$N] 无 base 频带"; exit 1; }
[ -s results/xen_controls/${N}.json ]     || python -u xen_controls.py --name $N
[ -s results/xen_band_ceiling/${N}.json ] || python -u xen_band_ceiling.py --name $N
[ -s results/xen_inductive/${N}.json ]    || python -u xen_inductive.py --name $N
[ -s results/xen_attrib/${N}.json ]       || python -u xen_attrib.py --name $N
[ -s results/xen_addsplit/${N}.json ]     || python -u xen_addsplit.py --name $N
[ -s results/blocks_xen_seeds/${N}.json ] || python -u src/blocks_xen_seeds.py --name $N
[ -s results/blocks_xen_ablate/${N}.json ]|| python -u src/blocks_xen_ablate.py --name $N
[ -s results/per_gene_xen/${N}.json ]     || python -u src/per_gene_xen.py --name $N
for SET in "k4 4 17 0.5" "k12 12 40 0.5" "lazy75 8 29 0.75"; do read TAG K CUT LZ <<< "$SET"
  [ -s results/blocks_xen_bands_${TAG}/${N}.json ] || python -u src/blocks_xen_bands.py --name $N --knn $K --cut_um $CUT --lazy $LZ --tag $TAG; done
echo "== 产出 =="; for d in xen_controls xen_band_ceiling xen_inductive xen_attrib xen_addsplit blocks_xen_seeds blocks_xen_ablate per_gene_xen blocks_xen_bands_k4 blocks_xen_bands_k12 blocks_xen_bands_lazy75; do [ -s results/$d/${N}.json ] && echo "OK  $d" || echo "缺  $d"; done
