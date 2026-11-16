#!/bin/bash
#SBATCH -J xen2blk
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-5 -c 8 --mem=160G -t 12:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# 第二批 Xenium：块识别上界 + 频带（与 blocks_xen.sh / blocks_xen_bands.sh 同参数，tag=base）。幂等。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
NAMES=(Xenium_V1_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE)
N=${NAMES[$SLURM_ARRAY_TASK_ID]}
[ -s results/emb_xen/emb_hibou_l_${N}.npy ] || { echo "[$N] 无嵌入"; exit 1; }
[ -s results/blocks_xen/${N}.json ] || python -u src/blocks_xen.py --name "$N"
[ -s results/blocks_xen_bands_base/${N}.json ] || python -u src/blocks_xen_bands.py --name "$N" --tag base
ls -la results/blocks_xen/${N}.json results/blocks_xen_bands_base/${N}.json
