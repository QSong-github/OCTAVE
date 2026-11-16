#!/bin/bash
#SBATCH -J xen2pfan
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-5 -c 8 --mem=48G -t 6:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
REG=(Xenium_V1_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE)
N=${REG[$SLURM_ARRAY_TASK_ID]}; [ -s results/emb_xen/emb_path_foundation_${N}.npy ] || { echo "无嵌入"; exit 0; }
[ -s results/blocks_xen_bands_path_foundation/${N}.json ] && exit 0
python -u src/blocks_xen_bands.py --name $N --tower path_foundation --tag path_foundation
