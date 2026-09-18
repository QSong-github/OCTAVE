#!/bin/bash
#SBATCH -J xensens
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-63 -c 8 --mem=48G -t 6:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
SET=("base 8 29 0.5" "k4 4 17 0.5" "k12 12 40 0.5" "lazy75 8 29 0.75")
I=$SLURM_ARRAY_TASK_ID; R=${REG[$((I % 16))]}; read TAG K CUT LZ <<< "${SET[$((I / 16))]}"
[ -s results/blocks_xen_bands_${TAG}/${R}.json ] && exit 0
python -u src/blocks_xen_bands.py --name $R --knn $K --cut_um $CUT --lazy $LZ --tag $TAG
