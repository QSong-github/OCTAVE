#!/bin/bash
#SBATCH -J xi57
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-263%30 -c 8 --mem=48G -t 24:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# Xenium partition and means both learned from training blocks, all 57 encoders (22 regions x 12 encoder chunks). Idempotent.
set -e
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Lung_Cancer_FFPE Xenium_V1_Human_Ovary_Cancer_FF)
python -u xen_ctrl57.py --name ${REG[$((SLURM_ARRAY_TASK_ID / 12))]} --mode ind --chunk $((SLURM_ARRAY_TASK_ID % 12)) --nchunk 12
