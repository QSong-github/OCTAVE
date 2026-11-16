#!/bin/bash
#SBATCH -J xenaltm
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-21 -c 8 --mem=64G --nice=50000 -t 12:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# OCTAVE next to simpler multiscale metrics on the same predictions (see xen_altmetrics.py).
# 8 threads as in jobs/xen_ctrl.sh, so that KMeans reproduces the published oracle.
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
REG=(Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S4_Top Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Bot Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Ovary_Cancer_FF Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE)
R=${REG[$SLURM_ARRAY_TASK_ID]}; [ -s results/xen_altmetrics/${R}.json ] && exit 0
python -u xen_altmetrics.py --name $R
