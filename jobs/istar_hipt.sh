#!/bin/bash
#SBATCH -J istarhipt
#SBATCH --qos=YOUR_QOS --partition=hpg-default --array=0-15 -c 4 --mem=48G -t 6:00:00 --nice=50000
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONWARNINGS=ignore
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
N=${REG[$SLURM_ARRAY_TASK_ID]}; [ -s results/istar_xen_hipt/${N}.json ] && exit 0
[ -s istar_run/xen_${N}_half/embeddings-hist.pickle ] || { echo "无 HIPT 特征"; exit 1; }
python -u istar_hipt_ridge_xen.py --name $N
