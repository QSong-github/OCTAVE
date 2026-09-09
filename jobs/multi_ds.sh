#!/bin/bash
#SBATCH -J multids
#SBATCH --qos=qsong1 --partition=hpg-default --array=0-175 -c 8 --mem=48G -t 6:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
ENC=(hibou_l uni_v2 virchow2 hoptimus1 gigapath phikon_v2 conch_v15 kaiko_vits16 ctranspath h0_mini midnight12k)
I=$SLURM_ARRAY_TASK_ID; N=${REG[$((I % 16))]}; E=${ENC[$((I / 16))]}
[ -s results/multi_ds/${E}_${N}.json ] && exit 0
python -u multi_downstream.py --name $N --tower $E
