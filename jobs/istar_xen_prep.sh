#!/bin/bash
#SBATCH -J istarprep
#SBATCH --qos=qsong1 --partition=hpg-default --array=0-15 -c 4 --mem=64G -t 4:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONWARNINGS=ignore
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
N=${REG[$SLURM_ARRAY_TASK_ID]}; S=istar_run/xen_${N}_shared; D=istar_run/xen_${N}_half
[ -s $S/he-raw.jpg ] || python -u istar_prep_xen.py image --name $N --out $S
[ -s $D/test.npz ] || python -u istar_prep_xen.py fold --name $N --shared $S --out $D
