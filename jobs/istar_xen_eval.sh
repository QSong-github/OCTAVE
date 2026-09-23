#!/bin/bash
#SBATCH -J istareval
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-15 -c 8 --mem=64G -t 6:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS=ignore
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
N=${REG[$SLURM_ARRAY_TASK_ID]}; [ -s results/istar_xen/${N}.json ] && exit 0
[ $(ls istar_run/xen_${N}_half/cnts-super 2>/dev/null | wc -l) -ge 1 ] || { echo "无超分输出"; exit 1; }
python -u istar_eval_xen.py --name $N
