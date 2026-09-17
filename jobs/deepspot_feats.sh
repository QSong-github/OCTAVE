#!/bin/bash
#SBATCH -J dsfeat
#SBATCH --qos=YOUR_QOS --partition=hpg-b200 --gres=gpu:1 --array=0-19 -c 6 --mem=64G -t 12:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
# DeepSpot 特征：spot tile + 3×3 子 tile，编码器 uni_v1 / hoptimus0（作者论文列出的基座之二）。幂等（按样本落盘）。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
COH=(SKCM HCC LUNG PAAD COAD READ IDC LYMPH_IDC PRAD CCRCC); ENC=(uni_v1 hoptimus0)
I=$SLURM_ARRAY_TASK_ID; C=${COH[$((I % 10))]}; E=${ENC[$((I / 10))]}
echo "[$E / $C] $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -u src/deepspot_feats.py --cohort $C --encoder $E
