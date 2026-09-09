#!/bin/bash
#SBATCH -J effresfull
#SBATCH --qos=YOUR_QOS --partition=hpg-default --array=0-26 -c 4 --mem=24G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
# 第一份审稿意见：HEST 上的 OCTAVE 式 Δ 需要 ridge 的逐带 PCC；27 个后批编码器此前只跑了 --skip_sigma。
# 输出到新文件名 hest_effres_ps_full_{X}.json，不覆盖现有文件。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mapfile -t ENC < jobs/effres_full_list.txt
X=${ENC[$SLURM_ARRAY_TASK_ID]}; O=results/hest_effres_ps_full_${X}.json; [ -s $O ] && exit 0
python -u src/hest_effres_ps.py --encoder $X --out $O
