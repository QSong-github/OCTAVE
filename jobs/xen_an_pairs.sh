#!/bin/bash
#SBATCH -J xenanfix
#SBATCH --qos=YOUR_QOS --partition=hpg-default --array=0-17 -c 8 --mem=48G -t 6:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
read -r E N < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" jobs/xen_missing_pairs.txt)
[ -s results/emb_xen/emb_${E}_${N}.npy ] || { echo "无嵌入"; exit 0; }
[ -s results/blocks_xen_bands_${E}/${N}.json ] && exit 0
python -u src/blocks_xen_bands.py --name $N --tower $E --tag $E
