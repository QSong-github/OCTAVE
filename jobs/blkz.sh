#!/bin/bash
#SBATCH -J blkz
#SBATCH --qos=qsong1 --partition=hpg-default --array=0-49 -c 4 --mem=24G -t 6:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export OMP_NUM_THREADS=4 BLK_K=20,50,200 BLK_ZSCORE=1
A=($(python3 -c "import json; print(' '.join(sorted(d['enc'] for d in json.load(open('results/hest_blocks_summary.json')))))"))
X=${A[$SLURM_ARRAY_TASK_ID]}; [ -s results/hest_blocks_z_${X}.json ] && exit 0
python -u hest_blocks.py $X
