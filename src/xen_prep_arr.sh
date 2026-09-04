#!/bin/bash
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
NAMES=($(ls data/xenium))
N=${NAMES[$SLURM_ARRAY_TASK_ID]}
python -u src/xen_prep.py --name "$N"
