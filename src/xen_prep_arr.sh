#!/bin/bash
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
NAMES=($(ls data/xenium))
N=${NAMES[$SLURM_ARRAY_TASK_ID]}
python -u src/xen_prep.py --name "$N"
