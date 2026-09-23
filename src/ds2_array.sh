#!/bin/bash
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
FILES=($(ls data/prepped_xen/*_bin16.h5ad))
F=${FILES[$SLURM_ARRAY_TASK_ID]}
N=$(basename "$F" _bin16.h5ad)
python -u src/downstream2.py --name "$N"
