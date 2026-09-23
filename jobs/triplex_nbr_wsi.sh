#!/bin/bash
#SBATCH -J tpxnbrwsi
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 4 --mem=48G -t 6:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate triplex
cd /path/to/project; export PYTHONWARNINGS=ignore
rm -f data/triplex/COAD/emb/neighbor/*/TENX111.h5
python -u src/triplex_nbr_from_wsi.py COAD TENX111
