#!/bin/bash
#SBATCH -J fig1fw
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 16 --mem=48G -t 01:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# data for the framework figure (Figure 1), see src/fig1_framework_data.py
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 MPLBACKEND=Agg OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "node $(hostname) $(date '+%F %T')"
python -u src/fig1_framework_data.py
echo "done $(date '+%F %T')"
