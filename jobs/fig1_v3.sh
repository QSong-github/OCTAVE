#!/bin/bash
#SBATCH -J fig1v3
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 32 --mem=48G -t 01:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# data for Figure 1 panel b (the 2 Visium HD windows of jobs/fig1_windows.sh), see src/fig1_v3_data.py. 32 cores and 32 threads as in jobs/fig1_windows.sh; with 16 cores sklearn K-means uses 16 threads and P2 reaches another partition (run kept in results/repro_vhd/fig1_v3_c16).
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
echo "node $(hostname) $(date '+%F %T')"
python -u src/fig1_v3_data.py
echo "done $(date '+%F %T')"
