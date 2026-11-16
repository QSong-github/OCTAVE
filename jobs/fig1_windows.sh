#!/bin/bash
#SBATCH -J fig1b214
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 32 --mem=16G -t 01:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# Figure 1 on the rebuilt Visium HD inputs, drawn by fig1_windows.py (the labels of the paper),
# with the 32 threads of jobs/vhd_repro2.sh task 0. Output in results/repro_vhd/fig1_windows; the first rerun in results/repro_vhd/fig1 stays.
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 MPLBACKEND=Agg
unset OPENBLAS_NUM_THREADS MKL_NUM_THREADS
echo "node $(hostname) $(date '+%F %T')"
FIG1_OUT=/path/to/project/results/repro_vhd/fig1_windows \
  OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 python -u src/vhd_repro_run.py fig1
python -u src/repro_vhd_check.py results/repro_vhd/fig1/fig1_v2_numbers.json results/repro_vhd/fig1_windows/fig1_v2_numbers.json || true
echo "done $(date '+%F %T')"
