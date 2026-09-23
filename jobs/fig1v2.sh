#!/bin/bash
#SBATCH --job-name=fig1v2
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --cpus-per-task=32 --mem=64G --time=3:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
export MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -u fig1_iclr_v2.py
