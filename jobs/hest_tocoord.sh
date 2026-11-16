#!/bin/bash
#SBATCH -J hesttoc
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 4 --mem=24G -t 3:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u hest_trainonly_coord.py
