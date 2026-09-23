#!/bin/bash
#SBATCH -J refig3
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 32 --mem=64G -t 1:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
export MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -u make_figs_new.py 3
pdftotext figures/Fig3_evaluation_protocol.pdf - | grep -i "peaks\|best at\|fixed grid" | head -4
