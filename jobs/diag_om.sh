#!/bin/bash
#SBATCH -J diagom
#SBATCH --qos=YOUR_QOS --partition=hpg-default -c 8 --mem=32G -t 2:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=8
python -u diag_openmidnight.py
