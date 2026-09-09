#!/bin/bash
#SBATCH -J seedagg
#SBATCH -o /path/to/systema4ST/logs/seedagg.out
#SBATCH -e /path/to/systema4ST/logs/seedagg.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST && python3 -u seed_agg.py
