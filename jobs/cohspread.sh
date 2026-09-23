#!/bin/bash
#SBATCH -J cohspread
#SBATCH -o /path/to/project/logs/cohspread.out
#SBATCH -e /path/to/project/logs/cohspread.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project && python3 -u cohort_spread.py
