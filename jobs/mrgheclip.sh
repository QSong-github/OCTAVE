#!/bin/bash
#SBATCH -J mrgheclip
#SBATCH -o /path/to/systema4ST/logs/mrgheclip.out
#SBATCH -e /path/to/systema4ST/logs/mrgheclip.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST && python3 -u merge_folds.py heclip
