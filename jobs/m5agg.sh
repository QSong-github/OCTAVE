#!/bin/bash
#SBATCH -J m5agg
#SBATCH -o /path/to/project/logs/m5agg.out
#SBATCH -e /path/to/project/logs/m5agg.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project && python3 -u methods_agg2.py
