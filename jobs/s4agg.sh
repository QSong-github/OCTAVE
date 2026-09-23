#!/bin/bash
#SBATCH -J s4agg
#SBATCH -o /path/to/project/logs/s4agg.out
#SBATCH -e /path/to/project/logs/s4agg.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project && python3 -u step4_agg.py
