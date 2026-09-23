#!/bin/bash
#SBATCH -J agg27
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH -c 2 --mem=16G -t 00:30:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project && python3 -u agg27.py
