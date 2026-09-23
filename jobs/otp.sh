#!/bin/bash
#SBATCH -J otp
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH -c 4 --mem=64G -t 01:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project && python3 -u omi_trunk_probe.py
