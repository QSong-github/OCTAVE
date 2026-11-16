#!/bin/bash
#SBATCH -J aggvhd57
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 1 --mem=4G -t 00:20:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# Visium HD column of Table 2 over the 57 encoders -> results/vhd_ctrl57_summary.json (agg_vhd_ctrl57.py).
# Run after the inductive tasks (jobs/vhd_ind57.sh) end.
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export PYTHONDONTWRITEBYTECODE=1
echo "vhd_ind57 files: $(ls results/vhd_ind57/*/*.json | wc -l)"
python -u agg_vhd_ctrl57.py
