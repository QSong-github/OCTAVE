#!/bin/bash
#SBATCH -J aggxen57
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 1 --mem=4G -t 00:20:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# Xenium column of Table 2 over the 57 encoders -> results/xen_ctrl57_summary.json (agg_xen_ctrl57.py).
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export PYTHONDONTWRITEBYTECODE=1
python -u agg_xen_ctrl57.py
