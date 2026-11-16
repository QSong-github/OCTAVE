#!/bin/bash
#SBATCH -J figs6
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=8G -t 00:20:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# Supp. Figure 6 (figures/Fig5_protocol_knobs.pdf) from the Visium HD reruns on the rebuilt inputs
# (results/repro_vhd) and the encoder range over the 57 encoders (results/repro_vhd/tower_sweep57).
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
echo "node $(hostname) $(date '+%F %T')"
python -u make_figs_new.py 6
echo "done $(date '+%F %T')"
