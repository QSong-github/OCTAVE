#!/bin/bash
#SBATCH -J tpxnomask
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-1 -c 4 --mem=64G -t 6:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate triplex
cd /path/to/systema4ST; export PYTHONWARNINGS=ignore TQDM_DISABLE=1
P=("COAD TENX111" "LYMPH_IDC NCBI684"); read C S <<< "${P[$SLURM_ARRAY_TASK_ID]}"
rm -f data/triplex/$C/emb/neighbor/*/$S.h5
python -u src/triplex_dump_nbr_nomask.py $C $S
