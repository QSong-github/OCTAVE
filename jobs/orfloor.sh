#!/bin/bash
#SBATCH -J orfloor
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-39 -c 2 --mem=12G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM); KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID; K=${KS[$((I/10))]}; C=${COH[$((I%10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_omiclip_raw; else OUT=results/hest_floor_k${K}_omiclip_raw; fi
mkdir -p $OUT
[ -s "$OUT/$C.json" ] && exit 0
python -u src/hest_floor.py --encoder omiclip_raw --cohort "$C" --k "$K" --out "$OUT/$C.json"
