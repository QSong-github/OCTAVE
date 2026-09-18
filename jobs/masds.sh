#!/bin/bash
#SBATCH -J masds
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-40 -c 2 --mem=12G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export OMP_NUM_THREADS=2
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM); KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID
[ "$I" = 40 ] && exec python -u src/hest_effres_ps.py --encoder mascaret --out results/hest_effres_ps_mascaret.json
K=${KS[$((I/10))]}; C=${COH[$((I%10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_mascaret; else OUT=results/hest_floor_k${K}_mascaret; fi
mkdir -p $OUT
python -u src/hest_floor.py --encoder mascaret --cohort "$C" --k "$K" --out "$OUT/$C.json"
