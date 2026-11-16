#!/bin/bash
#SBATCH -J vhdtw57
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 10 --mem=80G -t 12:00:00
#SBATCH --array=0-56%20
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# the encoder comparison of Supp. Figure 5d (0.116 across 25 encoders, 12.0% fall in sigma_A per 0.01 PCC), rerun with
# src/tower_sweep.py (leave one section out, 50 genes, ridge 1e4) on the 57 encoders of the paper.
# Features are those of results/emb_vhd, linked as data/vhd_legacy/emb57/emb_<encoder>_<P2|P5>.npy. Expression is the rebuilt
# data/vhd_legacy/binned_16um.h5ad. Submitted to start after the Visium HD feature workers end.
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=10
unset OPENBLAS_NUM_THREADS MKL_NUM_THREADS
T=($(ls data/vhd_legacy/emb57 | sed -n 's/^emb_\(.*\)_P2\.npy$/\1/p' | sort))
[ ${#T[@]} = 57 ] || { echo "expected 57 encoders, found ${#T[@]}"; exit 2; }
E=${T[$SLURM_ARRAY_TASK_ID]}
[ -s results/repro_vhd/tower_sweep57/$E.json ] && { echo "$E done"; exit 0; }
for P in P2 P5; do [ -s data/vhd_legacy/emb57/emb_${E}_$P.npy ] || { echo "missing features $E/$P"; exit 3; }; done
echo "node $(hostname) encoder $E $(date '+%F %T')"
python -u src/vhd_repro_run.py sweep --emb emb57 --tower $E --hvg 50 --tmax 2048 --outdir results/repro_vhd/tower_sweep57
