#!/bin/bash
#SBATCH -J vhdleg
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 16 --mem=160G -t 12:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
# rebuild the parent's binned_16um.h5ad inside the project, rerun src/ruler_fold.py on it with the
# Hibou-L features kept in results/emb9, and compare with results/ruler_fold.json. Writes only to data/vhd_legacy and results/repro_vhd.
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p data/vhd_legacy/emb results/repro_vhd
for P in P2 P5; do
  [ -s data/vhd_legacy/emb/emb_hibou_l_$P.npy ] || cp results/emb9/emb_hibou_l_Visium_HD_Human_Colon_Cancer_$P.npy data/vhd_legacy/emb/emb_hibou_l_$P.npy
done
[ -s data/vhd_legacy/binned_16um.h5ad ] || python -u src/rebuild_binned16.py
python -u - <<'PY'
import sys; sys.path.insert(0, "src")
import st_encoder_bench as SEB, effres
R = "/path/to/project"
SEB.H5AD, SEB.EMBDIR = f"{R}/data/vhd_legacy/binned_16um.h5ad", f"{R}/data/vhd_legacy/emb"
import ruler_fold
sys.argv = ["ruler_fold.py", "--out", f"{R}/results/repro_vhd"]
ruler_fold.main()
PY
python -u src/repro_vhd_check.py results/ruler_fold.json results/repro_vhd/ruler_fold.json
