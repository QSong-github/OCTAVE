#!/bin/bash
#SBATCH -J vhdrep2
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 32 --mem=64G -t 12:00:00
#SBATCH --array=0-7
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# the old Visium HD analyses of the paper, rerun on the rebuilt inputs (data/vhd_legacy, ) with the thread
# counts of their original job scripts. Outputs go to results/repro_vhd only (src/vhd_repro_run.py).
#   0    src/ruler_fold.py at 32 threads (jobs/ruler_fold.sh), compared with results/ruler_fold.json, then Figure 1
#   1-4  gene panel sweep with 20, 50, 100 and 200 genes (12 threads)
#   5    Hibou-L at 50 genes, the 224 px point of the patch-size sweep (10 threads)
#   6    exclusion buffer (src/buffer_cv.py)
#   7    K-means at 8 and 32 threads on the P5 half fold (src/kmeans_threads_vhd.py)
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; set -u
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 MPLBACKEND=Agg
unset OPENBLAS_NUM_THREADS MKL_NUM_THREADS
O=results/repro_vhd; I=$SLURM_ARRAY_TASK_ID
echo "node $(hostname) task $I $(date '+%F %T')"
case $I in
  0) OMP_NUM_THREADS=32 python -u src/vhd_repro_run.py ruler32
     python -u src/repro_vhd_check.py results/ruler_fold.json $O/t32/ruler_fold.json || true
     python -u src/repro_vhd_check.py $O/ruler_fold.json $O/t32/ruler_fold.json || true
     OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 python -u src/vhd_repro_run.py fig1 ;;
  1|2|3|4) H=$(echo 20 50 100 200 | cut -d" " -f$I)
     OMP_NUM_THREADS=12 python -u src/vhd_repro_run.py sweep --tower hibou_l --hvg $H --outdir $O/panel_sweep_hvg$H ;;
  5) OMP_NUM_THREADS=10 python -u src/vhd_repro_run.py sweep --tower hibou_l --hvg 50 --tmax 2048 --outdir $O/tower_sweep ;;
  6) OMP_NUM_THREADS=32 python -u src/vhd_repro_run.py buffer ;;
  7) for T in 8 32; do OMP_NUM_THREADS=$T python -u src/kmeans_threads_vhd.py --save; done
     python -u src/kmeans_threads_vhd.py --compare ;;
esac
echo "task $I done $(date '+%F %T')"
