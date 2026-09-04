#!/bin/bash
#SBATCH -J refig2
#SBATCH --qos=qsong1 --partition=hpg-default -c 32 --mem=64G -t 2:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -u fig1_iclr.py
python -u make_figs_new.py M 6 3
cp -f figures/Fig6_protocol_knobs.pdf figures/Fig5_protocol_knobs.pdf
echo "════ 复核 ════"
for f in Fig1_iclr Fig2_scale Fig2_what_pcc_measures Fig5_protocol_knobs Fig3_evaluation_protocol Fig6_downstream_consequences; do
  n=$(pdftotext figures/$f.pdf - 2>/dev/null | grep -oiE "buys|the ruler|knob|exchange rate|leakage buffer|Equivalent|ladder|blocks only|block oracle" | wc -l | tr -d " ")
  echo "  $f: $n 处"
done
