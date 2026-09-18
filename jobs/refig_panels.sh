#!/bin/bash
#SBATCH -J refigp
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 32 --mem=64G -t 1:30:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -u fig2_scale.py
python -u make_figs_new.py M 6 3
cp -f figures/Fig6_protocol_knobs.pdf figures/Fig5_protocol_knobs.pdf
for f in Fig2_scale Fig2_what_pcc_measures Fig3_evaluation_protocol Fig5_protocol_knobs; do echo "── $f: $(pdftotext figures/$f.pdf - | grep -c "more sensitive\|True gene-by-gene\|reverse sign\|exact P") 处旧面板文字"; done
