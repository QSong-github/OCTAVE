#!/bin/bash
#SBATCH --job-name=refig_ps
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --cpus-per-task=32 --mem=64G --time=3:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 逐页审查后的四张图重画：Fig1（标题 3.9–8.5×）、Fig4=Fig2_what_pcc_measures（插图挪到右下、面板 d 限定首发 15 区域）、
# Fig3（面板 e 负号裁切 xlim）、Fig5（面板 e 标题）。与 jobs/fig1v2.sh / jobs/refig2.sh 同一环境与资源。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
export MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -u fig1_iclr_v2.py
python -u make_figs_new.py M 6 3
echo "════ 复核 ════"
echo "Fig1 title:  $(pdftotext figures/Fig1_iclr_v2.pdf - | grep -o '[0-9.]*–[0-9.]*× too smooth')"
echo "Fig4 counts: $(pdftotext figures/Fig2_what_pcc_measures.pdf - | grep -o '[0-9]*/[0-9]* specimens (P = [0-9.]*)' | tr '\n' ' ')"
echo "Fig4 panel d: $(pdftotext figures/Fig2_what_pcc_measures.pdf - | grep -o '[0-9]*/[0-9]* regions' | tr '\n' ' ')"
echo "Fig3 panel e: $(pdftotext figures/Fig3_evaluation_protocol.pdf - | grep -o '[−-]*17\.6' | tr '\n' ' ')"
echo "Fig5 panel e: $(pdftotext figures/Fig5_protocol_knobs.pdf - | grep -o 'within one [a-z]*')"
ls -la --time-style=+%m-%d_%H:%M figures/Fig1_iclr_v2.pdf figures/Fig2_what_pcc_measures.pdf figures/Fig3_evaluation_protocol.pdf figures/Fig5_protocol_knobs.pdf | awk '{print $6, $7}'
