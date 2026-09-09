#!/bin/bash
#SBATCH -J hestselfull
#SBATCH --qos=YOUR_QOS --partition=hpg-default --array=0-56 -c 4 --mem=24G -t 8:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
# Unified benchmark baseline: ridge with the leave-one-cohort-out selected alpha (per cohort), with bands.
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
ENC=(biomedclip ciga clip_vitl14 conch_v1 conch_v15 ctranspath dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath gigapath_flash gpfm h0_mini hibou_b hibou_l hoptimus0 hoptimus1 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 kaiko_vits16 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco lunit_r50_swav lunit_vits16 lunit_vits8 mascaret midnight12k mstar musk omiclip openmidnight path_foundation pathgen_clip pathryoshka_b phaet phikon phikon_v2 plip quiltnet retccl siglip2 uni_v1 uni_v2 virchow virchow2)
X=${ENC[$SLURM_ARRAY_TASK_ID]}; O=results/hest_effres_ps_selfull_${X}.json; [ -s $O ] && exit 0
A=results/hest_rsel_ps_${X}.json; [ -s "$A" ] || { echo "no rsel file"; exit 1; }
python -u src/hest_effres_ps.py --encoder $X --alpha_by_cohort $A --out $O
