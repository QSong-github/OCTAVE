#!/bin/bash
#SBATCH -J cp10krsel
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-56 -c 2 --mem=12G --nice=50000 -t 2:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# CP10K normalisation under the same leave-one-cohort-out ridge alpha as Table 1,
# so that the normalisation comparison (Fig 5a-b, Appendix K.2) uses Table 1's protocol.
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
ENC=(biomedclip ciga clip_vitl14 conch_v15 conch_v1 ctranspath dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath_flash gigapath gpfm h0_mini hibou_b hibou_l hoptimus0 hoptimus1 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 kaiko_vits16 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco lunit_r50_swav lunit_vits16 lunit_vits8 mascaret midnight12k mstar musk omiclip openmidnight path_foundation pathgen_clip pathryoshka_b phaet phikon phikon_v2 plip quiltnet retccl siglip2 uni_v1 uni_v2 virchow2 virchow)
X=${ENC[$SLURM_ARRAY_TASK_ID]}; O=results/hest_cp10k_rsel_${X}.json; [ -s $O ] && exit 0
A=results/hest_rsel_ps_${X}.json; [ -s "$A" ] || { echo "no rsel file for $X"; exit 1; }
python -u src/hest_effres_ps.py --encoder $X --skip_sigma --target cp10k --alpha_by_cohort $A --out $O
