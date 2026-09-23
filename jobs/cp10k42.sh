#!/bin/bash
#SBATCH -J cp10k42
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-41 -c 4 --mem=24G -t 4:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=4
A=(omiclip ctranspath plip keep retccl biomedclip hibou_b mstar lunit_r50_moco lunit_vits16 quiltnet clip_vitl14 dinov3_vitb16 lunit_r50_swav lunit_r50_bt gigapath_flash dinov2_base path_foundation conch_v1 pathgen_clip musk uni_v1 pathryoshka_b siglip2 dinov2_giant dinov3_vith16 distillpath_ks16 kaiko_vitb8 phaet gpfm hibou_l hoptimus1 distillpath_is16 h0_mini genbio_pathfm litefm_s virchow litefm_l litefm litevirchow2 mascaret openmidnight)
X=${A[$SLURM_ARRAY_TASK_ID]}; O=results/hest_cp10k_${X}.json; [ -s $O ] && exit 0
python -u src/hest_effres_ps.py --encoder $X --skip_sigma --target cp10k --out $O
