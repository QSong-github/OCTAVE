#!/bin/bash
#SBATCH -J new16ds
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-207 -c 4 --mem=24G -t 8:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=4 BLK_K=20,50,200
A=( kaiko_vitb8 lunit_vits16 lunit_r50_swav lunit_r50_bt lunit_r50_moco ctranspath gpfm retccl dinov2_base dinov2_giant dinov3_vitb16 dinov3_vith16 clip_vitl14 pathgen_clip siglip2 biomedclip); AL=(0.1 1 10 100 1000 10000 100000 1000000 10000000 100000000 1000000000); I=$SLURM_ARRAY_TASK_ID; X=${A[$((I/13))]}; J=$((I%13))
if [ $J = 0 ]; then O=results/hest_effres_ps_${X}.json; [ -s $O ] && exit 0; exec python -u src/hest_effres_ps.py --encoder $X --skip_sigma --out $O; fi
if [ $J = 12 ]; then [ -s results/hest_blocks_${X}.json ] && exit 0; exec python -u hest_blocks.py $X; fi
a=${AL[$((J-1))]}; O=results/ridge_alpha/hest_ra_${X}_a${a}.json; [ -s $O ] && exit 0
python -u src/hest_effres_ps.py --encoder $X --skip_sigma --ridge_alpha $a --out $O
