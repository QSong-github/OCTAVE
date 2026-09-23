#!/bin/bash
#SBATCH -J blk
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-29 -c 4 --mem=24G -t 4:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=4 BLK_K=20,50,200
E=(ciga conch_v15 conch_v1 dinov2_large dinov3_vitl16 genbio_pathfm gigapath h0_mini hibou_l hoptimus0 hoptimus1 kaiko_vitb16 kaiko_vitl14 kaiko_vits16 keep lunit_vits8 mascaret midnight12k musk omiclip openmidnight phaet phikon phikon_v2 plip quiltnet uni_v1 uni_v2 virchow2 virchow ); N=${E[$SLURM_ARRAY_TASK_ID]}
[ -s "results/hest_blocks_${N}.json" ] && exit 0
python -u hest_blocks.py "$N"
