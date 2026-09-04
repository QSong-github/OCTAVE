#!/bin/bash
#SBATCH -J ralpha
#SBATCH --qos=qsong1 --partition=hpg-default --array=0-209 -c 2 --mem=16G -t 2:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export OMP_NUM_THREADS=2
E=(ciga conch_v15 conch_v1 dinov2_large dinov3_vitl16 genbio_pathfm gigapath h0_mini hibou_l hoptimus0 hoptimus1 kaiko_vitb16 kaiko_vitl14 kaiko_vits16 keep lunit_vits8 mascaret midnight12k musk omiclip openmidnight phaet phikon phikon_v2 plip quiltnet uni_v1 uni_v2 virchow2 virchow ); A=(0.1 1 10 100 1000 10000 100000); I=$SLURM_ARRAY_TASK_ID; N=${E[$((I/7))]}; AL=${A[$((I%7))]}
O=results/ridge_alpha/hest_ra_${N}_a${AL}.json
[ -s "$O" ] && exit 0
python -u src/hest_effres_ps.py --encoder "$N" --skip_sigma --ridge_alpha $AL --out "$O"
