#!/bin/bash
#SBATCH -J new16agg
#SBATCH --qos=YOUR_QOS --partition=hpg-default -c 2 --mem=16G -t 1:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
python3 ridge_loco.py kaiko_vitb8,lunit_vits16,lunit_r50_swav,lunit_r50_bt,lunit_r50_moco,ctranspath,gpfm,retccl,dinov2_base,dinov2_giant,dinov3_vitb16,dinov3_vith16,clip_vitl14,pathgen_clip,siglip2,biomedclip
python3 blk_agg.py | tail -12
