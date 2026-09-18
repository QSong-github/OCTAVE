#!/bin/bash
#SBATCH -J rselagg
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 1:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
python3 ridge_loco.py ciga,conch_v15,conch_v1,dinov2_large,dinov3_vitl16,genbio_pathfm,gigapath,h0_mini,hibou_l,hoptimus0,hoptimus1,kaiko_vitb16,kaiko_vitl14,kaiko_vits16,keep,lunit_vits8,mascaret,midnight12k,musk,omiclip,openmidnight,phaet,phikon,phikon_v2,plip,quiltnet,uni_v1,uni_v2,virchow2,virchow
python3 -u k_sens_rsel.py | grep -E "^\s+k=|跨度|移动|超参|翻转"
python3 -u cohort_spread_rsel.py | tail -3
python3 mlp_summary.py rsel
