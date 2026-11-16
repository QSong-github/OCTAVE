#!/bin/bash
#SBATCH -J vhdband
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-113%40 -c 8 --mem=96G -t 12:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# Visium HD P2/P5, src/blocks_xen_bands.py unchanged (through src/blocks_vhd_bands.py) for all 57 encoders. Idempotent.
set -e
source /path/to/project/jobs/act.sh; act hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
SEC=(Visium_HD_Human_Colon_Cancer_P2 Visium_HD_Human_Colon_Cancer_P5)
ENC=(hibou_l biomedclip ciga clip_vitl14 conch_v1 conch_v15 ctranspath dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath gigapath_flash gpfm h0_mini hibou_b hoptimus0 hoptimus1 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 kaiko_vits16 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco lunit_r50_swav lunit_vits16 lunit_vits8 mascaret midnight12k mstar musk omiclip openmidnight path_foundation pathgen_clip pathryoshka_b phaet phikon phikon_v2 plip quiltnet retccl siglip2 uni_v1 uni_v2 virchow virchow2)
I=$SLURM_ARRAY_TASK_ID; N=${SEC[$((I % 2))]}; E=${ENC[$((I / 2))]}
[ -s results/blocks_vhd_bands_${E}/${N}.json ] && exit 0
[ -s results/emb_vhd/emb_${E}_${N}.npy ] || { echo "[$E/$N] no embedding"; exit 1; }
python -u src/blocks_vhd_bands.py --name $N --tower $E --tag $E
