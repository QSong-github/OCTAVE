#!/bin/bash
#SBATCH -J vhdemb
#SBATCH --partition=YOUR_GPU_PARTITION --gres=gpu:1 -c 6 --mem=48G -t 12:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# Visium HD P2/P5 embeddings, 56 GPU encoders x 2 sections (tasks 0-111), same scripts and 61.4 um field of view as Xenium
# (xen2_emball.sh: 45 via xen_embed_all.py, 10 via src/hd_embed.py, hibou_l via src/hd_embed.py). path_foundation runs in vhd_pf.sh. Idempotent.
set -eo pipefail
source /path/to/project/jobs/act.sh; act hest; set -u
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
SEC=(Visium_HD_Human_Colon_Cancer_P2 Visium_HD_Human_Colon_Cancer_P5)
ENC=(biomedclip ciga clip_vitl14 conch_v1 dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath_flash gpfm hibou_b hoptimus0 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco lunit_r50_swav lunit_vits16 lunit_vits8 mascaret mstar musk omiclip openmidnight pathgen_clip pathryoshka_b phaet phikon plip quiltnet retccl siglip2 uni_v1 virchow uni_v2 virchow2 hoptimus1 gigapath phikon_v2 conch_v15 kaiko_vits16 ctranspath h0_mini midnight12k hibou_l)
I=$SLURM_ARRAY_TASK_ID; [ -z "${REV:-}" ] || I=$((111 - I)); N=${SEC[$((I % 2))]}; J=$((I / 2)); E=${ENC[$J]}
F=data/prepped/${N}_16um.h5ad; TIF=data/visiumhd/$N/${N}_PYRAMIDAL.tif; mkdir -p results/emb_vhd; OUT=results/emb_vhd/emb_${E}_${N}.npy
[ -s "$F" ] || { echo "[$N] no h5ad"; exit 1; }
[ -s "$OUT" ] && { echo "[$E/$N] exists"; exit 0; }
LOCK=results/emb_vhd/.lock_${E}_${N}; mkdir "$LOCK" 2>/dev/null || { echo "[$E/$N] running in another task"; exit 0; }; trap "rmdir $LOCK" EXIT   # two arrays can share the work (one forward, one with REV=1 backward)
CTX=$(python -c "import anndata as ad; a=ad.read_h5ad('$F',backed='r'); print(int(round(61.4*float(a.uns['px_per_um']))))")
echo "[$E / $N] ctx_px=$CTX $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [ $J -ge 45 ]; then case $E in kaiko_vits16) TN=kaiko-vits16;; h0_mini) TN=h0-mini;; *) TN=$E;; esac
  python -u src/hd_embed.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder "$TN" --ctx_px "$CTX" --batch 128
else python -u xen_embed_all.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder "$E" --ctx_px "$CTX" --batch 128; fi
