#!/bin/bash
#SBATCH -J xenembfix
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000 --gres=gpu:1 --array=0-17 -c 6 --mem=48G -t 12:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
read -r E N < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" jobs/xen_missing_pairs.txt)
F=data/prepped_xen/${N}_bin16.h5ad; OUT=results/emb_xen/emb_${E}_${N}.npy; [ -s "$OUT" ] && exit 0
CTX=$(python -c "
import anndata as ad; a=ad.read_h5ad('$F',backed='r'); print(int(round(61.4*float(a.uns['px_per_um']))))")
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif)
BS=128; [ "$E" = "kaiko_vitl14" ] && BS=48
echo "[$E / $N] ctx_px=$CTX bs=$BS $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -u xen_embed_all.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder "$E" --ctx_px "$CTX" --batch $BS
