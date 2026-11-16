#!/bin/bash
#SBATCH -J xen2emb
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 --array=0-5 -c 6 --mem=48G -t 12:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# 第二批 Xenium：主塔 hibou_l 嵌入，物理视野 61.4 µm（与 16 区域同协议）。幂等。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
NAMES=(Xenium_V1_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE)
N=${NAMES[$SLURM_ARRAY_TASK_ID]}; E=hibou_l
F=data/prepped_xen/${N}_bin16.h5ad; OUT=results/emb_xen/emb_${E}_${N}.npy
[ -s "$F" ] || { echo "[$N] 无 bin16"; exit 1; }
[ -s "$OUT" ] && { echo "[$E/$N] 已存在"; exit 0; }
CTX=$(python -c "
import anndata as ad; a=ad.read_h5ad('$F',backed='r'); print(int(round(61.4*float(a.uns['px_per_um']))))")
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif)
echo "[$E / $N] ctx_px=$CTX $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -u xen_embed_all.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder "$E" --ctx_px "$CTX" --batch 128
