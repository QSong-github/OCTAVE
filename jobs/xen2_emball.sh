#!/bin/bash
#SBATCH -J xen2emball
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 --array=0-329 -c 6 --mem=48G -t 12:00:00 --nice=50000
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# 第二批 6 区域 × 55 编码器（45 走 xen_embed_all.py，10 走 src/hd_embed.py，与 16 区域完全相同的脚本与视野 61.4 µm；hibou_l 与 path_foundation 另行）。幂等。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
REG=(Xenium_V1_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE)
ENC=(biomedclip ciga clip_vitl14 conch_v1 dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath_flash gpfm hibou_b hoptimus0 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco lunit_r50_swav lunit_vits16 lunit_vits8 mascaret mstar musk omiclip openmidnight pathgen_clip pathryoshka_b phaet phikon plip quiltnet retccl siglip2 uni_v1 virchow uni_v2 virchow2 hoptimus1 gigapath phikon_v2 conch_v15 kaiko_vits16 ctranspath h0_mini midnight12k)
I=$SLURM_ARRAY_TASK_ID; N=${REG[$((I % 6))]}; E=${ENC[$((I / 6))]}
F=data/prepped_xen/${N}_bin16.h5ad; OUT=results/emb_xen/emb_${E}_${N}.npy
[ -s "$F" ] || { echo "[$N] 无 bin16"; exit 1; }
[ -s "$OUT" ] && { echo "[$E/$N] 已存在"; exit 0; }
CTX=$(python -c "
import anndata as ad; a=ad.read_h5ad('$F',backed='r'); print(int(round(61.4*float(a.uns['px_per_um']))))")
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif)
echo "[$E / $N] ctx_px=$CTX $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [ $((I / 6)) -ge 45 ]; then case $E in kaiko_vits16) TN=kaiko-vits16;; h0_mini) TN=h0-mini;; *) TN=$E;; esac
  python -u src/hd_embed.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder "$TN" --ctx_px "$CTX" --batch 128
else python -u xen_embed_all.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder "$E" --ctx_px "$CTX" --batch 128; fi
