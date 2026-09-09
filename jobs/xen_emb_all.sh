#!/bin/bash
#SBATCH -J xenemball
#SBATCH --qos=YOUR_QOS --partition=hpg-b200,hpg-rtx6000 --gres=gpu:1 --array=0-719 -c 6 --mem=48G -t 12:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
# 其余 45 个编码器的 Xenium 嵌入（path_foundation 为 TensorFlow/CPU，单独处理）。物理视野 61.4 µm，与 hibou_l 同协议。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
ENC=(biomedclip ciga clip_vitl14 conch_v1 dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath_flash gpfm hibou_b hoptimus0 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco lunit_r50_swav lunit_vits16 lunit_vits8 mascaret mstar musk omiclip openmidnight pathgen_clip pathryoshka_b phaet phikon plip quiltnet retccl siglip2 uni_v1 virchow)
I=$SLURM_ARRAY_TASK_ID; N=${REG[$((I % 16))]}; E=${ENC[$((I / 16))]}
F=data/prepped_xen/${N}_bin16.h5ad; OUT=results/emb_xen/emb_${E}_${N}.npy
[ -s "$OUT" ] && { echo "[$E/$N] 已存在"; exit 0; }
CTX=$(python -c "
import anndata as ad; a=ad.read_h5ad('$F',backed='r'); print(int(round(61.4*float(a.uns['px_per_um']))))")
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif)
echo "[$E / $N] ctx_px=$CTX $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -u xen_embed_all.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder "$E" --ctx_px "$CTX" --batch 128
