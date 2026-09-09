#!/bin/bash
#SBATCH -J xenemb10
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000 --gres=gpu:1 --array=0-159 -c 6 --mem=48G -t 12:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
# 审稿意见：多个真实编码器在 Xenium 上的 OCTAVE 曲线。物理视野仍锁 61.4 µm（与 hibou_l 一致）。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
ENC=(uni_v2 virchow2 hoptimus1 gigapath phikon_v2 conch_v15 kaiko_vits16 ctranspath h0_mini midnight12k)
I=$SLURM_ARRAY_TASK_ID; N=${REG[$((I % 16))]}; E=${ENC[$((I / 16))]}
case $E in kaiko_vits16) TN=kaiko-vits16;; h0_mini) TN=h0-mini;; *) TN=$E;; esac
F=data/prepped_xen/${N}_bin16.h5ad; OUT=results/emb_xen/emb_${E}_${N}.npy
[ -s "$OUT" ] && { echo "[$E/$N] 已存在"; exit 0; }
CTX=$(python -c "
import anndata as ad; a=ad.read_h5ad('$F',backed='r'); print(int(round(61.4*float(a.uns['px_per_um']))))")
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif)
echo "[$E→$TN / $N] ctx_px=$CTX  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -u src/hd_embed.py --h5ad "$F" --tiff "$TIF" --out "$OUT" --encoder "$TN" --ctx_px "$CTX" --batch 128
