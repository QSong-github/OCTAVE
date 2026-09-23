#!/bin/bash
#SBATCH -J xenpfcut
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-15 -c 2 --mem=16G -t 12:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# 第一段（CPU 节点，hest 环境；hest 在 GPU 节点上 import anndata 会崩，所以切块必须在 CPU 节点做）：把该区域尚未嵌入的分片切成 gzip HDF5。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONWARNINGS=ignore
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
N=${REG[$SLURM_ARRAY_TASK_ID]}; F=data/prepped_xen/${N}_bin16.h5ad; OUT=results/emb_xen/emb_path_foundation_${N}.npy; [ -s "$OUT" ] && exit 0
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif); TMP=results/emb_xen/pf_tmp_${N}; mkdir -p $TMP
NB=$(python -c "import h5py; h=h5py.File('$F','r'); print(h['obsm']['pxl'].shape[0])"); echo "$NB" > $TMP/nbins.txt
SH_=20000; k=0; s=0
while [ $s -lt $NB ]; do e=$((s+SH_)); [ $e -gt $NB ] && e=$NB
  [ -s $TMP/emb_$k.npy ] || [ -s $TMP/tiles_$k.h5 ] || python -u xen_pf_cut.py --h5ad $F --tiff "$TIF" --out $TMP/tiles_$k.h5.part --start $s --end $e && { [ -f $TMP/tiles_$k.h5.part ] && mv $TMP/tiles_$k.h5.part $TMP/tiles_$k.h5; }
  k=$((k+1)); s=$e
done
echo "[$N] 切块完成，共 $k 个分片"
