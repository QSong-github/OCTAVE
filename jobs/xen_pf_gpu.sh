#!/bin/bash
#SBATCH -J xenpfgpu
#SBATCH --qos=qsong1 --partition=hpg-turin --gres=gpu:l4:1 --array=0-15 -c 2 --mem=24G -t 12:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
# 第二段（L4 GPU，tfpf 环境，只用 h5py/numpy/TF）：逐分片推理并删除分片，最后拼接。
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate tfpf
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
N=${REG[$SLURM_ARRAY_TASK_ID]}; OUT=results/emb_xen/emb_path_foundation_${N}.npy; [ -s "$OUT" ] && exit 0
TMP=results/emb_xen/pf_tmp_${N}; NB=$(cat $TMP/nbins.txt); nvidia-smi --query-gpu=name --format=csv,noheader | head -1
for f in $(ls $TMP/tiles_*.h5 | sort -t_ -k3 -n); do k=$(basename $f .h5 | sed s/tiles_//); [ -s $TMP/emb_$k.npy ] || python -u xen_pf_embed_shard.py --h5 $f --out $TMP/emb_$k.npy 2>&1 | grep -v "cuda_\|cpu_feature_guard\|rebuild TensorFlow\|TensorRT\|absl\|I0000\|W0000"; rm -f $f; done
python -c "
import numpy as np, glob, re; fs=sorted(glob.glob('$TMP/emb_*.npy'), key=lambda f:int(re.search(r'emb_(\d+)\.npy',f).group(1))); E=np.concatenate([np.load(f) for f in fs]); assert len(E)==$NB, (len(E), $NB); np.save('$OUT', E); print('saved', E.shape)"
rm -rf $TMP
