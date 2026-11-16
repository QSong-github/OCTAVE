#!/bin/bash
#SBATCH -J xen2pf
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 --array=0-5 -c 2 --mem=32G -t 24:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# Path Foundation（TF SavedModel）两段式（TF 2.17 的 CUDA 构建不支持部分较新的 GPU 架构，第二段请在它支持的 GPU 上运行）：hest 环境切块写分片 → tfpf 环境推理，逐分片交替，最后拼接。分片用完即删。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
cd /path/to/project; export PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
REG=(Xenium_V1_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE)
N=${REG[$SLURM_ARRAY_TASK_ID]}; F=data/prepped_xen/${N}_bin16.h5ad; OUT=results/emb_xen/emb_path_foundation_${N}.npy; [ -s "$OUT" ] && exit 0
TIF=$(ls data/xenium/$N/${N}_PYRAMIDAL.tif 2>/dev/null || ls data/xenium/$N/${N}_he_image.ome.tif)
TMP=results/emb_xen/pf_tmp_${N}; mkdir -p $TMP
conda activate hest; NB=$(python -c "import h5py; h=h5py.File('$F','r'); print(h['obsm']['pxl'].shape[0])"); conda deactivate
SH_=20000; k=0; s=0
while [ $s -lt $NB ]; do e=$((s+SH_)); [ $e -gt $NB ] && e=$NB
  if [ ! -s $TMP/emb_$k.npy ]; then
    conda activate hest; python -u xen_pf_cut.py --h5ad $F --tiff "$TIF" --out $TMP/tiles_$k.h5 --start $s --end $e; conda deactivate
    conda activate tfpf; python -u xen_pf_embed_shard.py --h5 $TMP/tiles_$k.h5 --out $TMP/emb_$k.npy 2>&1 | grep -v "cuda_\|cpu_feature_guard\|rebuild TensorFlow\|TensorRT\|absl\|I0000\|W0000"; conda deactivate
    rm -f $TMP/tiles_$k.h5
  fi
  k=$((k+1)); s=$e
done
conda activate hest; python -c "
import numpy as np, glob, re; fs=sorted(glob.glob('$TMP/emb_*.npy'), key=lambda f:int(re.search(r'emb_(\d+)\.npy',f).group(1))); E=np.concatenate([np.load(f) for f in fs]); assert len(E)==$NB, (len(E), $NB); np.save('$OUT', E); print('saved', E.shape)"
rm -rf $TMP
