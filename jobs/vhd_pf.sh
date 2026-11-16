#!/bin/bash
#SBATCH -J vhdpf
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 --array=0-1 -c 2 --mem=32G -t 24:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# Path Foundation on Visium HD P2/P5, two-stage as jobs/xen2_pf.sh (hest env cuts tiles, tfpf env runs the SavedModel on a GPU). Idempotent.
set -e
source /path/to/project/jobs/act.sh
cd /path/to/project; export PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
SEC=(Visium_HD_Human_Colon_Cancer_P2 Visium_HD_Human_Colon_Cancer_P5)
N=${SEC[$SLURM_ARRAY_TASK_ID]}; F=data/prepped/${N}_16um.h5ad; mkdir -p results/emb_vhd; OUT=results/emb_vhd/emb_path_foundation_${N}.npy; [ -s "$OUT" ] && exit 0
TIF=data/visiumhd/$N/${N}_PYRAMIDAL.tif
TMP=results/emb_vhd/pf_tmp_${N}; mkdir -p $TMP
act hest; NB=$(python -c "import h5py; h=h5py.File('$F','r'); print(h['obsm']['pxl'].shape[0])"); conda deactivate
SH_=20000; k=0; s=0
while [ $s -lt $NB ]; do e=$((s+SH_)); [ $e -gt $NB ] && e=$NB
  if [ ! -s $TMP/emb_$k.npy ]; then
    act hest; python -u xen_pf_cut.py --h5ad $F --tiff "$TIF" --out $TMP/tiles_$k.h5 --start $s --end $e; conda deactivate
    act tfpf; python -u xen_pf_embed_shard.py --h5 $TMP/tiles_$k.h5 --out $TMP/emb_$k.npy 2>&1 | grep -v "cuda_\|cpu_feature_guard\|rebuild TensorFlow\|TensorRT\|absl\|I0000\|W0000"; conda deactivate
    rm -f $TMP/tiles_$k.h5
  fi
  k=$((k+1)); s=$e
done
act hest; python -c "
import numpy as np, glob, re; fs=sorted(glob.glob('$TMP/emb_*.npy'), key=lambda f:int(re.search(r'emb_(\d+)\.npy',f).group(1))); E=np.concatenate([np.load(f) for f in fs]); assert len(E)==$NB, (len(E), $NB); np.save('$OUT', E); print('saved', E.shape)"
rm -rf $TMP
