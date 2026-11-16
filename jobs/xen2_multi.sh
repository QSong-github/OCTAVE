#!/bin/bash
#SBATCH -J xenmultiall
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --nice=50000
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# 第二批 6 区域 × 55 编码器的频带（与 xen_multi_an_tier.sh 同脚本同参数）；--mem/--array/-t 由提交时按区域与编码器维度分层给出；幂等。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
REG=(Xenium_V1_Human_Lung_Cancer_FFPE Xenium_Prime_Human_Lung_Cancer_FFPE Xenium_Prime_Breast_Cancer_FFPE Xenium_Prime_Human_Prostate_FFPE Xenium_Prime_Human_Skin_FFPE Xenium_Prime_Human_Lymph_Node_Reactive_FFPE)
ENC=(biomedclip ciga clip_vitl14 conch_v1 dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath_flash gpfm hibou_b hoptimus0 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco lunit_r50_swav lunit_vits16 lunit_vits8 mascaret mstar musk omiclip openmidnight pathgen_clip pathryoshka_b phaet phikon plip quiltnet retccl siglip2 uni_v1 virchow uni_v2 virchow2 hoptimus1 gigapath phikon_v2 conch_v15 kaiko_vits16 ctranspath h0_mini midnight12k)
I=$SLURM_ARRAY_TASK_ID; N=${REG[$((I % 6))]}; E=${ENC[$((I / 6))]}
[ -s results/emb_xen/emb_${E}_${N}.npy ] || { echo "[$E/$N] 无嵌入，跳过"; exit 0; }
[ -s results/blocks_xen_bands_${E}/${N}.json ] && exit 0
python -u src/blocks_xen_bands.py --name $N --tower $E --tag $E
