#!/bin/bash
#SBATCH -J xenmultiall
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 -t 6:00:00 --nice=50000
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
# 内存按区域分层提交（sbatch --mem 与 --array 由 restart_tiers.sh 按区域峰值 RSS 给出），作业名保持 xenmultiall 以便调速器统一节流。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
ENC=(biomedclip ciga clip_vitl14 conch_v1 dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath_flash gpfm hibou_b hoptimus0 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco lunit_r50_swav lunit_vits16 lunit_vits8 mascaret mstar musk omiclip openmidnight pathgen_clip pathryoshka_b phaet phikon plip quiltnet retccl siglip2 uni_v1 virchow)
I=$SLURM_ARRAY_TASK_ID; N=${REG[$((I % 16))]}; E=${ENC[$((I / 16))]}
[ -s results/emb_xen/emb_${E}_${N}.npy ] || { echo "[$E/$N] 无嵌入，跳过"; exit 0; }
[ -s results/blocks_xen_bands_${E}/${N}.json ] && exit 0
python -u src/blocks_xen_bands.py --name $N --tower $E --tag $E
