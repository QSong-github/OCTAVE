#!/bin/bash
#SBATCH -J istarxen
#SBATCH --qos=YOUR_QOS --gres=gpu:l4:1 --array=0-15 -c 8 --mem=96G -t 24:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
# 官方 iStar（只读仓库，istar 环境）在 Xenium 区域上：图像侧一次 + half 折训练/超分。协议同 jobs/istar_gpu.sh。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate /path/to/miniconda3/envs/istar
ISTAR=/path/to/spatial2exp/iStar; RUN=/path/to/systema4ST/istar_run; export PYTHONWARNINGS=ignore TQDM_DISABLE=1
REG=(Human_Breast_Biomarkers_S1_Bot Human_Breast_Biomarkers_S1_Mid Human_Breast_Biomarkers_S1_Top Human_Breast_Biomarkers_S2_Bot Human_Breast_Biomarkers_S2_Mid Human_Breast_Biomarkers_S2_Top Human_Breast_Biomarkers_S3_Bot Human_Breast_Biomarkers_S3_Mid Human_Breast_Biomarkers_S3_Top Human_Breast_Biomarkers_S4_Bot Human_Breast_Biomarkers_S4_Mid Human_Breast_Biomarkers_S4_Top Xenium_Prime_Cervical_Cancer_FFPE Xenium_Prime_Ovarian_Cancer_FFPE_XRrun Xenium_V1_Human_Kidney_FFPE_Protein_updated Xenium_V1_Human_Ovary_Cancer_FF)
N=${REG[$SLURM_ARRAY_TASK_ID]}; S=${RUN}/xen_${N}_shared/; D=${RUN}/xen_${N}_half/
[ -s ${D}test.npz ] || { echo "prep 未完成"; exit 1; }
[ $(ls ${D}cnts-super 2>/dev/null | wc -l) -ge 200 ] && { echo "已完成"; exit 0; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $ISTAR
[ -s ${S}embeddings-hist.pickle ] || { python -u rescale.py ${S} --image; python -u preprocess.py ${S} --image; python -u extract_features.py ${S} --device=cuda; python -u get_mask.py ${S}embeddings-hist.pickle ${S}mask-small.png; }
cp -aln ${S}. ${D} || true
python -u rescale.py ${D} --locs --radius
python -u impute.py ${D} --epochs=400 --device=cuda
echo "===== [$N] 完成, cnts-super: $(ls ${D}cnts-super | wc -l) 个基因 ====="
