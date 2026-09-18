#!/bin/bash
#SBATCH --job-name=istar_gpu
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 阶段 2/3: 官方 iStar 计算(istar 环境, GPU)。全程只用 istar env, 不碰 hest ——
# hest 在 GPU 分区上 import anndata 会崩, 所以数据准备/评测都拆到 CPU 作业。
# 图像侧(rescale/preprocess/HIPT 特征/mask)与 train-test 划分无关, 每片只做一次,
# 再把产物硬链接过继给该片的两个 fold, 省掉一半 GPU 时间。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/istar
ISTAR=/path/to/istar_workspace        # 只读: 官方仓库
RUN=/path/to/systema4ST/istar_run
export PYTHONWARNINGS=ignore
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for TAG in P2 P5; do
  S=${RUN}/${TAG}_shared/
  echo "########## [$TAG] 图像侧(只做一次) ##########"
  cd $ISTAR
  python -u rescale.py ${S} --image
  python -u preprocess.py ${S} --image
  python -u extract_features.py ${S} --device=cuda
  python -u get_mask.py ${S}embeddings-hist.pickle ${S}mask-small.png
  for SPLIT in checker half; do
    D=${RUN}/${TAG}_${SPLIT}/
    echo "########## [$TAG/$SPLIT] 过继图像产物 + 训练/超分 ##########"
    cp -aln ${S}. ${D} || true          # -n: 已存在的(he-raw.jpg 等)不覆盖
    cd $ISTAR
    python -u rescale.py ${D} --locs --radius
    python -u impute.py ${D} --epochs=400 --device=cuda
    echo "===== [$TAG/$SPLIT] 完成, cnts-super: $(ls ${D}cnts-super 2>/dev/null | wc -l) 个基因 ====="
  done
done
echo "===== ISTAR GPU DONE ====="
