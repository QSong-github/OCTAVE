#!/bin/bash
#SBATCH --job-name=istar_prep
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 阶段 1/3: 数据准备(hest 环境, CPU 分区 —— hest 在 GPU 节点上 import anndata 就崩)
# 官方 iStar 的计算阶段单独提交 GPU 作业, 见 jobs/istar_gpu.sh
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore

for SLIDE in Visium_HD_Human_Colon_Cancer_P2 Visium_HD_Human_Colon_Cancer_P5; do
  TAG=$(echo $SLIDE | sed 's/.*_//')
  SHARED=/path/to/systema4ST/istar_run/${TAG}_shared
  mkdir -p $SHARED
  echo "===== [image] $SLIDE ====="
  python -u src/istar_prep.py image --slide $SLIDE --out ${SHARED}/
  for SPLIT in checker half; do
    D=/path/to/systema4ST/istar_run/${TAG}_${SPLIT}
    mkdir -p $D
    # 图像产物与划分无关: 从 shared 硬链接过去, 避免重复提特征
    for f in he-raw.jpg pixel-size-raw.txt pixel-size.txt level-downsample.txt heraw-dims.txt; do
      [ -f "$SHARED/$f" ] && ln -f "$SHARED/$f" "$D/$f"
    done
    echo "===== [fold] $SLIDE / $SPLIT ====="
    python -u src/istar_prep.py fold --slide $SLIDE --split $SPLIT --shared ${D}/ --out ${D}/
  done
done
echo "===== PREP DONE ====="
