#!/bin/bash
#SBATCH --job-name=hgprobe
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
# 只读代码, 不跑计算。三个方法同源(HisToGene → Hist2ST → THItoGene), 一次看清:
# 模型构造签名、数据集接口、训练入口、以及各自对输入形状的硬编码假设。
set -u
M=/path/to/project/methods
for k in HisToGene Hist2ST THItoGene; do
  echo "################################ $k ################################"
  echo "--- 模型类签名 ---"
  grep -nE "^class |def __init__" $M/$k/vis_model.py $M/$k/HIST2ST.py 2>/dev/null | head -12
  echo "--- 数据集接口(构造+getitem 返回) ---"
  grep -nE "^class |def __init__|def __getitem__|return |self\.(exp|im|center|loc|adj|patch)" $M/$k/dataset.py 2>/dev/null | head -22
  echo "--- 训练/预测入口 ---"
  grep -nE "def |load_from_checkpoint|Trainer|fit\(|predict|n_genes|n_layers|patch_size|dim" $M/$k/predict.py 2>/dev/null | head -14
  echo "--- 对 spot 数/基因数的硬编码 ---"
  grep -rnE "n_genes *= *[0-9]+|n_pos *= *[0-9]+|=785|=3467|=785|patch_size *= *[0-9]+" $M/$k/*.py 2>/dev/null | head -8
done
echo "######## PROBE DONE ########"
