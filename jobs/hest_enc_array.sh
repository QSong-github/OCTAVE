#!/bin/bash
#SBATCH --job-name=hestenc
#SBATCH --qos=YOUR_QOS
#SBATCH --array=0-8%9
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%A_%a.out
# GPU 数组: 9 个开放权重编码器 × 72 个 HEST 样本, 一 GPU 一个编码器。
# 把 Visium HD 那条线的 25 塔结论扩展到广度线。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export HF_HOME=/path/to/systema4ST/.hf
# dinov2/dinov3 已缓存在 phenofm, 复用避免重下
export HF_HUB_CACHE=/path/to/systema4ST/.hf/hub
ENC=(ciga hibou_b dinov2_large dinov3_vitl16 lunit_vits8 kaiko_vitb16 kaiko_vits16 kaiko_vitl14 phikon)
E=${ENC[$SLURM_ARRAY_TASK_ID]}
echo "节点 $(hostname)  编码器 $E"; nvidia-smi --query-gpu=name --format=csv,noheader
python -u src/hest_embed.py --encoder $E
