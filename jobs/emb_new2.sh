#!/bin/bash
#SBATCH --job-name=embnew2
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%A_%a.out
# 五个需要下载的新编码器。各自带预处理，不套 ImageNet 统计量。
# 不设 HF_HUB_OFFLINE，这几个要现下权重。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
ENC=(hoptimus1 h0_mini genbio_pathfm plip quiltnet)
E=${ENC[$SLURM_ARRAY_TASK_ID]}
echo "节点 $(hostname)  编码器 $E"; nvidia-smi --query-gpu=name --format=csv,noheader
python -u src/hest_embed_v2.py --encoder "$E" --batch 128
