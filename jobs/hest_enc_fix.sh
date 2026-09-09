#!/bin/bash
#SBATCH --job-name=hestencfix
#SBATCH --qos=YOUR_QOS
#SBATCH --array=0-1
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%A_%a.out
# 修两个失败: dinov3 权重其实已缓存在 phenofm(我把 HF_HUB_CACHE 指错了);
#            kaiko_vitl14 需要 518 输入而非 224。
# hibou_b 也是 gated(401), 需你的 HF 授权, 本轮跳过。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export HF_HOME=/path/to/systema4ST/.hf
# 关键: 指向已有 dinov3/dinov2 权重的缓存
export HF_HUB_CACHE=/path/to/phenofm/hf_cache/hub
export HF_HUB_OFFLINE=1
ENC=(dinov3_vitl16 kaiko_vitl14)
E=${ENC[$SLURM_ARRAY_TASK_ID]}
echo "节点 $(hostname)  编码器 $E"
[ "$E" = "kaiko_vitl14" ] && unset HF_HUB_OFFLINE && export HF_HUB_CACHE=/path/to/systema4ST/.hf/hub
python -u src/hest_embed.py --encoder $E
