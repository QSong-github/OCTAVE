#!/bin/bash
#SBATCH --job-name=hggepb2
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
# 小显存 GPU 溢出：作者的 build_adj_hypergraph 对每个节点算全图欧氏距离，
# 他们在 HER2ST 上每片 300-700 点，本协议是 MAXSPOT=4000。改 MAXSPOT 会破坏
# 与其它方法的可比性，所以换大卡而不是改协议。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u src/hggep_hest.py --cohorts HCC,LUNG --match_steps 3200 \
    --out results/hggep_smoke.json
echo "######## HGGEP SMOKE DONE ########"
