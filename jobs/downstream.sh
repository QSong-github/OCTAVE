#!/bin/bash
#SBATCH --job-name=hestdown
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
# 计算节点专用。CPU 分区(hest 环境在 GPU 节点上 import anndata 会崩)。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=32
echo "节点: $(hostname)  作业: $SLURM_JOB_ID"
for e in phikon phikon_v2; do
  [ -f results/hest_emb/INT1_${e}.npz ] || { echo "[skip] $e 无缓存嵌入"; continue; }
  echo "================= $e ================="
  python -u src/hest_downstream.py --encoder $e --tmax 8192
done
echo "######## DOWNSTREAM DONE ########"
