#!/bin/bash
#SBATCH --job-name=towersweep
#SBATCH --qos=YOUR_QOS
#SBATCH --array=0-24%10
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/project/logs/%x_%A_%a.out
# 数组作业: 每个任务一个图像塔。%10 限并发 10, 留出配额给 GPU 方法作业。
# 纯 CPU(Ridge + kNN 检索 + 图扩散), 不占 GPU。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=10
echo "节点 $(hostname)  数组任务 $SLURM_ARRAY_TASK_ID"
python -u src/tower_sweep.py --idx $SLURM_ARRAY_TASK_ID --hvg 50 --tmax 2048
