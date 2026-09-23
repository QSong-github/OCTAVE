#!/bin/bash
#SBATCH --job-name=betads
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/project/logs/%x_%A_%a.out
# 补算每个 (区域, 分箱) 的 β1；PCC 与冻结值不符即中止，保证重现的是同一批预测。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=8
mapfile -t N < <(ls results/downstream_*.json | grep -v summary | xargs -n1 basename \
                 | sed 's/^downstream_//; s/\.json$//')
echo "节点 $(hostname)  区域 ${N[$SLURM_ARRAY_TASK_ID]}"
python -u src/beta_vs_pcc.py --name "${N[$SLURM_ARRAY_TASK_ID]}"
