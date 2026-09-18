#!/bin/bash
#SBATCH --job-name=omips
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
# 显式给 --out：脚本默认写 hest_effres_<enc>.json，而下游读的是带 _ps_ 的名字
python -u src/hest_effres_ps.py --encoder omiclip --out results/hest_effres_ps_omiclip.json
echo "######## OMICLIP PS DONE ########"
