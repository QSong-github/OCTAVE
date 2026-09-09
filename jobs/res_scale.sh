#!/bin/bash
#SBATCH --job-name=resscale
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=32
echo "节点 $(hostname)"
python -u src/res_scaling.py --ums 2,4,8,16,32 --hvg 50 --tmax 1024
