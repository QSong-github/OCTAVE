#!/bin/bash
#SBATCH -J dshest
#SBATCH --qos=YOUR_QOS --partition=YOUR_GPU_PARTITION --gres=gpu:1 --array=0-3 -c 6 --mem=64G -t 8:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
# DeepSpot 在 HEST 上：2 个编码器 × 2 种训练长度（作者 notebook 的 max_epochs=10；按作者步数 500 反推）。幂等（按样本续跑）。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate deepspot
cd /path/to/project; export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 PYTHONWARNINGS=ignore PYTHONPATH=/path/to/project/methods/DeepSpot:${PYTHONPATH:-}
ENC=(uni_v1 hoptimus0); I=$SLURM_ARRAY_TASK_ID; E=${ENC[$((I % 2))]}
if [ $((I / 2)) -eq 0 ]; then python -u src/deepspot_hest.py --encoder $E; else python -u src/deepspot_hest.py --encoder $E --match_steps 500 --tag _steps500; fi
