#!/bin/bash
#SBATCH --job-name=heclipfull
#SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 冒烟已过（损失 11.3->0.26，无零值）。全量 10 队列，逐折落盘可续跑。
# 小队列在前，先拿覆盖面，再啃 PRAD/CCRCC。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
echo "节点 $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u src/heclip_hest.py --cohorts SKCM,HCC,LUNG,PAAD,COAD,READ,IDC,LYMPH_IDC \
    --out results/heclip_hest.json
python -u src/heclip_hest.py --cohorts PRAD,CCRCC \
    --out results/heclip_hest.json
echo "######## HECLIP FULL DONE ########"
