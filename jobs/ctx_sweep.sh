#!/bin/bash
#SBATCH --job-name=ctxsweep
#SBATCH --qos=qsong1
#SBATCH --array=0-11%8
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
# 上下文尺寸扫描: hibou_l_ctx320/448/640/896/1344、phikon_ctx448、dinov3_ctx448 及 grid 变体。
# 问题: 给编码器更大的空间上下文, 有效分辨率是变细还是变粗?
# 若"上下文越大→等价σ越粗", 就为低通行为给出了机制性解释, 而不只是现象描述。
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=10
python -u src/tower_sweep.py --mode ctx --idx $SLURM_ARRAY_TASK_ID --outdir results/ctx_sweep
