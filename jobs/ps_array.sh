#!/bin/bash
#SBATCH --job-name=effps
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-default
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
# 逐样本带 PCC：15 个编码器 × 72 个 HEST 样本。与 hest_effres.py 完全同算法，
# 只是额外保存逐样本值，使"距形态学下界的差距"能做队列层级配对符号检验。
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
ENC=(ciga conch_v15 dinov2_large dinov3_vitl16 gigapath hoptimus0 kaiko_vitb16 kaiko_vitl14 kaiko_vits16 lunit_vits8 midnight12k phikon phikon_v2 uni_v2 virchow2)
E=${ENC[$SLURM_ARRAY_TASK_ID]}
echo "节点 $(hostname)  编码器 $E"
python -u src/hest_effres_ps.py --encoder $E --out results/hest_effres_ps_$E.json
