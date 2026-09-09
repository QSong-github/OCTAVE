#!/bin/bash
#SBATCH --job-name=flrmx
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-default
#SBATCH --array=0-149
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%A_%a.out
# 配对形态学下界：每个编码器用它**自己的**特征做 kNN 检索下界。
# 之前只用 phikon_v2 特征做了一条下界，拿去和全部 15 个编码器比 ——
# 弱编码器输给它只说明特征差，不能归因于"回归 vs 检索"。
# 15 编码器 × 10 队列 = 150 个任务，不设并发上限。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
ENC=(ciga conch_v15 dinov2_large dinov3_vitl16 gigapath hoptimus0 kaiko_vitb16 kaiko_vitl14 kaiko_vits16 lunit_vits8 midnight12k phikon phikon_v2 uni_v2 virchow2)
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM)
I=$SLURM_ARRAY_TASK_ID
E=${ENC[$((I / 10))]}
C=${COH[$((I % 10))]}
mkdir -p results/hest_floor_$E
echo "节点 $(hostname)  编码器 $E  队列 $C"
python -u src/hest_floor.py --cohort $C --encoder $E --out results/hest_floor_$E/$C.json
