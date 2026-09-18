#!/bin/bash
#SBATCH --job-name=flrk
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-449
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%A_%a.out
# 配对下界的 k 敏感性：§46 的「7/15 编码器低于自身检索下界」是否依赖 k=50 这个超参。
# 审计已指出深度线的 image_floor（余弦+softmax，PCA-64）与广度线的（欧氏+均匀，PCA-256）
# 是两个不同估计器；此处只动 k，其余完全不变，看结论稳不稳。
# 15 编码器 × 10 队列 × 3 个 k = 450 任务，不设并发上限。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
ENC=(ciga conch_v15 dinov2_large dinov3_vitl16 gigapath hoptimus0 kaiko_vitb16 kaiko_vitl14 kaiko_vits16 lunit_vits8 midnight12k phikon phikon_v2 uni_v2 virchow2)
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM)
KS=(10 200 800)
I=$SLURM_ARRAY_TASK_ID
K=${KS[$((I / 150))]}
J=$((I % 150))
E=${ENC[$((J / 10))]}
C=${COH[$((J % 10))]}
OUT=results/hest_floor_k${K}_${E}
mkdir -p $OUT
if [ -f "$OUT/$C.json" ]; then echo "已存在，跳过 $E/$C k=$K"; exit 0; fi
echo "节点 $(hostname)  编码器 $E  队列 $C  k=$K"
python -u src/hest_floor.py --cohort $C --encoder $E --k $K --out $OUT/$C.json
