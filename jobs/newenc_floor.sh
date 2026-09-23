#!/bin/bash
#SBATCH --job-name=nfloor
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-479
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/project/logs/%x_%A_%a.out
# 新编码器的匹配检索地板：12 编码器 × 10 队列 × 4 个 k = 480。
# k=50 落在 hest_floor_<E>/，其余落在 hest_floor_k<K>_<E>/，与既有命名一致。
# 不设并发上限；QOS 的 CPU 配额是唯一约束。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
ENC=(uni_v1 virchow conch_v1 keep openmidnight hibou_l h0_mini plip quiltnet hoptimus1 genbio_pathfm omiclip)
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM)
KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID
K=${KS[$((I / 120))]}
J=$((I % 120))
E=${ENC[$((J / 10))]}
C=${COH[$((J % 10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_${E}; else OUT=results/hest_floor_k${K}_${E}; fi
# 嵌入未抽齐的编码器直接跳过，等它抽完再补跑本任务
N=$(ls results/hest_emb/*_${E}.npz 2>/dev/null | wc -l)
if [ "$N" -lt 72 ]; then echo "跳过 $E：嵌入只有 $N/72"; exit 0; fi
mkdir -p $OUT
if [ -s "$OUT/$C.json" ]; then echo "已存在，跳过 $E/$C k=$K"; exit 0; fi
echo "节点 $(hostname)  编码器 $E  队列 $C  k=$K"
python -u src/hest_floor.py --encoder "$E" --cohort "$C" --k "$K" --out "$OUT/$C.json"
