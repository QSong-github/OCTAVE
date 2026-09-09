#!/bin/bash
# 网格依赖性检验：σ 会不会只是 16µm 分箱本身的产物？
# 做法：同一批数据换分箱尺度重做。Xenium 单细胞可往细里压（8µm），
# 两个平台都可往粗里放（32/64µm）。若 σ 减去噪声地板在各尺度上稳定，
# 则 σ 测的是模型性质，不是网格的产物。
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
BINS=(8 32 64)
B=${BINS[$SLURM_ARRAY_TASK_ID]}
echo "=== Xenium 重新分箱 @ ${B}µm ==="
python -u src/xen_prep.py --bin_um $B 2>&1 | grep -E "✔|✘|细胞="
