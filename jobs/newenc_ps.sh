#!/bin/bash
#SBATCH --job-name=nps
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-default
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=4
ENC=(uni_v1 virchow conch_v1 keep openmidnight hibou_l h0_mini plip quiltnet hoptimus1 genbio_pathfm omiclip)
E=${ENC[$SLURM_ARRAY_TASK_ID]}
N=$(ls results/hest_emb/*_${E}.npz 2>/dev/null | wc -l)
if [ "$N" -lt 72 ]; then echo "跳过 $E：嵌入只有 $N/72"; exit 0; fi
[ -s "results/hest_effres_ps_${E}.json" ] && { echo "已存在，跳过 $E"; exit 0; }
echo "节点 $(hostname)  编码器 $E"
python -u src/hest_effres_ps.py --encoder "$E"
