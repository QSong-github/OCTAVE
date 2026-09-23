#!/bin/bash
#SBATCH --job-name=omidiag
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
python -c "import timm, open_clip; print('timm', timm.__version__, '| open_clip', open_clip.__version__)"
WANT=7662370634
CK=methods/OmiCLIP/checkpoint.pt
HAVE=$(stat -c%s "$CK")
echo "checkpoint $HAVE / $WANT"
[ "$HAVE" = "$WANT" ] || { echo "大小不符"; exit 1; }
python -u omi_diag.py; exit 0
