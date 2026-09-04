#!/bin/bash
#SBATCH --job-name=omi2
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 上一轮 pip 装 open_clip 时试图改 timm，OSError 中止（timm 已验证完好）。
# 这次用 --no-deps，绝不碰 timm —— kaiko/hoptimus1/h0_mini/virchow 都靠它。
set -u
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
pip install --no-cache-dir --no-deps open_clip_torch 2>&1 | tail -2
pip install --no-cache-dir ftfy regex 2>&1 | tail -1
python -c "import timm, open_clip; print('timm', timm.__version__, '| open_clip', open_clip.__version__)"
python - <<'PY'
import os, torch
p = "/blue/qsong1/wang.qing/systema4ST/methods/OmiCLIP/checkpoint.pt"
print("checkpoint %.2f GB" % (os.path.getsize(p) / 2**30))
from open_clip import create_model_from_pretrained
m, pre = create_model_from_pretrained("coca_ViT-L-14", device="cpu", pretrained=p)
with torch.no_grad():
    f = m.encode_image(torch.randn(1, 3, 224, 224))
print("encode_image 输出", tuple(f.shape))
PY
python -u src/hest_embed_v2.py --encoder omiclip --batch 96
echo "######## OMICLIP EMB DONE ########"
