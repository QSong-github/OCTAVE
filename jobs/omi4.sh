#!/bin/bash
#SBATCH --job-name=omi4
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=hpg-b200,hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
python -c "import timm, open_clip; print('timm', timm.__version__, '| open_clip', open_clip.__version__)"
WANT=7662370634
CK=methods/OmiCLIP/checkpoint.pt
HAVE=$(stat -c%s "$CK")
echo "checkpoint $HAVE / $WANT"
[ "$HAVE" = "$WANT" ] || { echo "大小不符"; exit 1; }
python - <<'PY'
import zipfile, torch, numpy as np
p = "methods/OmiCLIP/checkpoint.pt"
assert zipfile.is_zipfile(p), "zip 校验失败"
# 与 hest_embed_v2.py 里同一处理：只放行这一个 global，不整体关掉 weights_only
pass
from open_clip import create_model_from_pretrained
m, pre = create_model_from_pretrained("coca_ViT-L-14", device="cpu", pretrained=p, weights_only=False)
with torch.no_grad():
    f = m.encode_image(torch.randn(1, 3, 224, 224))
print("encode_image 输出", tuple(f.shape))
print("preprocess:", pre)
PY
python -u src/hest_embed_v2.py --encoder omiclip --batch 96
echo "######## OMICLIP DONE ########"
