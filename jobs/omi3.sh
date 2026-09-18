#!/bin/bash
#SBATCH --job-name=omi3
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 上次 checkpoint 只下到 4.5/7.1 GB 就被截断（zipfile.is_zipfile 为 False）。
# 这次断点续传并按 HF 报的字节数校验，不足就判失败，不让半截文件进管线。
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
pip install --no-cache-dir --no-deps open_clip_torch 2>&1 | tail -1
pip install --no-cache-dir ftfy regex 2>&1 | tail -1
python -c "import timm, open_clip; print('timm', timm.__version__, '| open_clip', open_clip.__version__)"
WANT=7662370634
CK=methods/OmiCLIP/checkpoint.pt
for a in 1 2 3; do
  HAVE=$(stat -c%s "$CK" 2>/dev/null || echo 0)
  [ "$HAVE" = "$WANT" ] && break
  echo "第 $a 次续传：已有 $HAVE / $WANT"
  curl -L -C - --retry 5 --retry-delay 10 -o "$CK" \
    "https://huggingface.co/WangGuangyuLab/Loki/resolve/main/checkpoint.pt"
done
HAVE=$(stat -c%s "$CK")
echo "最终 $HAVE / $WANT"
[ "$HAVE" = "$WANT" ] || { echo "大小不符，中止，不让半截文件进管线"; exit 1; }
python -c "
import zipfile, torch
p='methods/OmiCLIP/checkpoint.pt'
assert zipfile.is_zipfile(p), 'zip 校验失败'
from open_clip import create_model_from_pretrained
m, pre = create_model_from_pretrained('coca_ViT-L-14', device='cpu', pretrained=p)
with torch.no_grad(): f = m.encode_image(torch.randn(1,3,224,224))
print('encode_image 输出', tuple(f.shape))
"
python -u src/hest_embed_v2.py --encoder omiclip --batch 96
echo "######## OMICLIP DONE ########"
