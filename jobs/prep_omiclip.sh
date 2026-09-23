#!/bin/bash
#SBATCH --job-name=prepomi
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
pip install --no-cache-dir open_clip_torch 2>&1 | tail -3
python -c "import open_clip; print('open_clip', open_clip.__version__)"
mkdir -p methods/OmiCLIP
cd methods/OmiCLIP
if [ ! -s checkpoint.pt ]; then
  echo "下载 OmiCLIP checkpoint ..."
  curl -L --retry 3 -o checkpoint.pt \
    "https://huggingface.co/WangGuangyuLab/Loki/resolve/main/checkpoint.pt"
fi
ls -lh checkpoint.pt
python - <<'PY'
import torch, os
p = "/path/to/project/methods/OmiCLIP/checkpoint.pt"
print("大小 %.2f GB" % (os.path.getsize(p) / 2**30))
from open_clip import create_model_from_pretrained
m, pre = create_model_from_pretrained("coca_ViT-L-14", device="cpu", pretrained=p)
x = torch.stack([torch.randn(3, 224, 224)])
with torch.no_grad():
    f = m.encode_image(x)
print("encode_image 输出", tuple(f.shape), " preprocess:", pre)
PY
echo "######## OMICLIP READY ########"
