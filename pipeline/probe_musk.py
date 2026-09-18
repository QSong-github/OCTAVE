#!/usr/bin/env python
"""MUSK 探针。加载与前向参数全部取自作者 README：
   384x384、Inception mean/std、with_head=False, out_norm=False, ms_aug=True, return_global=True
   （作者指明这组用于线性探针/MIL，正是本文岭回归的用法）。"""
import sys, torch
sys.path.insert(0, "/path/to/methods/MUSK")
from musk import utils, modeling          # noqa: F401  注册 timm 模型
from timm.models import create_model
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = create_model("musk_large_patch16_384")
utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", m, "model|module", "")
m.to(device=dev, dtype=torch.float16).eval()
x = torch.randn(2, 3, 384, 384).to(dev, dtype=torch.float16)
with torch.inference_mode():
    f = m(image=x, with_head=False, out_norm=False, ms_aug=True, return_global=True)[0]
print("MUSK 输出", tuple(f.shape), " dtype", f.dtype)
print("行范数中位 %.4f" % float(f.float().norm(dim=1).median()))
