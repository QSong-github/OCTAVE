#!/usr/bin/env python
"""第三轮探针：2026 年的 Phaet / Mascaret，以及 MUSK（Ruijiang Li 组）。"""
import torch, timm
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(2, 3, 224, 224).to(dev)
ok, bad = [], []


def try_(name, fn, pool="auto"):
    try:
        m = fn()
        m = m.eval().to(dev)
        with torch.no_grad():
            o = m(x)
        f = o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else o
        if isinstance(f, (tuple, list)): f = f[0]
        d = tuple(f.shape)
        print("  OK  %-16s 输出 %s" % (name, d), flush=True)
        ok.append((name, d))
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print("  --  %-16s %s" % (name, str(e).split("\n")[0][:120]), flush=True)
        bad.append(name)


print("timm", timm.__version__, "\n")
# Phaet 是 Phikon-v2 的微调版（ViT-L/16），Mascaret 是 Midnight-12k 的微调版
from transformers import AutoModel
try_("phaet_hf",   lambda: AutoModel.from_pretrained("wearewaiv/phaet", trust_remote_code=True))
try_("mascaret_hf", lambda: AutoModel.from_pretrained("wearewaiv/mascaret", trust_remote_code=True))
try_("phaet_timm", lambda: timm.create_model("hf-hub:wearewaiv/phaet", pretrained=True, num_classes=0))
try_("mascaret_timm", lambda: timm.create_model("hf-hub:wearewaiv/mascaret", pretrained=True, num_classes=0))
# MUSK：作者用 timm 注册的自定义模型；先试通用路径
try_("musk_timm", lambda: timm.create_model("hf-hub:xiangjx/musk", pretrained=True, num_classes=0))
try_("musk_hf",   lambda: AutoModel.from_pretrained("xiangjx/musk", trust_remote_code=True))
print("\n可用: %s" % [o[0] for o in ok])
print("不可用: %s" % bad)
