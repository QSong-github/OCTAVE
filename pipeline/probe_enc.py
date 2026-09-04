#!/usr/bin/env python
"""探针：逐个尝试加载候选编码器并做一次前向，报告哪些可用、输出维度多少。
不写任何结果文件，只打印。加载失败的原因原样打印，便于判断是权限还是接口。"""
import os, sys, traceback
import numpy as np, torch

CAND = [
    # (名字, 类型, 仓库或标识)
    ("genbio_pathfm",  "hf",     "genbio-ai/genbio-pathfm"),
    ("plip",           "clip",   "vinid/plip"),
    ("quiltnet",       "clip",   "wisdomik/QuiltNet-B-32"),
    ("openmidnight",   "hf",     "kaiko-ai/midnight"),
    ("hibou_l",        "hf",     "histai/hibou-L"),
    ("hibou_b",        "hf",     "histai/hibou-b"),
    ("uni_v1",         "hf",     "MahmoodLab/UNI"),
    ("virchow",        "hf",     "paige-ai/Virchow"),
    ("conch_v1",       "hf",     "MahmoodLab/conch"),
    ("h0_mini",        "hf",     "bioptimus/H0-mini"),
    ("hoptimus1",      "hf",     "bioptimus/H-optimus-1"),
]
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(2, 3, 224, 224).to(dev)
print("device=%s  torch=%s  HF_HUB_OFFLINE=%s\n" % (dev, torch.__version__, os.environ.get("HF_HUB_OFFLINE")))
ok, bad = [], []
for name, kind, repo in CAND:
    try:
        if kind == "clip":
            from transformers import CLIPModel
            m = CLIPModel.from_pretrained(repo).eval().to(dev)
            with torch.no_grad():
                f = m.get_image_features(pixel_values=x)
        else:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(repo, trust_remote_code=True).eval().to(dev)
            with torch.no_grad():
                o = m(x)
                f = o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else \
                    (o.pooler_output if hasattr(o, "pooler_output") else o)
        d = tuple(f.shape)
        print("  OK  %-16s %-34s 输出 %s" % (name, repo, d), flush=True)
        ok.append((name, repo, d[-1]))
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        msg = str(e).split("\n")[0][:120]
        print("  --  %-16s %-34s %s" % (name, repo, msg), flush=True)
        bad.append((name, repo, msg))
print("\n可用 %d 个: %s" % (len(ok), [o[0] for o in ok]))
print("不可用 %d 个: %s" % (len(bad), [b[0] for b in bad]))
