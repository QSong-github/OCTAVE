# -*- coding: utf-8 -*-
"""检验本地 HF 缓存里的 gated 权重能否离线加载 —— 若可以，HEST 广度线就能补齐旗舰塔。"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from trident.patch_encoder_models.load import encoder_factory
CAND = ["uni_v1", "uni_v2", "virchow", "virchow2", "gigapath", "conch_v1", "conch_v15",
        "hoptimus0", "keep", "gpfm", "midnight12k", "openmidnight", "phikon_v2", "hibou_l"]
ok, bad = [], []
for e in CAND:
    try:
        m = encoder_factory(e)
        d = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  ✔ {e:14s} {d:7.1f}M 参数", flush=True)
        ok.append(e); del m; torch.cuda.empty_cache()
    except Exception as ex:
        print(f"  ✘ {e:14s} {type(ex).__name__}: {str(ex)[:100]}", flush=True)
        bad.append(e)
print(f"\n可离线加载 {len(ok)}/{len(CAND)}: {' '.join(ok)}")
if bad: print(f"不可用: {' '.join(bad)}")
