# -*- coding: utf-8 -*-
"""19 个新编码器的参数量：与 pcount/pcount2 同一口径——加载后对象的全部 parameters() 求和。path_foundation 是 TF，另由 pcount_pf.py 计。"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/path/to/systema4ST/src")
import torch
from hest_embed_v2 import encoder as load_encoder
P = "/path/to/systema4ST/results/encoder_params.json"
NEW = sys.argv[1:] or ["biomedclip", "clip_vitl14", "ctranspath", "dinov2_base", "dinov2_giant", "dinov3_vitb16", "dinov3_vith16",
       "gigapath_flash", "gpfm", "hibou_b", "kaiko_vitb8", "lunit_r50_bt", "lunit_r50_moco", "lunit_r50_swav",
       "lunit_vits16", "pathgen_clip", "retccl", "siglip2"]
for e in NEW:
    out = json.load(open(P))
    if e in out: print("  %-14s 已有" % e); continue
    try:
        m, kind = load_encoder(e, "cpu")
        n = sum(p.numel() for p in m.parameters())
        out[e] = {"params": int(n), "hidden": getattr(getattr(m, "config", None), "hidden_size", None)}
        json.dump(out, open(P, "w"), indent=1)          # 逐个落盘，中途失败不丢
        print("  %-14s %9.1f M  %s" % (e, n / 1e6, type(m).__name__), flush=True); del m
    except Exception as ex:
        print("  %-14s 失败: %s" % (e, str(ex)[:90]), flush=True)
print("\n共 %d 个" % len(json.load(open(P))))
