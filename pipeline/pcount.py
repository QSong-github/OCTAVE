# -*- coding: utf-8 -*-
"""逐个编码器数真实参数量：用与 hest_embed_v2.py 完全相同的加载路径，数实际被前向用到的那个塔。"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/blue/qsong1/wang.qing/systema4ST/src")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from hest_embed_v2 import encoder as load_encoder, ALL_ENC
out = {}
for e in sorted(set(ALL_ENC) - {"omiclip_raw", "resnet50", "hibou_b", "gpfm"}):
    try:
        m, kind = load_encoder(e, "cpu")
        n = sum(p.numel() for p in m.parameters())
        d = getattr(getattr(m, "config", None), "hidden_size", None)
        out[e] = {"params": int(n), "hidden": d}
        print("  %-16s %9.1f M   %s" % (e, n / 1e6, type(m).__name__), flush=True)
        del m
    except Exception as ex:
        print("  %-16s 失败: %s" % (e, str(ex)[:60]), flush=True)
json.dump(out, open("/blue/qsong1/wang.qing/systema4ST/results/encoder_params.json", "w"), indent=1)
print("\n已存 results/encoder_params.json （%d 个）" % len(out))
