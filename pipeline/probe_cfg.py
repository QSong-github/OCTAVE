# -*- coding: utf-8 -*-
"""CPU 探针 + 打印 timm data config（核对 mean/std 与模型卡一致）。"""
import sys, time, numpy as np, torch, timm
from timm.data import resolve_model_data_config
sys.path.insert(0, "/path/to/systema4ST/src")
import hest_embed_v2 as H
P = "/path/to/he2st/HEST/eval/bench_data/CCRCC/patches/INT1.h5"
torch.set_num_threads(4)
for name in sys.argv[1:]:
    t0 = time.time()
    try:
        m, kind = H.encoder(name, "cpu")
        cfg = resolve_model_data_config(m) if isinstance(m, torch.nn.Module) and hasattr(m, "pretrained_cfg") else {}
        n = sum(p.numel() for p in m.parameters())
        X, bc = H.embed_and_barcodes(m, kind, P, "cpu", bs=32)
        print("  ✓ %-17s X=%s |mean|=%.3f std=%.3f nan=%d params=%.1fM in=%s mean=%s std=%s %.0fs" % (
            name, X.shape, abs(X.mean()), X.std(), int(np.isnan(X).sum()), n / 1e6,
            cfg.get("input_size"), cfg.get("mean"), cfg.get("std"), time.time() - t0), flush=True)
    except Exception as e:
        print("  ✗ %-17s %s: %s" % (name, type(e).__name__, str(e)[:300]), flush=True)
