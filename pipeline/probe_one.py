# -*- coding: utf-8 -*-
"""CPU 探针：给定编码器名，走真实的 encoder()+embed_and_barcodes() 跑最小的一片。"""
import sys, time, numpy as np, torch
sys.path.insert(0, "/path/to/project/src")
import hest_embed_v2 as H
P = "/path/to/he2st/HEST/eval/bench_data/CCRCC/patches/INT1.h5"
torch.set_num_threads(4)
for name in sys.argv[1:]:
    t0 = time.time()
    try:
        m, kind = H.encoder(name, "cpu")
        n = sum(p.numel() for p in m.parameters())
        X, bc = H.embed_and_barcodes(m, kind, P, "cpu", bs=32)
        print("  ✓ %-15s X=%s  |mean|=%.3f std=%.3f  nan=%d  params=%.1fM  kind=%s  %.0fs" % (
            name, X.shape, abs(X.mean()), X.std(), int(np.isnan(X).sum()), n / 1e6,
            kind if not isinstance(kind, tuple) else kind[0], time.time() - t0), flush=True)
    except Exception as e:
        print("  ✗ %-15s %s: %s" % (name, type(e).__name__, str(e)[:300]), flush=True)
