# -*- coding: utf-8 -*-
"""邻居 patch 的完整性检查：作者的 save_patches 见文件即跳过，被中断留下的半截 h5 会被当成完整的。
合格条件：可打开、含 img/coords/barcode、attrs['matched_to_target'] 为真、条码数与目标 patch 一致。不合格者删除以便重切。"""
import sys, os, glob, h5py, numpy as np
d = f"/path/to/systema4ST/data/triplex/{sys.argv[1]}"; bad = 0; ok = 0
for p in sorted(glob.glob(f"{d}/patches/*.h5")):
    sid = os.path.basename(p)[:-3]; n = f"{d}/patches/neighbor/{sid}.h5"
    if not os.path.exists(n): continue
    try:
        with h5py.File(p, "r") as h: bt = set(np.asarray(h["barcode"][:]).flatten().astype(str))   # 目标 patch 可能已下采样，故用子集判定
        with h5py.File(n, "r") as h:
            bn = set(np.asarray(h["barcode"][:]).flatten().astype(str))
            good = all(k in h for k in ("img", "coords", "barcode")) and bool(h.attrs.get("matched_to_target", False)) and bt.issubset(bn)
    except Exception: good = False
    if good: ok += 1
    else: os.remove(n); bad += 1; print(f"  删除不完整的邻居块 {sid}", flush=True)
print(f"{sys.argv[1]}: 合格 {ok}，删除 {bad}")
if bad: sys.exit(0)
