# -*- coding: utf-8 -*-
"""邻居嵌入按目标 patch 的条码顺序过滤（目标 patch 已下采样，邻居块 h5 未改）。幂等：attrs['filtered_to_target']。"""
import sys, os, glob, h5py, numpy as np
d = f"/path/to/project/data/triplex/{sys.argv[1]}"; m = sys.argv[2]
for p in sorted(glob.glob(f"{d}/patches/*.h5")):
    sid = os.path.basename(p)[:-3]; e = f"{d}/emb/neighbor/{m}/{sid}.h5"
    if not os.path.exists(e): continue
    with h5py.File(p, "r") as h: bt = np.asarray(h["barcode"][:]).flatten().astype(str)
    with h5py.File(e, "r+") as h:
        if h.attrs.get("filtered_to_target", 0) == 1: continue
        bn = np.asarray(h["barcodes"][:]).flatten().astype(str)
        if len(bn) == len(bt) and (bn == bt).all(): h.attrs["filtered_to_target"] = 1; continue
        pos = {b: i for i, b in enumerate(bn)}; miss = [b for b in bt if b not in pos]
        assert not miss, f"{sid}: {len(miss)} 个目标条码不在邻居嵌入中"
        idx = np.array([pos[b] for b in bt])
        data = {k: h[k][:][idx] for k in h.keys()}
        for k in list(h.keys()): del h[k]
        for k, v in data.items(): h.create_dataset(k, data=v)
        h.attrs["filtered_to_target"] = 1
        print(f"  {sid}: 邻居嵌入 {len(bn)} → {len(bt)}", flush=True)
print(f"{sys.argv[1]}/{m}: 邻居嵌入对齐完成")
