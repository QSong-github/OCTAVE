# -*- coding: utf-8 -*-
"""把一个 cohort 的目标 patch、表达、global 嵌入、neighbor 嵌入统一对齐到同一组条码（四者的交集，保持目标 patch 的顺序）。
邻居块由 HEST 的 dump_patches 按组织掩膜切出，个别 spot 会被丢掉（作者的 match_to_target 同样是取交集）；
目标 patch 又经过 MAXSPOT=4000 下采样。此脚本在抽完特征后做一次对齐，幂等（attrs['aligned']）。"""
import sys, os, glob, h5py, numpy as np, anndata as ad
d = f"/path/to/systema4ST/data/triplex/{sys.argv[1]}"; m = sys.argv[2]
def bc_of(h, k="barcode"): return np.asarray(h[k][:]).flatten().astype(str)
for p in sorted(glob.glob(f"{d}/patches/*.h5")):
    sid = os.path.basename(p)[:-3]
    ge, ne = f"{d}/emb/global/{m}/{sid}.h5", f"{d}/emb/neighbor/{m}/{sid}.h5"
    if not (os.path.exists(ge) and os.path.exists(ne)): print(f"  {sid}: 缺嵌入，跳过", flush=True); continue
    with h5py.File(p, "r") as h: bt = bc_of(h)
    with h5py.File(ge, "r") as h: bg = bc_of(h, "barcodes")
    with h5py.File(ne, "r") as h:
        if h.attrs.get("aligned", 0) == 1: continue
        bn = bc_of(h, "barcodes")
    keep = [b for b in bt if b in set(bg) and b in set(bn)]
    assert len(keep) >= 0.5 * len(bt), f"{sid}: 交集 {len(keep)}/{len(bt)} 过小"   # 邻居块按组织掩膜丢点，个别片可达 14%；作者的 match_to_target 同样只取交集
    if len(keep) < len(bt): print(f"  {sid}: 目标 {len(bt)} → 交集 {len(keep)}（邻居块按组织掩膜丢掉 {len(bt)-len(keep)} 个）", flush=True)
    def refilter(path, key, order):
        with h5py.File(path, "r+") as h:
            cur = bc_of(h, key); pos = {b: i for i, b in enumerate(cur)}; idx = np.array([pos[b] for b in order])
            if len(cur) == len(order) and (cur == np.array(order)).all(): h.attrs["aligned"] = 1; return
            data = {k: h[k][:][idx] for k in list(h.keys())}
            for k in list(h.keys()): del h[k]
            for k, v in data.items(): h.create_dataset(k, data=v)
            h.attrs["aligned"] = 1
    refilter(ge, "barcodes", keep); refilter(ne, "barcodes", keep)
    if len(keep) < len(bt):
        with h5py.File(p, "r+") as h:
            pos = {b: i for i, b in enumerate(bt)}; idx = np.array([pos[b] for b in keep])
            data = {k: h[k][:][idx] for k in ("img", "coords", "barcode") if k in h}; attrs = {k: dict(h[k].attrs) for k in data}
            for k in data: del h[k]
            for k, v in data.items():
                ds = h.create_dataset(k, data=v)
                for ak, av in attrs[k].items(): ds.attrs[ak] = av
        a = ad.read_h5ad(f"{d}/adata/{sid}.h5ad"); pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}
        a[np.array([pos[b] for b in keep])].copy().write_h5ad(f"{d}/adata/{sid}.h5ad")
print(f"{sys.argv[1]}/{m}: 对齐完成")
