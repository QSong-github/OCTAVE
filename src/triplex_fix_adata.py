# -*- coding: utf-8 -*-
"""TRIPLEX 的 prepare_data 用 HEST 原始 st/{id}.h5ad 重建表达，并按 index 与目标 patch 的条码取交集；
对部分样本（如 TENX111）两边条码体系不同，交集为空，训练集长度为 0。基准任务本身带 adata（bench_data/{C}/adata/{id}.h5ad，
即本项目其它六个方法用的同一份表达），其条码与 patch 完全一致。此脚本用基准 adata 覆盖 TRIPLEX 副本，并按 patch 条码顺序对齐。"""
import sys, os, glob, h5py, numpy as np, anndata as ad
B = "/path/to/he2st/HEST/eval/bench_data"; C = sys.argv[1]; d = f"/path/to/systema4ST/data/triplex/{C}"
for p in sorted(glob.glob(f"{d}/patches/*.h5")):
    sid = os.path.basename(p)[:-3]
    with h5py.File(p, "r") as h:
        k = "barcode" if "barcode" in h else "barcodes"; bc = np.asarray(h[k][:]).flatten().astype(str)
    a = ad.read_h5ad(f"{B}/{C}/adata/{sid}.h5ad"); pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}
    miss = [b for b in bc if b not in pos]; assert not miss, f"{sid}: {len(miss)} 个 patch 条码不在基准 adata 中"
    sub = a[np.array([pos[b] for b in bc])].copy()
    for c in ("array_row", "array_col"):
        assert c in sub.obs.columns, f"{sid}: 缺 {c}"
    sub.write_h5ad(f"{d}/adata/{sid}.h5ad")
    print(f"  {sid}: {sub.n_obs} spots × {sub.n_vars} genes（patch {len(bc)}）", flush=True)
print(f"{C}: 完成")
