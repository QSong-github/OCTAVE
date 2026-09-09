# -*- coding: utf-8 -*-
"""探查: ① 全转录组在哪 ② CosMx/MERFISH 是否配对 H&E ③ 可用规模。"""
import os, glob, numpy as np, anndata as ad, h5py, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from effres import SLIDES, SEB

def desc(p, tag):
    print(f"\n--- {tag}\n    {p}")
    try:
        with h5py.File(p, "r") as h:                       # 只读元数据, 不载入矩阵
            n_obs = len(h["obs"][list(h["obs"].attrs.get("column-order", ["_index"]))[0]]) \
                if "obs" in h else -1
            print(f"    顶层键: {list(h.keys())}")
            for k in ("obs", "var", "obsm", "uns", "layers"):
                if k in h: print(f"    {k}: {list(h[k].keys())[:22]}")
    except Exception as e:
        print(f"    [h5py 失败 {e}]")

print("="*72); print("① 我们一直在用的预处理文件"); print("="*72)
a = ad.read_h5ad(SEB.H5AD, backed="r")
print(f"  {SEB.H5AD}\n  n_obs={a.n_obs} n_vars={a.n_vars}  var 列={list(a.var.columns)}")

print("\n"+"="*72); print("② 原始 16µm 全转录组"); print("="*72)
for s in SLIDES:
    p = f"/path/to/spatial2exp/he2st_align/st_bench/data/{s}/adata_16um.h5ad"
    b = ad.read_h5ad(p, backed="r")
    print(f"  [{s[-2:]}] n_obs={b.n_obs} n_vars={b.n_vars} obs={list(b.obs.columns)[:8]} obsm={list(b.obsm.keys())}")

print("\n"+"="*72); print("③ CosMx / MERFISH —— 是否配对 H&E"); print("="*72)
for p in ["/path/to/spatial2exp/stbench/cosmx_hs_nsclc.h5ad"]:
    desc(p, "CosMx NSCLC")
    c = ad.read_h5ad(p, backed="r")
    print(f"    n_obs={c.n_obs} n_vars={c.n_vars}")
    print(f"    obs 列: {list(c.obs.columns)}")
    print(f"    obsm: {list(c.obsm.keys())}  uns: {list(c.uns.keys())[:12]}")
    for col in list(c.obs.columns)[:12]:
        v = c.obs[col]
        try: print(f"      {col:24s} nuniq={v.nunique():6d} 例={list(map(str, v.unique()[:3]))}")
        except Exception: pass
for p in sorted(glob.glob("/path/to/spatial2exp/**/MERFISH*16um*.h5ad", recursive=True))[:2]:
    desc(p, "MERFISH 16µm")
print("\n"+"="*72); print("④ 该目录下所有 H&E / 图像文件"); print("="*72)
n = 0
for root in ["/path/to/spatial2exp"]:
    for d, _, fs in os.walk(root):
        if any(x in d for x in ("miniconda", "site-packages", ".git")): continue
        for f in fs:
            if f.lower().endswith((".tif", ".tiff", ".svs", ".ndpi", ".btf")):
                fp = os.path.join(d, f)
                print(f"  {os.path.getsize(fp)/1e9:7.2f} GB  {fp}"); n += 1
                if n > 30: break
        if n > 30: break
print(f"  共 {n} 个（截断于 30）")
