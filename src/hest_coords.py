# -*- coding: utf-8 -*-
"""
坐标与规模核验 —— 跨 72 样本实验的前置条件。

这个项目已经在坐标上摔过两次(prep_bin 的 NaN→0 质心污染、x_um 的同源污染),
所以在跑广度实验前先把每个样本的坐标单位钉死, 而不是假设。

要回答:
  1. obsm['spatial'] 是否就等于 obs 的 pxl_*_in_fullres? 列序是 (x,y) 还是 (row,col)?
  2. 每样本 µm/px —— 从 HEST 元数据的 pixel_size_um_estimated 按 ID 关联
  3. 换算成 µm 后, 最近邻间距是否落在该平台应有的 spot 间距上
     (Visium 100µm / legacy ST 200µm / Xenium 分箱后视具体)
  4. spot 数与评测基因可得性 —— 定出该排除哪些样本
只读。
"""
import os, glob, json, csv, numpy as np, anndata as ad
from scipy.spatial import cKDTree

B = "/path/to/he2st/HEST/eval/bench_data"
META = "/path/to/he2st/HEST/assets/HEST_v1_1_0.csv"

meta = {r["id"]: r for r in csv.DictReader(open(META, encoding="utf-8-sig"))}
print(f"元数据 {len(meta)} 条")

rows = []
for c in sorted(os.listdir(B)):
    d = os.path.join(B, c, "adata")
    if not os.path.isdir(d):
        continue
    genes50 = set(json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"])
    for p in sorted(glob.glob(os.path.join(d, "*.h5ad"))):
        sid = os.path.basename(p)[:-5]
        a = ad.read_h5ad(p)
        sp = np.asarray(a.obsm["spatial"], np.float64)
        ob = a.obs
        # 1) spatial 与 pxl_* 的关系
        rel = "?"
        if {"pxl_col_in_fullres", "pxl_row_in_fullres"} <= set(ob.columns):
            pc = ob["pxl_col_in_fullres"].to_numpy(np.float64)
            pr = ob["pxl_row_in_fullres"].to_numpy(np.float64)
            for nm, cand in (("(col,row)", np.stack([pc, pr], 1)),
                             ("(row,col)", np.stack([pr, pc], 1))):
                if np.allclose(sp, cand, rtol=0, atol=1e-6):
                    rel = "==pxl" + nm; break
            else:
                s = np.median(cand[:, 0] / np.where(sp[:, 0] == 0, np.nan, sp[:, 0]))
                rel = f"≠pxl(比例≈{s:.3f})"
        m = meta.get(sid)
        umpx = float(m["pixel_size_um_estimated"]) if m and m.get("pixel_size_um_estimated") else np.nan
        tech = (m or {}).get("st_technology", "?")
        d2, _ = cKDTree(sp).query(sp, k=2)
        nn_px = float(np.median(d2[:, 1]))
        nn_um = nn_px * umpx if umpx == umpx else np.nan
        n50 = len(genes50 & set(map(str, a.var_names)))
        rows.append(dict(cohort=c, sid=sid, n=a.n_obs, ngene=a.n_vars, tech=tech,
                         umpx=umpx, nn_px=nn_px, nn_um=nn_um, rel=rel, n50=n50))

print(f"\n{'队列':10s}{'样本':10s}{'spots':>7s}{'基因':>7s}{'技术':>12s}"
      f"{'µm/px':>8s}{'NN(px)':>8s}{'NN(µm)':>8s}{'50基因':>7s}  spatial关系")
for r in rows:
    print(f"{r['cohort']:10s}{r['sid']:10s}{r['n']:>7d}{r['ngene']:>7d}{str(r['tech'])[:11]:>12s}"
          f"{r['umpx']:>8.3f}{r['nn_px']:>8.1f}{r['nn_um']:>8.1f}{r['n50']:>7d}  {r['rel']}")

nn = np.array([r["nn_um"] for r in rows], float)
ns = np.array([r["n"] for r in rows])
print(f"\n=== 汇总 ===")
print(f"样本 {len(rows)} 个; spot 数 中位={np.median(ns):.0f} 最小={ns.min()} 最大={ns.max()}")
print(f"NN(µm) 中位={np.nanmedian(nn):.1f} 范围=[{np.nanmin(nn):.1f}, {np.nanmax(nn):.1f}]")
print(f"spatial==pxl 的样本: {sum(1 for r in rows if r['rel'].startswith('==pxl'))}/{len(rows)}")
print(f"50 评测基因全部可得: {sum(1 for r in rows if r['n50']==50)}/{len(rows)}")
print(f"spot<1000 的样本: {[r['sid'] for r in rows if r['n']<1000]}")
json.dump(rows, open("/path/to/systema4ST/results/hest_samples.json", "w"),
          indent=2, ensure_ascii=False)
print("\n已存 results/hest_samples.json")
