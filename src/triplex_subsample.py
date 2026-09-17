# -*- coding: utf-8 -*-
"""TRIPLEX 与其它六个已发表方法同协议：每片空间分层下采样至 MAXSPOT=4000（40×40 网格按比例取，见 src/hist2st_hest.py）。
必要性：GlobalEncoder 对整片做自注意力，非 flash 分支显式构造 n×n 矩阵；TENX99 有 20761 个 spot（n²×16 heads×4B ≈ 27 GB）在反向时必然 OOM。
作者原始数据每片 300–700 spot。此处就地过滤 patches/{sid}.h5（img/coords/barcode）；邻居块 h5 体积巨大不改，改为在抽完特征后按条码过滤嵌入行。
种子按样本固定（default_rng(abs(hash)) 不稳定，改用 sid 的 sha1），可复现。"""
import sys, os, glob, json, hashlib, h5py, numpy as np
MAXSPOT = 4000
d = f"/path/to/systema4ST/data/triplex/{sys.argv[1]}"; sel_all = {}
for p in sorted(glob.glob(f"{d}/patches/*.h5")):
    sid = os.path.basename(p)[:-3]
    with h5py.File(p, "r+") as h:
        if h.attrs.get("subsampled", 0) == 1: sel_all[sid] = "已处理"; continue
        n = h["img"].shape[0]
        if n <= MAXSPOT:
            h.attrs["subsampled"] = 1; sel_all[sid] = list(range(n)); print(f"  {sid}: n={n} ≤ {MAXSPOT}，全保留", flush=True); continue
        ctr = np.asarray(h["coords"][:], np.int64)
        rng = np.random.default_rng(int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16))
        gx = (ctr[:, 0] - ctr[:, 0].min()) // max(1, (np.ptp(ctr[:, 0]) + 1) // 40)
        gy = (ctr[:, 1] - ctr[:, 1].min()) // max(1, (np.ptp(ctr[:, 1]) + 1) // 40)
        key = gx * 1000 + gy; sel = []
        for k in np.unique(key):
            idx = np.where(key == k)[0]; take = max(1, int(round(MAXSPOT * len(idx) / n)))
            sel.append(rng.choice(idx, min(take, len(idx)), replace=False))
        sel = np.sort(np.concatenate(sel)[:MAXSPOT]); sel_all[sid] = sel.tolist()
        data = {k: h[k][:][sel] for k in ("img", "coords", "barcode") if k in h}
        attrs = {k: dict(h[k].attrs) for k in data}
        for k in data: del h[k]
        for k, v in data.items():
            ds = h.create_dataset(k, data=v)
            for ak, av in attrs[k].items(): ds.attrs[ak] = av
        h.attrs["subsampled"] = 1
        print(f"  {sid}: n={n} → {len(sel)}", flush=True)
json.dump({k: (v if isinstance(v, list) else v) for k, v in sel_all.items()}, open(f"{d}/subsample_idx.json", "w"))
print(f"{sys.argv[1]}: 完成（MAXSPOT={MAXSPOT}）")
