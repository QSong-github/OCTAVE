# -*- coding: utf-8 -*-
"""DeepSpot 在 HEST 基准上的特征：作者协议（medRxiv 2025.02.09.25321567 §4.2.1）——每个 spot 一个 tile、3×3 非重叠子 tile、
邻居 = 阵列坐标半径 1 内的 spot（训练时直接取邻居 spot 的 tile 特征）。tile 用 HEST 预切的 224×224（与其它六个方法同数据源）；
子 tile 为 224 的 3×3 网格（74×74），送编码器前重采样到 224（作者的 get_morphology_model_and_preprocess 亦把每个 tile resize 到 224）。
编码器加载与前向复用 hest_embed_v2 / xen_embed_all.embed_batch（与 57 编码器扫描同一份权重与预处理）。
输出 results/deepspot_emb/{enc}/{cohort}/{sid}.npz: spot (n,d), sub (n,9,d), bc (n,), xy_array (n,2)。"""
import os, sys, glob, argparse, numpy as np, torch, h5py, anndata as ad
sys.path.insert(0, "/path/to/systema4ST"); sys.path.insert(0, "/path/to/systema4ST/src")
import hest_embed_v2 as HE
from xen_embed_all import embed_batch
B = "/path/to/he2st/HEST/eval/bench_data"
ap = argparse.ArgumentParser(); ap.add_argument("--cohort", required=True); ap.add_argument("--encoder", required=True)
ap.add_argument("--out", default="/path/to/systema4ST/results/deepspot_emb"); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--sub_batch", type=int, default=256); ap.add_argument("--grid", type=int, default=3)
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
model, kind = HE.encoder(a.encoder, dev)
od = f"{a.out}/{a.encoder}/{a.cohort}"; os.makedirs(od, exist_ok=True)
g = a.grid; cell = 224 // g
def feats(arr):
    out = []
    for i in range(0, len(arr), a.sub_batch): out.append(embed_batch(model, kind, arr[i:i + a.sub_batch], dev).cpu().numpy().astype(np.float32))
    return np.concatenate(out)
for f in sorted(glob.glob(f"{B}/{a.cohort}/patches/*.h5")):
    sid = os.path.basename(f)[:-3]; o = f"{od}/{sid}.npz"
    if os.path.exists(o): print(f"[{a.encoder}/{a.cohort}/{sid}] 已存在", flush=True); continue
    with h5py.File(f, "r") as h:
        bk = "barcodes" if "barcodes" in h else "barcode"
        bc = np.asarray(h[bk][:]).flatten().astype(str); imgs = np.asarray(h["img"][:])[..., :3]
    obs = ad.read_h5ad(f"{B}/{a.cohort}/adata/{sid}.h5ad", backed="r").obs
    pos = {b: i for i, b in enumerate(obs.index.astype(str))}; keep = np.array([pos[b] for b in bc])
    xy = np.stack([obs["array_row"].to_numpy()[keep], obs["array_col"].to_numpy()[keep]], 1).astype(np.int64)
    spot, sub = [], []
    with torch.inference_mode():
        for i in range(0, len(imgs), a.batch):
            arr = imgs[i:i + a.batch]
            spot.append(feats(arr))
            tiles = np.stack([arr[:, r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] for r in range(g) for c in range(g)], 1)   # (b,9,cell,cell,3)
            sub.append(feats(np.ascontiguousarray(tiles.reshape(-1, cell, cell, 3))).reshape(len(arr), g * g, -1))
    spot = np.concatenate(spot); sub = np.concatenate(sub)
    np.savez(o, spot=spot, sub=sub, bc=bc, xy_array=xy)
    print(f"[{a.encoder}/{a.cohort}/{sid}] n={len(bc)} spot={spot.shape} sub={sub.shape}", flush=True)
