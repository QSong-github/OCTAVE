#!/usr/bin/env python
"""THItoGene（Brief Bioinform 2024）在 HEST-benchmark 上的运行 —— 模型原样用作者代码，只换数据源。
由 src/hggep_hest.py 生成（mk_thitogene.py），替换点见生成器注释。"""
import os, sys, glob, json, argparse, time
import numpy as np, h5py, torch
import torch.nn.functional as F

H2 = "/path/to/project/methods/THItoGene"
# shims 必须在 H2 之前：Hist2ST/transformer.py 有一行遗留的 `from easydl import *`，
# 但该文件不使用 easydl 的任何符号。空垫片避免为一行无用导入而污染 hest 环境。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shims"))
sys.path.insert(1, H2)
sys.path.insert(0, "/path/to/he2st/HEST/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_model import THItoGene                              # 作者模型，原样
from graph_construction import calcADJ                   # 作者的图构造，原样
from hest.bench.st_dataset import load_adata
import evaluate as E
import anndata as ad

B = "/path/to/he2st/HEST/eval/bench_data"
MAXSPOT = 4000        # 与 histogene_hest.py 一致；calcADJ 是 n×n 稠密阵，4000² = 64MB


def load_slide(sid, cohort, genes, rng):
    with h5py.File(os.path.join(B, cohort, "patches", f"{sid}.h5"), "r") as h:
        bk = "barcodes" if "barcodes" in h else "barcode"
        bc = np.asarray(h[bk][:]).flatten().astype(str).tolist()
        imgs = np.asarray(h["img"][:])
    Y = load_adata(os.path.join(B, cohort, "adata", f"{sid}.h5ad"),
                   genes=genes, barcodes=bc, normalize=True).values.astype(np.float32)
    # ZINB 损失要原始 counts 与 size factor（作者 dataset.py: sf = 行计数 / 其中位数）
    ORI = load_adata(os.path.join(B, cohort, "adata", f"{sid}.h5ad"),
                     genes=genes, barcodes=bc, normalize=False).values.astype(np.float32)
    a = ad.read_h5ad(os.path.join(B, cohort, "adata", f"{sid}.h5ad"))
    bidx = {b: i for i, b in enumerate(map(str, a.obs_names))}
    keep = np.array([bidx[b] for b in bc])
    ctr = np.stack([a.obs["array_row"].to_numpy()[keep],
                    a.obs["array_col"].to_numpy()[keep]], 1).astype(np.int64)
    n_total = imgs.shape[0]
    if n_total > MAXSPOT:
        gx = (ctr[:, 0] - ctr[:, 0].min()) // max(1, (np.ptp(ctr[:, 0]) + 1) // 40)
        gy = (ctr[:, 1] - ctr[:, 1].min()) // max(1, (np.ptp(ctr[:, 1]) + 1) // 40)
        key = gx * 1000 + gy
        sel = []
        for k in np.unique(key):
            idx = np.where(key == k)[0]
            take = max(1, int(round(MAXSPOT * len(idx) / n_total)))
            sel.append(rng.choice(idx, min(take, len(idx)), replace=False))
        sel = np.concatenate(sel)[:MAXSPOT]
        imgs, Y, ctr, ORI = imgs[sel], Y[sel], ctr[sel], ORI[sel]
    x = torch.from_numpy(imgs[..., :3]).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=112, mode="bilinear", align_corners=False)   # 224 → 112
    # 作者 dataset.py 的默认是 neighs=4, prune='Grid'（非 calcADJ 的 k=8 默认值）——
    # 用发表配置，否则图的连通性与作者的不同，比较的就不是同一个方法。
    adj = calcADJ(ctr.astype(np.float32), k=4, pruneTag="NA")   # (n,n) 稠密
    # Grid 裁边会留下度为 0 的点（PRAD 的 2/15 片各 1 个），而 gcn.py 做 adj.div(num_neigh)
    # ⇒ 0/0 = NaN 污染整个前向。最小修复：把孤立点连到其最近邻，保留 Grid 裁边本身。
    deg = adj.sum(1)
    iso = (deg == 0).nonzero().flatten()
    if len(iso):
        from scipy.spatial import cKDTree as _KD
        _, nn_i = _KD(ctr.astype(np.float64)).query(ctr[iso.numpy()].astype(np.float64), k=2)
        for a_, b_ in zip(iso.tolist(), nn_i[:, 1].tolist()):
            adj[a_, b_] = 1.0; adj[b_, a_] = 1.0
        print(f"    [{sid}] Grid 裁边留下 {len(iso)} 个孤立点，已连最近邻", flush=True)
    nc = ORI.sum(1)
    sfs = nc / max(float(np.median(nc)), 1e-9)
    return (x, torch.from_numpy(ctr), torch.from_numpy(Y), adj,
            torch.from_numpy(ORI), torch.from_numpy(sfs.astype(np.float32)),
            imgs.shape[0], n_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--match_steps", type=int, default=3200,
                    help="按该步数反推 epoch，消除「队列片数少 ⇒ 步数少」这一协议混杂")
    ap.add_argument("--fold", type=int, default=-1,
                    help="只跑该折；-1 为全部。数组作业逐折并行时用，"
                         "每任务写自己的 --out，避免并发写同一文件互相覆盖")
    ap.add_argument("--cohorts", default="SKCM,HCC,LUNG,PAAD,COAD,READ,IDC,LYMPH_IDC,PRAD,CCRCC")
    ap.add_argument("--out", default="results/thitogene_hest.json")
    ap.add_argument("--seed", type=int, default=-1, help="≥0 时固定 random/numpy/torch 种子；-1 为原始未播种运行")
    a_ = ap.parse_args()
    if a_.seed >= 0:
        import random as _r; _r.seed(a_.seed); np.random.seed(a_.seed)
        torch.manual_seed(a_.seed); torch.cuda.manual_seed_all(a_.seed)
        print(f'seed={a_.seed}', flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    print(f"device={dev}", flush=True)

    res = {}
    if os.path.exists(a_.out):
        res = {k: {"cohort": v["cohort"], "folds": v["folds"]}
               for k, v in json.load(open(a_.out)).items()}
        print(f"续跑: 已有 {len(res)} 个样本", flush=True)

    for c in a_.cohorts.split(","):
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        for kf in range(len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))):
            if a_.fold >= 0 and kf != a_.fold:
                continue
            rd = lambda f: [l.split(",")[0] for l in open(os.path.join(B, c, "splits", f)
                            ).read().splitlines()[1:] if l.strip()]
            tr, te = rd(f"train_{kf}.csv"), rd(f"test_{kf}.csv")
            if all(s in res for s in te):
                print(f"[{c} fold{kf}] 已完成, 跳过", flush=True); continue
            data = {s: load_slide(s, c, genes, rng) for s in tr + te}
            npos = int(max(d[1].max().item() for d in data.values())) + 1
            if not np.isfinite(np.concatenate([d[2].numpy().ravel() for d in data.values()])).all():
                print("  ⚠ 表达含非有限值", flush=True)
            n_ep = a_.epochs if a_.match_steps <= 0 else max(1, round(a_.match_steps / max(len(tr), 1)))
            print(f"\n[{c} fold{kf}] train={len(tr)} test={len(te)} n_pos={npos} "
                  f"epochs={n_ep} 步数={len(tr)*n_ep}", flush=True)
            model = THItoGene(n_genes=len(genes), n_pos=npos, learning_rate=a_.lr,
                              route_dim=64, caps=20, heads=[16, 8], n_layers=4).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=a_.lr)
            # 作者 configure_optimizers 的调度，原样
            sch = None   # THItoGene 的 configure_optimizers 只有 Adam，无调度
            model.train()
            t0 = time.time()
            for ep in range(n_ep):
                tot = 0.0
                for s in tr:
                    p, ct, y, adj, ori, sf, _, _ = data[s]
                    # adj 不带批维：作者 training_step 里有 `adj=adj.squeeze(0)`，
                    # forward 内部的 GNN 用 mask.mm(x)，要求 2D。
                    pt, ctt, adt = p[None].to(dev), ct[None].to(dev), adj.to(dev)
                    pred = model(pt, ctt, adt)
                    yt = y[None].to(dev)
                    loss = F.mse_loss(pred.view_as(yt), yt)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item()
                if sch is not None: sch.step()
                if ep < 3 or ep % max(1, n_ep // 20) == 0 or ep == n_ep - 1:
                    el = time.time() - t0
                    rate = (ep + 1) / max(el, 1e-9)
                    print(f"    ep{ep:4d} loss={tot/max(len(tr),1):.4f} 用时{el:.0f}s "
                          f"({rate:.2f} ep/s, 预计全折 {n_ep/max(rate,1e-9)/60:.0f}min)", flush=True)
            model.eval()
            with torch.no_grad():
                for s in te:
                    p, ct, y, adj, _, _, _, _ = data[s]
                    pred = model(p[None].to(dev), ct[None].to(dev),
                                 adj.to(dev)).squeeze(0).cpu().numpy()
                    v = float(E.per_gene_pcc(pred, y.numpy()).mean())
                    res.setdefault(s, {"cohort": c, "folds": []})["folds"].append(v)
                    print(f"    test={s} PCC={v:.4f}", flush=True)
                    os.makedirs("results", exist_ok=True)
                    json.dump({k2: {"cohort": v2["cohort"], "folds": v2["folds"],
                                    "pcc": float(np.mean(v2["folds"]))} for k2, v2 in res.items()},
                              open(a_.out, "w"), indent=2, ensure_ascii=False)
            del data, model
            torch.cuda.empty_cache()

    print(f"\n=== THItoGene: {len(res)} 样本 ===")
    for c in sorted({v["cohort"] for v in res.values()}):
        v = [float(np.mean(d["folds"])) for d in res.values() if d["cohort"] == c]
        print(f"  {c:10s} n={len(v):3d} PCC={np.mean(v):.4f}")
    print(f"已存 {a_.out}")


if __name__ == "__main__":
    main()
