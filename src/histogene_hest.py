# -*- coding: utf-8 -*-
"""
HisToGene 在 HEST-benchmark 上的运行 —— 模型原样用作者代码, 只换数据源。

作者接口(vis_model.py:144-170):
    forward(patches, centers): patches (B,N,3*112*112) 展平; centers (B,N,2) long
                               → Embedding(n_pos, dim); 返回 (B,N,n_genes)
    training_step: loss = F.mse_loss(pred.view_as(exp), exp);  Adam(lr=self.learning_rate)
一个"样本" = 一整张切片(整片进 Transformer), 不是按 spot 采样。

三处必须处理的设计约束(来自代码探查, 非我的选择):
 ① n_pos=64 的位置嵌入网格: HEST 的 array_row/col 会超 64 → 按 fold 计算并显式传入,
    否则 Embedding 越界。
 ② 整片输入 + n² 注意力: 作者原始数据 HER2ST 每片 300–700 spot, HEST 是 1084–25080。
    上限取 4000 —— 这不是我拍的数, 是**作者自己 __main__ 冒烟用的规模**
    (torch.rand(1,4000,3*112*112))。超出的切片做空间分层下采样, 并如实报告。
 ③ patch 尺寸: 作者 patch_size=112, HEST 预切的是 224×224(0.5µm/px) → 双线性降采样到 112。
    覆盖的物理视野完全相同, 只是分辨率减半。

评测: HEST 官方病人级划分、官方 50 基因、log1p —— 与本项目其它方法同口径。
"""
import os, sys, glob, json, argparse, numpy as np, h5py, torch
import torch.nn.functional as F

HG = "/path/to/systema4ST/methods/HisToGene"
sys.path.insert(0, HG)
sys.path.insert(0, "/path/to/he2st/HEST/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vis_model import HisToGene                        # 作者模型, 原样
from hest.bench.st_dataset import load_adata
import evaluate as E
import anndata as ad

B = "/path/to/he2st/HEST/eval/bench_data"
MAXSPOT = 4000                                          # 见文件头 ②


def load_slide(sid, cohort, genes, rng):
    """返回 (patches_flat, centers_int, expr, n_kept, n_total)。"""
    with h5py.File(os.path.join(B, cohort, "patches", f"{sid}.h5"), "r") as h:
        bk = "barcodes" if "barcodes" in h else "barcode"
        bc = np.asarray(h[bk][:]).flatten().astype(str).tolist()
        imgs = np.asarray(h["img"][:])                  # (n,224,224,3) uint8
    Y = load_adata(os.path.join(B, cohort, "adata", f"{sid}.h5ad"),
                   genes=genes, barcodes=bc, normalize=True).values.astype(np.float32)
    a = ad.read_h5ad(os.path.join(B, cohort, "adata", f"{sid}.h5ad"))
    bidx = {b: i for i, b in enumerate(map(str, a.obs_names))}
    keep = np.array([bidx[b] for b in bc])
    ctr = np.stack([a.obs["array_row"].to_numpy()[keep],
                    a.obs["array_col"].to_numpy()[keep]], 1).astype(np.int64)
    n_total = imgs.shape[0]
    if n_total > MAXSPOT:                               # 空间分层下采样: 按网格分块均匀取
        gx = (ctr[:, 0] - ctr[:, 0].min()) // max(1, (np.ptp(ctr[:, 0]) + 1) // 40)
        gy = (ctr[:, 1] - ctr[:, 1].min()) // max(1, (np.ptp(ctr[:, 1]) + 1) // 40)
        key = gx * 1000 + gy
        sel = []
        for k in np.unique(key):
            idx = np.where(key == k)[0]
            take = max(1, int(round(MAXSPOT * len(idx) / n_total)))
            sel.append(rng.choice(idx, min(take, len(idx)), replace=False))
        sel = np.concatenate(sel)[:MAXSPOT]
        imgs, Y, ctr = imgs[sel], Y[sel], ctr[sel]
    x = torch.from_numpy(imgs[..., :3]).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=112, mode="bilinear", align_corners=False)   # 224 → 112
    return x.reshape(x.shape[0], -1), torch.from_numpy(ctr), torch.from_numpy(Y), \
        imgs.shape[0], n_total


def main():
    ap = argparse.ArgumentParser()
    # 作者 tutorial.ipynb 的发表配置: HisToGene(n_layers=8, learning_rate=1e-5), max_epochs=100
    # 首轮(38940897)我误用了函数签名默认值(n_layers=4, lr=1e-4), lr 高 10 倍导致欠拟合。
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    # 按发表配置跑过一轮(38945010)后, 结果并未改善 ⇒ "lr 错"这个诊断被证伪。
    # 真正的混杂是梯度步数: HisToGene 一整片=一个样本, 步数=切片数×epoch。
    # 作者 HER2ST 32 片×100ep=3200 步; HEST 队列只有 2–20 片 ⇒ 200–2000 步。
    # --match_steps 按队列调 epoch 使步数统一到 3200, 直接消除这个混杂。
    ap.add_argument("--match_steps", type=int, default=0, help=">0 时按该步数反推 epoch")
    ap.add_argument("--cohorts", default="SKCM,HCC,LUNG,PAAD,COAD,READ,IDC,LYMPH_IDC,PRAD,CCRCC")
    ap.add_argument("--out", default="results/histogene_hest.json")
    a_ = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    print(f"device={dev}", flush=True)

    res = {}
    if os.path.exists(a_.out):                          # 断点续跑
        res = {k: {"cohort": v["cohort"], "folds": v["folds"]}
               for k, v in json.load(open(a_.out)).items()}
        print(f"续跑: 已有 {len(res)} 个样本", flush=True)

    for c in a_.cohorts.split(","):
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        for kf in range(len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))):
            rd = lambda f: [l.split(",")[0] for l in open(os.path.join(B, c, "splits", f)
                            ).read().splitlines()[1:] if l.strip()]
            tr, te = rd(f"train_{kf}.csv"), rd(f"test_{kf}.csv")
            if all(s in res for s in te):
                print(f"[{c} fold{kf}] 已完成, 跳过", flush=True); continue
            data = {s: load_slide(s, c, genes, rng) for s in tr + te}
            npos = int(max(d[1].max().item() for d in data.values())) + 1   # 见文件头 ①
            print(f"\n[{c} fold{kf}] train={len(tr)} test={len(te)} n_pos={npos} "
                  f"spot(采样/原始)=" + ",".join(f"{d[3]}/{d[4]}" for d in data.values()), flush=True)
            model = HisToGene(n_genes=len(genes), n_pos=npos, patch_size=112,
                              n_layers=a_.n_layers, learning_rate=a_.lr).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=a_.lr)
            n_ep = a_.epochs if a_.match_steps <= 0 else max(1, round(a_.match_steps / max(len(tr), 1)))
            print(f"    配置: n_layers={a_.n_layers} lr={a_.lr} epochs={n_ep} "
                  f"→ 梯度步数={len(tr)*n_ep} (作者 HER2ST: 32片×100ep=3200)", flush=True)
            model.train()
            for ep in range(n_ep):
                tot = 0.0
                for s in tr:
                    p, ct, y, _, _ = data[s]
                    pred = model(p[None].to(dev), ct[None].to(dev))
                    loss = F.mse_loss(pred.view_as(y[None].to(dev)), y[None].to(dev))
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item()
                if ep % max(1, n_ep // 5) == 0 or ep == n_ep - 1:
                    print(f"    ep{ep:3d} loss={tot/max(len(tr),1):.4f}", flush=True)
            model.eval()
            with torch.no_grad():
                for s in te:
                    p, ct, y, _, _ = data[s]
                    pred = model(p[None].to(dev), ct[None].to(dev)).squeeze(0).cpu().numpy()
                    v = float(E.per_gene_pcc(pred, y.numpy()).mean())
                    res.setdefault(s, {"cohort": c, "folds": []})["folds"].append(v)
                    print(f"    test={s} PCC={v:.4f}", flush=True)
                    os.makedirs("results", exist_ok=True)
                    json.dump({k2: {"cohort": v2["cohort"], "folds": v2["folds"],
                                    "pcc": float(np.mean(v2["folds"]))} for k2, v2 in res.items()},
                              open(a_.out, "w"), indent=2, ensure_ascii=False)
            del data, model
            torch.cuda.empty_cache()

    print(f"\n=== HisToGene: {len(res)} 样本 ===")
    for c in sorted({v["cohort"] for v in res.values()}):
        v = [float(np.mean(d["folds"])) for d in res.values() if d["cohort"] == c]
        print(f"  {c:10s} n={len(v):3d} PCC={np.mean(v):.4f}")
    print(f"已存 {a_.out}")


if __name__ == "__main__":
    main()
