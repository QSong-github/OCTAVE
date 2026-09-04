# -*- coding: utf-8 -*-
"""
专有 H&E→表达 预测方法的基线对比。
全部跑在**完全相同**的协议下(同图像塔、跨切片留一、top-50 HVG、per-gene PCC),
数字才可直接与我们的"冻结 ST 编码器 + 对齐 + 检索"比较。

基线(按其方法核心复现,非原仓库):
  ridge      : HEST-benchmark 协议 —— 图像 embedding → 标准化 → Ridge 回归 → 表达。
               扫 alpha 取最优(偏向基线的 oracle 调参,如实标注)。
  mlp_reg    : 图像 embedding → MLP → 表达。**通用基线,不对应任何具体方法**
               (不可称为 ST-Net/iStar: ST-Net 端到端微调 CNN、iStar 用 HIPT 层次特征+超分目标,
                二者的核心机制此处都没有。原版见 native_baselines.py)。
  bleep      : BLEEP-style(**冻结图像塔变体**,非原版:原版微调 ResNet50)——
               图像头 + **可训练表达编码器**,InfoNCE 对齐,再检索借表达。
               它与我们的关键差别: 表达表示是**在本数据上从头学**的, 而我们用**预训练冻结 ST 编码器**。
  knn_image  : 纯图像 kNN 检索(不用任何表达表示)—— 下界。
  ours       : 我们的方法(冻结 ST 编码器 + 线性头 InfoNCE+JEPA + 检索)。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E, retrieval as R
from align import build_aligner
from st_encoder_bench import load_model_emb, H5AD, COLLAB, EMBDIR, SLIDES


def ridge_predict(Xtr, Ytr, Xte, alpha):
    from sklearn.linear_model import Ridge
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    m = Ridge(alpha=alpha).fit((Xtr - mu) / sd, Ytr)
    return m.predict((Xte - mu) / sd).astype(np.float32)


def mlp_regress(Xtr, Ytr, Xte, hidden=512, epochs=40, lr=1e-3, batch=512, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(xt.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, Ytr.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = xt.shape[0]; bs = min(batch, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=dev))
    return out.cpu().numpy().astype(np.float32)


def bleep(Xtr_img, Etr_expr, Xte_img, ref_expr, k=50, proj=256, hidden=512,
          epochs=40, lr=1e-3, batch=512, temp=0.07, seed=0):
    """BLEEP: 图像头 + 可训练表达编码器, InfoNCE, 再检索。"""
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    imu, isd = Xtr_img.mean(0), Xtr_img.std(0) + 1e-8
    emu, esd = Etr_expr.mean(0), Etr_expr.std(0) + 1e-8
    I = torch.tensor((Xtr_img - imu) / isd, dtype=torch.float32, device=dev)
    S = torch.tensor((Etr_expr - emu) / esd, dtype=torch.float32, device=dev)
    fi = nn.Linear(I.shape[1], proj).to(dev)                       # 图像头(冻结特征之上)
    fs = nn.Sequential(nn.Linear(S.shape[1], hidden), nn.GELU(),
                       nn.Linear(hidden, proj)).to(dev)            # 可训练表达编码器
    opt = torch.optim.AdamW(list(fi.parameters()) + list(fs.parameters()), lr=lr, weight_decay=1e-4)
    n = I.shape[0]; bs = min(batch, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            zi = F.normalize(fi(I[idx]), dim=1); zs = F.normalize(fs(S[idx]), dim=1)
            logits = zi @ zs.T / temp
            lab = torch.arange(idx.shape[0], device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad(); loss.backward(); opt.step()
    fi.eval(); fs.eval()
    with torch.no_grad():
        q = F.normalize(fi(torch.tensor((Xte_img - imu) / isd, dtype=torch.float32, device=dev)), dim=1).cpu().numpy()
        r = F.normalize(fs(S), dim=1).cpu().numpy()
    return R.retrieve_cross_modal(q.astype(np.float32), r.astype(np.float32), ref_expr, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="hibou_l")
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--jepa", type=float, default=4.0)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    a = ad.read_h5ad(H5AD)
    expr = np.nan_to_num(np.asarray(a.X, np.float32))
    slide = a.obs["slide_id"].astype(str).values
    n = a.n_obs
    nps = [int((slide == s).sum()) for s in SLIDES]
    img = np.nan_to_num(np.concatenate([
        np.load(os.path.join(EMBDIR, f"emb_{args.tower}_P2.npy")),
        np.load(os.path.join(EMBDIR, f"emb_{args.tower}_P5.npy"))]).astype(np.float32))
    collab = np.nan_to_num(np.asarray(ad.read_h5ad(COLLAB).obsm["img_emb"], np.float32))
    print(f"塔={args.tower}({img.shape[1]}d)  N={n}  k={args.k}", flush=True)

    ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    acc = {}

    def add(name, v):
        acc.setdefault(name, []).append(v)

    for g in sorted(set(slide.tolist())):
        te = (slide == g); tr = ~te
        gidx = E.topk_hvg(expr[tr], args.hvg)
        Ytr = expr[tr]
        print(f"[fold {g}] train={tr.sum()} test={te.sum()}", flush=True)

        # 1) Ridge (HEST 协议)
        for al in ALPHAS:
            p = ridge_predict(img[tr], Ytr, img[te], al)
            add(f"ridge(alpha={al:g})", float(E.per_gene_pcc(p, expr[te], gidx).mean()))
        # 2) MLP 回归
        p = mlp_regress(img[tr], Ytr, img[te])
        add("mlp_regression", float(E.per_gene_pcc(p, expr[te], gidx).mean()))
        # 3) BLEEP
        p = bleep(img[tr], Ytr, img[te], Ytr, k=args.k)
        add("BLEEP(可训练表达编码器)", float(E.per_gene_pcc(p, expr[te], gidx).mean()))
        # 4) 纯图像 kNN
        p = R.image_floor(img[te], img[tr], Ytr, k=args.k)
        add("kNN图像检索(下界)", float(E.per_gene_pcc(p, expr[te], gidx).mean()))
        # 5) 我们的方法
        alg = build_aligner("mlp", hidden=0, jepa_weight=args.jepa, temp=args.temp, epochs=40).fit(img[tr], collab[tr])
        p = R.retrieve_cross_modal(alg.project_img(img[te]), alg.project_st(collab[tr]), Ytr, k=args.k)
        add("★ 我们(冻结ST编码器+对齐+检索)", float(E.per_gene_pcc(p, expr[te], gidx).mean()))

    rows = sorted(((k, float(np.mean(v)), float(np.std(v))) for k, v in acc.items()),
                  key=lambda r: -r[1])
    print(f"\n=== 基线对比 (塔={args.tower}, 跨切片留一, top-{args.hvg} HVG, k={args.k}) ===")
    print(f"{'方法':34s} {'per-gene PCC':>13s} {'std':>8s}")
    for k, m, s in rows:
        print(f"{k:34s} {m:>13.4f} {s:>8.4f}")
    os.makedirs(args.out, exist_ok=True)
    op = os.path.join(args.out, f"baselines_{args.tower}_k{args.k}_hvg{args.hvg}.json")
    json.dump([{"method": k, "pcc": m, "std": s} for k, m, s in rows],
              open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {op}")


if __name__ == "__main__":
    main()
