#!/usr/bin/env python
"""HECLIP 在 HEST-benchmark 上的运行 —— 模型原样用作者代码，只换数据源。

HECLIP 的文件头自称 "developed based on the BLEEP"。逐项核对后，实质差异只有三处，
本文件据此从 src/bleep_hest.py 改成：
  ① 无 spot 投影头。作者 forward 里 spot_embeddings = spot_features，参考侧嵌入
     就是原始表达，所以 projection_dim 必须等于基因数（本协议 50）。漏掉这条
     模型维度就对不上，或退化成无意义的投影。
  ② 只有 image-centric 单向损失，targets 只由 image 相似度构造（BLEEP 是双向、
     targets 取两侧相似度均值）。这一条在作者模型内部，本文件不介入。
  ③ 超参用作者默认：lr=1e-3, wd=1e-3, max_epochs=15, batch=128, temperature=1.0。

推断方式与 BLEEP 相同且已核对：作者 infer.py 的 find_matches 对两侧做 L2 归一化后
取 top-50 点积近邻，再对其真值表达做均匀平均——与 src/bleep_hest.py 的 retrieve() 一致。

协议与 bleep_hest.py 完全相同：同 HEST 官方划分、同 50 基因、同 load_adata。
"""
import os, sys, glob, json, argparse, numpy as np, h5py, torch
from torch.utils.data import Dataset, DataLoader

HEC = "/path/to/systema4ST/methods/HECLIP/code"
sys.path.insert(0, HEC)
sys.path.insert(0, "/path/to/he2st/HEST/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_hvg import HECLIPModel                      # HECLIP 的模型, 原样
from hest.bench.st_dataset import load_adata           # HEST 官方数据加载
import evaluate as E

B = "/path/to/he2st/HEST/eval/bench_data"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class HESTPatchDataset(Dataset):
    """喂给 BLEEP 的 batch 结构与原版一致: {'image': (3,224,224), 'reduced_expression': (G,)}"""
    def __init__(self, sids, cohort, genes):
        self.items = []
        for sid in sids:
            with h5py.File(os.path.join(B, cohort, "patches", f"{sid}.h5"), "r") as h:
                bk = "barcodes" if "barcodes" in h else "barcode"
                bc = np.asarray(h[bk][:]).flatten().astype(str).tolist()
                imgs = np.asarray(h["img"][:])                      # (n,224,224,3) uint8
            Y = load_adata(os.path.join(B, cohort, "adata", f"{sid}.h5ad"),
                           genes=genes, barcodes=bc, normalize=True).values.astype(np.float32)
            assert imgs.shape[0] == Y.shape[0], f"{sid}: {imgs.shape[0]} vs {Y.shape[0]}"
            self.items.append((imgs, Y, sid))
        self.index = [(i, j) for i, (im, _, _) in enumerate(self.items) for j in range(im.shape[0])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, j = self.index[k]
        im, Y, _ = self.items[i]
        x = (im[j][..., :3].astype(np.float32) / 255.0 - MEAN) / STD
        return {"image": torch.from_numpy(x).permute(2, 0, 1).float(),
                "reduced_expression": torch.from_numpy(Y[j])}


@torch.no_grad()
def embed_all(model, ds, dev, bs=256):
    """返回 (图像投影, spot 投影, 真值表达)。"""
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    I, Sp, Yv = [], [], []
    for b in dl:
        img = b["image"].to(dev); expr = b["reduced_expression"].to(dev)
        I.append(model.image_projection(model.image_encoder(img)).cpu())
        # HECLIP 没有 spot 投影头：作者 forward 里 spot_embeddings = spot_features，
        # 即原始表达本身充当参考侧嵌入。因此 projection_dim 必须等于基因数。
        Sp.append(expr.cpu())
        Yv.append(b["reduced_expression"])
    return torch.cat(I), torch.cat(Sp), torch.cat(Yv)


def retrieve(q_img, ref_spot, ref_expr, k=50):
    """BLEEP 的检索式推理: 查询图像嵌入 → 最近 k 个参考 spot 嵌入 → 取其真值表达均值。"""
    q = torch.nn.functional.normalize(q_img, dim=1)
    r = torch.nn.functional.normalize(ref_spot, dim=1)
    out = torch.empty(q.shape[0], ref_expr.shape[1])
    for i in range(0, q.shape[0], 1024):
        sim = q[i:i + 1024] @ r.T
        idx = sim.topk(min(k, r.shape[0]), dim=1).indices
        out[i:i + 1024] = ref_expr[idx].mean(1)
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)      # 作者 max_epochs
    ap.add_argument("--batch_size", type=int, default=128) # 作者 batch_size
    ap.add_argument("--lr", type=float, default=1e-3)      # 作者 lr
    ap.add_argument("--wd", type=float, default=1e-3)      # 作者 wd
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--fold", type=int, default=-1,
                    help="只跑该折；-1 为全部。数组作业逐折并行时用，"
                         "每任务写自己的 --out，避免并发写同一文件互相覆盖")
    ap.add_argument("--cohorts", default="all")
    ap.add_argument("--out", default="results/heclip_hest.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}", flush=True)

    cohorts = sorted([c for c in os.listdir(B) if os.path.isdir(os.path.join(B, c, "adata"))]) \
        if a.cohorts == "all" else a.cohorts.split(",")
    # 每折即时落盘并支持续跑: 48h 墙钟下最忌"跑完才写", 超时即全部丢失
    res = {}
    if os.path.exists(a.out):
        for s_, d_ in json.load(open(a.out)).items():
            res[s_] = {"cohort": d_["cohort"], "folds": d_["folds"]}
        print(f"续跑: 已有 {len(res)} 个样本的结果", flush=True)
    done_folds = {(d["cohort"], s_) for s_, d in res.items()}
    for c in cohorts:
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        nsp = len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))
        for kf in range(nsp):
            if a.fold >= 0 and kf != a.fold:
                continue
            rd = lambda f: [l.split(",")[0] for l in open(os.path.join(B, c, "splits", f)
                            ).read().splitlines()[1:] if l.strip()]
            tr, te = rd(f"train_{kf}.csv"), rd(f"test_{kf}.csv")
            if all((c, s_) in done_folds for s_ in te):
                print(f"[{c} fold{kf}] 已完成, 跳过", flush=True); continue
            print(f"\n[{c} fold{kf}] train={tr} test={te} "
                  f"(训练 patch 约 {sum(1 for _ in tr) * 3000})", flush=True)
            dtr = HESTPatchDataset(tr, c, genes)
            # 作者 init() 按数据集把 projection_dim 设成该数据集的基因数；这里是 50。
            cfg = {"projection_dim": len(genes), "temperature": 1.0,
                   "embedding_dim": 2048, "model_name": "resnet50", "dropout": 0.1}
            model = HECLIPModel(cfg).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
            dl = DataLoader(dtr, batch_size=a.batch_size, shuffle=True, num_workers=4, drop_last=True)
            model.train()
            for ep in range(a.epochs):
                tot = n = 0
                for b in dl:
                    b = {k2: v.to(dev) for k2, v in b.items()}
                    loss = model(b)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item() * b["image"].shape[0]; n += b["image"].shape[0]
                if ep % 10 == 0 or ep == a.epochs - 1:
                    print(f"    ep{ep:3d} loss={tot / max(n,1):.4f}", flush=True)
            model.eval()
            _, ref_spot, ref_expr = embed_all(model, dtr, dev, a.batch_size)
            for sid in te:
                dte = HESTPatchDataset([sid], c, genes)
                q_img, _, Yte = embed_all(model, dte, dev, a.batch_size)
                pred = retrieve(q_img, ref_spot, ref_expr, k=a.k)
                v = float(E.per_gene_pcc(pred, Yte.numpy()).mean())
                res.setdefault(sid, {"cohort": c, "folds": []})["folds"].append(v)
                print(f"    test={sid} PCC={v:.4f}", flush=True)
                os.makedirs("results", exist_ok=True)
                json.dump({s2: {"cohort": d2["cohort"], "folds": d2["folds"],
                                "pcc": float(np.mean(d2["folds"]))} for s2, d2 in res.items()},
                          open(a.out, "w"), indent=2, ensure_ascii=False)

    out = {s: {"cohort": d["cohort"], "folds": d["folds"], "pcc": float(np.mean(d["folds"]))}
           for s, d in res.items()}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== HECLIP: {len(out)} 样本 ===")
    for c in sorted(set(d["cohort"] for d in out.values())):
        v = [d["pcc"] for d in out.values() if d["cohort"] == c]
        print(f"  {c:10s} n={len(v):3d} PCC={np.mean(v):.4f} [{min(v):.4f},{max(v):.4f}]")
    print(f"已存 {a.out}")


if __name__ == "__main__":
    main()
