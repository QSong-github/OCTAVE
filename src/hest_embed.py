# -*- coding: utf-8 -*-
"""
跨 72 样本的模型半 —— HEST-benchmark 官方协议下的"报告 PCC"。

首版是我自己复刻协议, 有四处偏差, 数字全废:
  Ridge alpha 我用 1e4, 官方是 100/(D×G)≈9.8e-4 —— 差 7 个数量级, 回归被压成常数
  降维       我漏了 StandardScaler
  归一化     我用 CP10k+log1p, 官方 normalize_adata 只做 sc.pp.log1p
  对齐       我按行序, 官方按 barcode(Xenium 的 patch 数 < spot 数, 行序必错)
本版直接调用 HEST 自带的 load_adata / train_test_reg, 不再复刻。

官方默认(BenchmarkConfig): normalize=True, dimreduce='PCA', latent_dim=256, method='ridge'
划分 CSV: sample_id,patches_path,expr_path
编码器: torchvision ResNet50(ImageNet) —— HEST 默认配置内, 权重无需申请。
        编码器不是本文的变量, 之后按同一流程追加更强的即可。
"""
import os, sys, glob, json, argparse, numpy as np

print("=== 冒烟测试 ===", flush=True)
import torch, h5py
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    _ = (torch.rand(64, 64, device="cuda") @ torch.rand(64, 64, device="cuda")).sum().item()

sys.path.insert(0, "/path/to/he2st/HEST/src")
from hest.bench.st_dataset import load_adata            # 官方: barcode 子集 + log1p
from hest.bench.trainer import train_test_reg           # 官方: alpha=100/(D*G), lsqr, no intercept
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

B = "/path/to/he2st/HEST/eval/bench_data"
EMB = "/path/to/systema4ST/results/hest_emb"


# 全部开放权重(无需 HF 授权)。UNI/Virchow2/GigaPath/H-optimus/CONCH 是 gated,
# 本地也无缓存 —— HEST 这条线暂时覆盖不到, 需账号授权。
# Visium HD 那条线用母项目已提取的 .npy 嵌入, 不受此限(见 tower_sweep.py 的 25 塔)。
HF_REPOS = {"phikon": "owkin/phikon", "phikon_v2": "owkin/phikon-v2",
            "hibou_b": "histai/hibou-b",
            "dinov2_large": "facebook/dinov2-large",
            "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m"}
TIMM_SIZE = {"kaiko_vitl14": 518}      # 该模型 pretrain 于 518, 224 会断言失败
TIMM_REPOS = {"lunit_vits8": "hf_hub:1aurent/vit_small_patch8_224.lunit_dino",
              "kaiko_vitb16": "hf_hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
              "kaiko_vits16": "hf_hub:1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms",
              "kaiko_vitl14": "hf_hub:1aurent/vit_large_patch14_reg4_dinov2.kaiko_ai_towards_large_pathology_fms"}
CIGA_CKPT = "/path/to/systema4ST/stflow_run/weights/fm_v1/ciga/tenpercent_resnet18.ckpt"
ALL_ENC = ["resnet50", "ciga"] + list(HF_REPOS) + list(TIMM_REPOS)


def encoder(name, dev):
    if name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        return m.eval().to(dev), None
    if name == "ciga":            # Ciga 等人自监督病理 ResNet18(GitHub release, 开放)
        import torchvision, torch.nn as nn
        m = torchvision.models.resnet18(weights=None); m.fc = nn.Identity()
        sd = torch.load(CIGA_CKPT, map_location="cpu")["state_dict"]
        sd = {k.replace("model.resnet.", ""): v for k, v in sd.items() if "fc." not in k}
        m.load_state_dict(sd, strict=True)
        return m.eval().to(dev), None
    if name in TIMM_REPOS:
        import timm
        m = timm.create_model(TIMM_REPOS[name], pretrained=True, num_classes=0)
        return m.eval().to(dev), None
    from transformers import AutoModel
    m = AutoModel.from_pretrained(HF_REPOS[name], trust_remote_code=True).eval().to(dev)
    return m, "hf"


@torch.no_grad()
def embed_and_barcodes(model, kind, path, dev, bs=256, size=224):
    with h5py.File(path, "r") as h:
        bk = "barcodes" if "barcodes" in h else "barcode"
        bc = np.asarray(h[bk][:]).flatten().astype(str).tolist()
        n = h["img"].shape[0]
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        out = []
        for i in range(0, n, bs):
            x = torch.tensor(np.asarray(h["img"][i:i + bs]), device=dev).float()
            x = x[..., :3].permute(0, 3, 1, 2) / 255.0
            if size != 224:
                x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
            if kind == "hf":
                o = model(pixel_values=(x - mean) / std)
                f = o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else o.pooler_output
            else:
                f = model((x - mean) / std)
            out.append(f.cpu().numpy())
    return np.concatenate(out).astype(np.float32), bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon", choices=ALL_ENC)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=256,
                    help="kaiko_vitl14 在 518×518 下 256 会 CUDA OOM，用 64")
    args = ap.parse_args()
    args.out = args.out or f"results/hest_reported_pcc_{args.encoder}.json"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, kind = encoder(args.encoder, dev)
    os.makedirs(EMB, exist_ok=True)

    per_sample, cohort_of = {}, {}
    for c in sorted(os.listdir(B)):
        if not os.path.isdir(os.path.join(B, c, "adata")):
            continue
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        cache = {}

        def get(sid):
            if sid in cache:
                return cache[sid]
            f = os.path.join(EMB, f"{sid}_{args.encoder}.npz")
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True); X, bc = z["X"], z["bc"].tolist()
            else:
                X, bc = embed_and_barcodes(model, kind, os.path.join(B, c, "patches", f"{sid}.h5"),
                                           dev, bs=args.batch, size=TIMM_SIZE.get(args.encoder, 224))
                np.savez(f, X=X, bc=np.array(bc, dtype=object))
            # 官方: 按 barcode 子集 adata, 只取 50 基因, log1p
            Y = load_adata(os.path.join(B, c, "adata", f"{sid}.h5ad"),
                           genes=genes, barcodes=bc, normalize=True).values.astype(np.float32)
            assert X.shape[0] == Y.shape[0], f"{sid}: emb {X.shape[0]} vs expr {Y.shape[0]}"
            cache[sid] = (X, Y)
            cohort_of[sid] = c
            print(f"  {c}/{sid}: X{X.shape} Y{Y.shape}", flush=True)
            return cache[sid]

        n_splits = len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))
        for k in range(n_splits):
            rd = lambda f: [l.split(",")[0] for l in
                            open(os.path.join(B, c, "splits", f)).read().splitlines()[1:] if l.strip()]
            tr, te = rd(f"train_{k}.csv"), rd(f"test_{k}.csv")
            Xtr = np.concatenate([get(s)[0] for s in tr]); Ytr = np.concatenate([get(s)[1] for s in tr])
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("PCA", PCA(n_components=min(args.latent_dim, Xtr.shape[1],
                                                          Xtr.shape[0] - 1), random_state=0))]).fit(Xtr)
            Ztr = pipe.transform(Xtr)
            for s in te:
                Xs, Ys = get(s)
                r, _ = train_test_reg(Ztr, pipe.transform(Xs), Ytr, Ys, genes=genes, method="ridge")
                v = float(r["pearson_mean"])
                per_sample.setdefault(s, []).append(v)
                print(f"  [{c} fold{k}] test={s} pearson_mean={v:.4f}", flush=True)

    out = {s: {"cohort": cohort_of[s], "folds": v, "pcc": float(np.mean(v))}
           for s, v in per_sample.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== {len(out)} 样本 ===")
    for c in sorted(set(cohort_of.values())):
        v = [d["pcc"] for d in out.values() if d["cohort"] == c]
        print(f"  {c:10s} n={len(v):3d} PCC 均值={np.mean(v):.4f} 范围=[{min(v):.4f},{max(v):.4f}]")
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()
