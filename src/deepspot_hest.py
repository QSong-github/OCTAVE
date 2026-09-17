# -*- coding: utf-8 -*-
"""DeepSpot（Nonchev et al., medRxiv 2025.02.09.25321567）在 HEST 基准上：模型原样用作者代码（deepspot.spot.model.DeepSpot），只换数据源。
输入按作者协议（§4.2.1）：spot tile 特征、3×3 子 tile 特征（SUM 聚合）、阵列坐标半径 1 内的邻居 spot 特征（MAX 聚合，
零填充到该数据集内的最大邻居数，与作者 DeepSpotDataLoader 一致）；特征由 src/deepspot_feats.py 生成。
目标：官方 50 基因、log1p（与本项目其它方法同口径）；训练集 StandardScaler 标准化（作者 normalize='standard'），预测后逆变换再算逐基因 PCC。
训练：作者 notebook 配置——MSE、AdamW lr 1e-4 wd 1e-6、batch 1024、n_ensemble 10、dropout 0.3、seed 2024、
EarlyStopping(monitor='train_step', patience=3, min_delta=0.01, mode='min')、max_epochs=10；
--match_steps>0 时按步数反推 max_epochs（作者在 Tumor Profiler 上约 50 步/epoch × 10 epoch ≈ 500 步；同 HisToGene 适配器的做法）。
划分：HEST 官方折；不做空间下采样（逐 spot 模型不需要）。输出 results/deepspot_hest_{enc}{tag}.json（与 histogene_hest.json 同格式）。"""
import os, sys, glob, json, math, argparse, numpy as np, torch
sys.path.insert(0, "/path/to/he2st/HEST/src"); sys.path.insert(0, "/path/to/systema4ST/src")
from hest.bench.st_dataset import load_adata
import evaluate as E
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from deepspot.spot.model import DeepSpot
B = "/path/to/he2st/HEST/eval/bench_data"; FE = "/path/to/systema4ST/results/deepspot_emb"

def load_sample(sid, cohort, enc, genes, r=1):
    z = np.load(f"{FE}/{enc}/{cohort}/{sid}.npz", allow_pickle=True)
    spot, sub, bc, xy = z["spot"].astype(np.float32), z["sub"].astype(np.float32), z["bc"].astype(str).tolist(), z["xy_array"]
    Y = load_adata(os.path.join(B, cohort, "adata", f"{sid}.h5ad"), genes=genes, barcodes=bc, normalize=True).values.astype(np.float32)
    key = {}
    for i, (x, y) in enumerate(xy.tolist()): key.setdefault((x, y), []).append(i)
    nb = [[j for dx in range(-r, r + 1) for dy in range(-r, r + 1) for j in key.get((x + dx, y + dy), []) if j != i]
          for i, (x, y) in enumerate(xy.tolist())]                      # 作者 compute_neighbors：|Δx|≤r 且 |Δy|≤r 的其它 spot
    return dict(spot=spot, sub=sub, Y=Y, nb=nb, n=len(bc))

class DS(Dataset):
    def __init__(self, samples, max_n, ykey):
        self.items = [(s, i) for s in samples for i in range(s["n"])]; self.max_n = max_n; self.ykey = ykey
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        s, i = self.items[k]; d = s["spot"].shape[1]
        nbf = s["spot"][s["nb"][i]] if s["nb"][i] else np.zeros((0, d), np.float32)
        nbf = np.concatenate([nbf, np.zeros((self.max_n - len(nbf), d), np.float32)])   # 作者 add_zero_padding
        return (s["spot"][i][None], s["sub"][i], nbf), s[self.ykey][i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--match_steps", type=int, default=0); ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--cohorts", default="SKCM,HCC,LUNG,PAAD,COAD,READ,IDC,LYMPH_IDC,PRAD,CCRCC"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); out_f = f"results/deepspot_hest_{a.encoder}{a.tag}.json"
    res = {}
    if os.path.exists(out_f):
        res = {k: {"cohort": v["cohort"], "folds": v["folds"]} for k, v in json.load(open(out_f)).items()}; print(f"续跑: 已有 {len(res)} 个样本", flush=True)
    for c in a.cohorts.split(","):
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        for kf in range(len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))):
            rd = lambda f: [l.split(",")[0] for l in open(os.path.join(B, c, "splits", f)).read().splitlines()[1:] if l.strip()]
            tr, te = rd(f"train_{kf}.csv"), rd(f"test_{kf}.csv")
            if all(s in res for s in te): print(f"[{c} fold{kf}] 已完成, 跳过", flush=True); continue
            data = {s: load_sample(s, c, a.encoder, genes) for s in tr + te}
            scaler = StandardScaler().fit(np.concatenate([data[s]["Y"] for s in tr]))
            for s in tr: data[s]["Ys"] = scaler.transform(data[s]["Y"]).astype(np.float32)
            max_tr = max(len(n) for s in tr for n in data[s]["nb"]); d = data[tr[0]]["spot"].shape[1]
            ds = DS([data[s] for s in tr], max_tr, "Ys"); spe = math.ceil(len(ds) / a.batch)
            n_ep = a.epochs if a.match_steps <= 0 else max(1, round(a.match_steps / spe))
            print(f"\n[{c} fold{kf}] train={len(tr)} test={len(te)} spots={len(ds)} steps/epoch={spe} max_epochs={n_ep} max_nb={max_tr} d={d}", flush=True)
            model = DeepSpot(input_size=d, output_size=len(genes), scaler=scaler)      # 其余全为作者默认值
            trainer = L.Trainer(max_epochs=n_ep, logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                                accelerator="auto", devices=1, callbacks=[EarlyStopping(monitor="train_step", patience=3, min_delta=0.01, mode="min")])
            trainer.fit(model, DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, persistent_workers=False))
            ep_run = trainer.current_epoch + 1; print(f"    训练 {ep_run} epoch, 末 loss={model.training_loss[-1]:.4f}", flush=True)
            model.eval(); dev = next(model.parameters()).device
            for s in te:
                max_te = max(len(n) for n in data[s]["nb"]); tds = DS([data[s]], max_te, "Y"); preds = []
                with torch.no_grad():
                    for X, _ in DataLoader(tds, batch_size=a.batch, shuffle=False, num_workers=2):
                        preds.append(model([x.to(dev).float() for x in X]).cpu().numpy())
                pred = model.inverse_transform(np.concatenate(preds)); v = float(E.per_gene_pcc(pred, data[s]["Y"]).mean())
                res.setdefault(s, {"cohort": c, "folds": []})["folds"].append(v); res[s]["epochs"] = ep_run; res[s]["n_spots"] = data[s]["n"]
                print(f"    test={s} n={data[s]['n']} PCC={v:.4f}", flush=True)
                os.makedirs("results", exist_ok=True)
                json.dump({k2: dict(v2, pcc=float(np.mean(v2["folds"]))) for k2, v2 in res.items()}, open(out_f, "w"), indent=2, ensure_ascii=False)
            del data, model, trainer; torch.cuda.empty_cache()
    print(f"\n=== DeepSpot[{a.encoder}{a.tag}]: {len(res)} 样本 ===")
    for c in sorted({v["cohort"] for v in res.values()}):
        v = [float(np.mean(d["folds"])) for d in res.values() if d["cohort"] == c]; print(f"  {c:<10} n={len(v):2d} mean PCC={np.mean(v):.4f}")
if __name__ == "__main__": main()
