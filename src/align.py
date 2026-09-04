# -*- coding: utf-8 -*-
"""
对齐器 —— 把冻结的【图像 embedding】和【ST embedding】投到共享空间。
编码器全程冻结，这里只学一个轻量对齐（或零训练）。可插拔:
  - cca   : Canonical Correlation Analysis，零深度学习、快，作强基线/地板
  - mlp   : 两个 MLP 投影头 + 对称 InfoNCE(+可选 JEPA 预测项)，需 torch
每个对齐器都提供 project_img(img)->shared 和 project_st(st)->shared。
检索在 shared 空间做（保证 ST 编码器真正参与）。
"""
import numpy as np


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class CCAAligner:
    """CCA：在 train 的 (img, st) 上拟合，两侧各自可独立投到共享空间。"""
    def __init__(self, n_components=24, max_iter=300, **_):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, img_tr, st_tr):
        from sklearn.cross_decomposition import CCA
        k = int(min(self.n_components, img_tr.shape[1], st_tr.shape[1], img_tr.shape[0] - 1))
        self.cca = CCA(n_components=k, max_iter=self.max_iter, tol=1e-4)
        self.cca.fit(img_tr, st_tr)
        return self

    def project_img(self, img):
        return _l2(np.asarray(self.cca.transform(img), dtype=np.float32))

    def project_st(self, st):
        c = self.cca                                    # 兼容不同 sklearn 版本的属性名
        ymean = getattr(c, "_y_mean", getattr(c, "y_mean_", 0.0))
        ystd  = getattr(c, "_y_std",  getattr(c, "y_std_", 1.0))
        Yc = (st - ymean) / ystd
        return _l2(np.asarray(Yc @ c.y_rotations_, dtype=np.float32))


class MLPInfoNCEAligner:
    """两个 MLP 投影头，对称 InfoNCE (+可选 JEPA 预测项)。需 torch。"""
    def __init__(self, n_components=64, hidden=512, epochs=40, lr=1e-3, batch=512,
                 temp=0.07, jepa_weight=0.0, device=None, seed=0, **_):
        self.p = dict(proj=n_components, hidden=hidden, epochs=epochs, lr=lr,
                      batch=batch, temp=temp, jepa=jepa_weight, seed=seed)
        self.device = device

    def _head(self, din, torch, nn):
        # hidden<=0 → 线性头（低容量，防图像头"记忆配对、绕过 ST 内容"的容量陷阱）
        if self.p["hidden"] <= 0:
            return nn.Linear(din, self.p["proj"])
        return nn.Sequential(nn.Linear(din, self.p["hidden"]), nn.GELU(),
                             nn.Linear(self.p["hidden"], self.p["proj"]))

    def fit(self, img_tr, st_tr):
        import torch, torch.nn as nn, torch.nn.functional as F
        torch.manual_seed(self.p["seed"])
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dev = dev
        self.fi = self._head(img_tr.shape[1], torch, nn).to(dev)
        self.fs = self._head(st_tr.shape[1], torch, nn).to(dev)
        I = torch.tensor(img_tr, dtype=torch.float32, device=dev)
        S = torch.tensor(st_tr, dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW(list(self.fi.parameters()) + list(self.fs.parameters()),
                                lr=self.p["lr"], weight_decay=1e-4)
        n = I.shape[0]; bs = min(self.p["batch"], n)
        for ep in range(self.p["epochs"]):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                zi = F.normalize(self.fi(I[idx]), dim=1)
                zs = F.normalize(self.fs(S[idx]), dim=1)
                logits = zi @ zs.T / self.p["temp"]
                lab = torch.arange(idx.shape[0], device=dev)
                loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
                if self.p["jepa"] > 0:                      # JEPA 式预测项：图像潜表征预测 ST 潜表征
                    loss = loss + self.p["jepa"] * (1 - (zi * zs).sum(1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.fi.eval(); self.fs.eval()
        return self

    def _proj(self, head, x):
        import torch, torch.nn.functional as F
        with torch.no_grad():
            z = F.normalize(head(torch.tensor(x, dtype=torch.float32, device=self.dev)), dim=1)
        return z.cpu().numpy().astype(np.float32)

    def project_img(self, img):
        return self._proj(self.fi, img)

    def project_st(self, st):
        return self._proj(self.fs, st)


def build_aligner(name, **kw):
    name = name.lower()
    if name == "cca":
        return CCAAligner(**kw)
    if name in ("mlp", "infonce", "mlp_infonce"):
        return MLPInfoNCEAligner(**kw)
    raise ValueError(f"unknown aligner: {name}")
