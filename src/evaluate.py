# -*- coding: utf-8 -*-
"""评测 —— per-gene Pearson（HEST 同款主指标），在 top-K HVG 上取平均。"""
import numpy as np


def topk_hvg(expr_ref, k=50):
    """按参考(训练)表达的方差选 top-K 高变基因索引。"""
    v = expr_ref.var(0)
    k = min(k, expr_ref.shape[1])
    return np.argsort(v)[-k:]


def per_gene_pcc(pred, true, gene_idx=None):
    """逐基因跨 spot 的 Pearson；常量基因记 0。返回 (n_gene,) 数组。"""
    if gene_idx is not None:
        pred, true = pred[:, gene_idx], true[:, gene_idx]
    p = pred - pred.mean(0); t = true - true.mean(0)
    num = (p * t).sum(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 1e-8, num / den, 0.0)
    return np.nan_to_num(r, nan=0.0)


def summarize(pcc):
    pcc = np.asarray(pcc, dtype=float)
    return dict(mean=float(pcc.mean()), std=float(pcc.std()),
                median=float(np.median(pcc)), n=int(pcc.size))
