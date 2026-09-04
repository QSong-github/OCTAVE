# -*- coding: utf-8 -*-
"""
检索式预测 —— query 图像 → 对齐空间 → 检索参考集的 ST → 借其原始表达。
内存安全：逐 query 块算 top-k，绝不物化整个 (Nq×Nr) 相似度矩阵（百万级 spot 必需）。
  retrieve_cross_modal : query 图像投影 ↔ 参考 ST 投影（跨模态；ST 编码器在此起作用）—— 主路径
  image_floor          : query 图像 ↔ 参考图像（不用 ST 编码器）—— 地板对照
"""
import numpy as np


def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _block_topk(qb, ref, ref_expr, k, temp, exclude_b=None):
    """qb:(b,d) ref:(Nr,d) 都已归一化 → 该块每行 top-k 加权借表达 → (b,G)。"""
    sim = qb @ ref.T                                   # (b, Nr)
    if exclude_b is not None:
        sim[exclude_b] = -1e9
    kk = min(k, sim.shape[1])
    idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    row = np.arange(sim.shape[0])[:, None]
    w = sim[row, idx]
    w = np.exp((w - w.max(1, keepdims=True)) / temp)
    w /= w.sum(1, keepdims=True)
    return np.einsum("bk,bkg->bg", w, ref_expr[idx]).astype(np.float32)


def _retrieve(q, ref, ref_expr, k, temp, chunk, exclude):
    q, ref = _norm(q), _norm(ref)
    out = np.empty((q.shape[0], ref_expr.shape[1]), np.float32)
    for i in range(0, q.shape[0], chunk):
        eb = exclude[i:i + chunk] if exclude is not None else None
        out[i:i + chunk] = _block_topk(q[i:i + chunk], ref, ref_expr, k, temp, eb)
    return out


def retrieve_cross_modal(q_img_shared, ref_st_shared, ref_expr, k=50, temp=0.03,
                         chunk=2048, exclude=None):
    """主路径：query 图像投影 vs 参考 ST 投影。"""
    return _retrieve(q_img_shared, ref_st_shared, ref_expr, k, temp, chunk, exclude)


def image_floor(q_img, ref_img, ref_expr, k=50, temp=0.03, reduce_dim=64,
                chunk=2048, exclude=None, whiten=False):
    """地板：纯图像-图像检索借表达（不经 ST 编码器）。PCA(可选白化) 压一下再算余弦。"""
    from sklearn.decomposition import PCA
    d = min(reduce_dim, ref_img.shape[1], ref_img.shape[0] - 1)
    pca = PCA(n_components=d, whiten=whiten).fit(ref_img)   # whiten: 各主成分缩到单位方差(去相关+均衡)
    return _retrieve(pca.transform(q_img), pca.transform(ref_img), ref_expr, k, temp, chunk, exclude)


def spatial_exclude_mask(q_coords, ref_coords, radius):
    """within-slide 用（小规模）：距 query 半径内的参考点排除。返回 (Nq,Nr) bool。"""
    d2 = ((q_coords[:, None, :] - ref_coords[None, :, :]) ** 2).sum(-1)
    return d2 < radius ** 2
