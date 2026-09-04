# -*- coding: utf-8 -*-
"""
数据接口 —— H&E↔ST 冻结编码器对齐+检索 pipeline。

期望的真实数据 schema（每个数据集一个 .h5ad）:
  adata.obsm['<img_key>']     : 图像 embedding（冻结图像编码器输出）      (N, d_img)
  adata.obsm['<st_key_i>']    : ST embedding（冻结 ST 编码器输出，可多个）  (N, d_st)  —— 每个 ST 编码器一个 obsm key
  adata.X 或 layers['<expr>'] : 基因表达（检索目标 + 评测用）              (N, G)
  adata.obsm['spatial'] 或 obs[x,y] : 空间坐标                            (N, 2)
  adata.obs['<slide_col>']    : slide id（跨片划分用）
  adata.obs['<patient_col>']  : patient id（跨病人划分用；没有就用 slide）
所有编码器都冻结 —— 这里只吃 embedding，不含任何图像/ST 模型。

load_real() 需要 anndata（HPC 上 hest 环境有）；make_synthetic() 只用 numpy，
本地即可端到端 smoke（并演示"更好的 ST embedding → 更高 PCC"）。
"""
import numpy as np


def make_synthetic(n_slides=6, spots=800, n_latent=12, n_genes=200,
                   d_img=256, d_st=128, n_centers=8, seed=0):
    """合成配对数据：潜因子 Z 空间平滑；expr/img/st 都是 Z 的带噪线性映射。
    返回多个 ST 变体（好/差/随机）以验证 pipeline 能否分辨 ST 编码器质量。"""
    rng = np.random.default_rng(seed)
    n_vis = 2 * n_latent // 3                             # 图像只"看得到"前 2/3 因子；
    W_expr = rng.normal(size=(n_latent, n_genes))         # 剩下 1/3 是形态学看不出、只有 ST 才有的表达信息
    W_img  = rng.normal(size=(n_vis, d_img))
    W_sg   = rng.normal(size=(n_latent, d_st))            # 好 ST：捕获全部 Z（含形态学看不到的部分）
    W_sb   = rng.normal(size=(n_latent // 3, d_st))       # 差 ST：只捕获 1/3 因子

    img, expr, coords, slide, patient = [], [], [], [], []
    st_good, st_bad, st_rand = [], [], []
    for s in range(n_slides):
        xy = rng.uniform(size=(spots, 2))
        # 空间平滑的潜因子：K 个中心各带一个潜向量，按距离软加权
        centers = rng.uniform(size=(n_centers, 2))
        cvec = rng.normal(size=(n_centers, n_latent))
        d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # (spots, K)
        w = np.exp(-d2 / (2 * 0.12 ** 2)); w /= w.sum(1, keepdims=True)
        Z = w @ cvec + 0.05 * rng.normal(size=(spots, n_latent))     # (spots, n_latent)

        e = np.maximum(Z @ W_expr, 0) + 0.10 * rng.normal(size=(spots, n_genes))
        im = Z[:, :n_vis] @ W_img + 0.40 * rng.normal(size=(spots, d_img))  # 图像只见前 2/3 因子
        sg = Z @ W_sg + 0.15 * rng.normal(size=(spots, d_st))        # 好 ST：低噪、全因子
        sb = Z[:, :n_latent // 3] @ W_sb + 0.80 * rng.normal(size=(spots, d_st))
        sr = rng.normal(size=(spots, d_st))                          # 随机 ST：无信息

        img.append(im); expr.append(e); coords.append(xy)
        st_good.append(sg); st_bad.append(sb); st_rand.append(sr)
        slide.append(np.full(spots, s)); patient.append(np.full(spots, s // 2))

    cat = lambda L: np.concatenate(L).astype(np.float32)
    return {
        "img": cat(img), "expr": cat(expr), "coords": cat(coords),
        "slide": np.concatenate(slide), "patient": np.concatenate(patient),
        "genes": np.array([f"g{i}" for i in range(n_genes)]),
        "st": {"st_good": cat(st_good), "st_bad": cat(st_bad), "st_rand": cat(st_rand)},
    }


def load_real(h5ad_path, img_key, st_keys, expr_layer=None,
              slide_col="slide_id", patient_col=None, spatial_key="spatial",
              genes=None):
    """从 .h5ad 读成统一 dict。st_keys: list[str]，每个是一个 ST 编码器的 obsm key。"""
    import anndata as ad
    from scipy import sparse
    a = ad.read_h5ad(h5ad_path)
    X = a.layers[expr_layer] if expr_layer else a.X
    if sparse.issparse(X):
        X = X.toarray()
    expr = np.asarray(X, dtype=np.float32)
    if spatial_key in a.obsm:
        coords = np.asarray(a.obsm[spatial_key], dtype=np.float32)[:, :2]
    else:
        coords = a.obs[["x", "y"]].values.astype(np.float32)
    slide = a.obs[slide_col].astype(str).values
    patient = a.obs[patient_col].astype(str).values if patient_col and patient_col in a.obs else slide
    nz = lambda x: np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    st = {k: nz(a.obsm[k]) for k in st_keys}            # 部分 bin 的图像/ST 特征可能有 NaN → 清 0
    img = nz(a.obsm[img_key])
    n_nan = int(np.isnan(np.asarray(a.obsm[img_key])).any(1).sum())
    if n_nan:
        print(f"  ⚠ img_emb 有 {n_nan} 个含 NaN 的 bin，已清 0", flush=True)
    return {
        "img": img, "expr": nz(expr), "coords": nz(coords), "slide": slide, "patient": patient,
        "genes": np.asarray(a.var_names if genes is None else genes),
        "st": st,
    }
