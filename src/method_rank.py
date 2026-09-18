#!/usr/bin/env python
"""方法级检验：PCC 排序与 σ 排序是否一致？

这是全篇的枢纽。此前只在**编码器**之间测过（ρ=0.987，排序一致 ⇒ 指标只压缩幅度，不误导选择）。
但编码器共用同一下游回归器与同一栅格，输出分辨率本就相近；**方法**之间不同——
iStar 是超分辨方法，imageKNN/BLEEP 是检索方法（输出被 k 个参考谱平均而天然粗化）。
若 σ 排序在方法之间翻转，则「领域用一个会选错方法的指标」成立。

数据来源：母项目导出的逐 bin 预测（同一批 bin、同一批基因、同一真值）。
**评测本身独立设计**，不沿用母项目的协议：
  · HVG 只在 train 上选（防泄漏）
  · σ 用阶梯匹配（估计量 A，与 25 塔扫描 / nine.py 同口径）
  · 只在 split=='test' 上评分；within-slide 文件的 excluded_margin 天然排除
  · 5 个 ST 编码器消融合并为 1 个代表（它们是同一方法的变体，不是独立方法）

注意：该导出基于旧表达基准（200 基因面板），**其绝对值不可与新基准的数字并列**；
但本检验只用到文件内部的相对排序，不受基准影响。
"""
import argparse, json
import numpy as np
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, kendalltau

PX = {"Visium_HD_Human_Colon_Cancer_P2": 3.6499, "Visium_HD_Human_Colon_Cancer_P5": 3.6526}

# 5 个 ST 编码器消融只保留一个代表 —— 它们是同一方法换内部组件，不是独立方法
ABLATIONS = {"stEnc_st_pca", "stEnc_NicheFormer", "stEnc_scGPT_spatial",
             "stEnc_scGPT", "stEnc_Novae"}

# iStar 默认排除：内部记录 记录了两处未解决的不对称 ——
#   ① 它被喂的是 log-expr 而非原始 counts；
#   ② 其余方法用 Hibou-L 特征，而 iStar 用自带 HIPT 骨干。
# 在这两处修好之前，把它与其余方法并列排名是无效比较（骨干与输入都不同）。
# 需要包含时显式传 --include_istar，且结论必须标注该限制。
ISTAR = "baseline_iStar_official"
NICE = {"ours_collaboratorST": "ours", "baseline_Ridge_HEST": "Ridge(HEST协议)",
        "baseline_BLEEPstyle": "BLEEP式检索", "baseline_MLPregression": "MLP回归",
        "baseline_imageKNN": "imageKNN(下界)", "baseline_iStar_official": "iStar官方"}


def build_operator(xy_um, k=8, cut_um=29.0):
    n = xy_um.shape[0]
    d, idx = cKDTree(xy_um).query(xy_um, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    ok = d <= cut_um
    rows = np.repeat(np.arange(n), ok.sum(1))
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, idx[ok])), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel(); iso = deg == 0; deg[iso] = 1.0
    W = (sparse.identity(n, format="csr", dtype=np.float32)
         + sparse.diags(1.0 / deg).tocsr() @ A) * 0.5
    return W.astype(np.float32), float(deg[~iso].mean())


def calib_sigma(W, xy, cps, n_seed=192, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.choice(xy.shape[0], size=min(n_seed, xy.shape[0]), replace=False)
    V = np.zeros((xy.shape[0], len(s)), np.float32); V[s, np.arange(len(s))] = 1.0
    out, t = {}, 0
    for cp in cps:
        while t < cp:
            V = W @ V; t += 1
        w = np.maximum(V, 0).astype(np.float64); tot = w.sum(0) + 1e-12
        mx = (w * xy[:, [0]]).sum(0) / tot; my = (w * xy[:, [1]]).sum(0) / tot
        m2 = ((w * xy[:, [0]] ** 2).sum(0) + (w * xy[:, [1]] ** 2).sum(0)) / tot
        out[cp] = float(np.sqrt(max(np.median(m2 - mx ** 2 - my ** 2), 0.0) / 2.0))
    return out


def pgp(pred, true):
    p = pred - pred.mean(0); t = true - true.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    return np.where(den > 1e-8, (p * t).sum(0) / den, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--tmax", type=int, default=2048)
    ap.add_argument("--out", default="results/method_rank.json")
    ap.add_argument("--include_istar", action="store_true",
                    help="纳入 iStar（骨干与输入均不对称，结论须标注该限制）")
    args = ap.parse_args()

    cps = [1]
    while cps[-1] < args.tmax:
        cps.append(cps[-1] * 2)

    a = ad.read_h5ad(args.h5ad)
    X = np.asarray(a.X.todense() if sparse.issparse(a.X) else a.X, np.float32)
    slide = a.obs["slide_id"].astype(str).values
    split = a.obs["split"].astype(str).values
    methods = [m for m in a.layers if m not in ABLATIONS]
    if not args.include_istar and ISTAR in methods:
        methods.remove(ISTAR)
        print(f"已排除 {ISTAR}：骨干(HIPT vs Hibou-L)与输入(log-expr vs counts)均不对称，"
              f"与其余方法并列排名无效（内部记录）", flush=True)
    print(f"{a.n_obs} bin × {a.n_vars} 基因；方法 {len(methods)}: {methods}", flush=True)

    per = {m: {"pcc": [], "sig": [], "flag": []} for m in methods}
    for sl in sorted(set(slide)):
        te = (slide == sl) & (split == "test")
        tr = (slide == sl) & (split == "train")
        if te.sum() == 0:
            continue
        # HVG 只在 train 上选 —— 防止评测泄漏
        v = X[tr].var(0)
        gidx = np.argsort(-v)[:args.hvg]
        # 公共有限子集：iStar 在 P2 上有 0.09% 散点 nan（超分辨网格外的 bin），
        # 它们会污染整列的 mean/sum，使该方法全部 50 个基因的相关变成 nan。
        # 更重要的是协议问题——各方法必须在**同一批 bin** 上评分才可比，
        # 故取所有方法均有有限预测的交集，而不是各自丢各自的 nan。
        idx_te = np.where(te)[0]
        ok_rows = np.ones(len(idx_te), bool)
        for m in methods:
            ok_rows &= np.isfinite(np.asarray(a.layers[m][idx_te][:, gidx], np.float32)).all(1)
        drop = int((~ok_rows).sum())
        if drop:
            print(f"  公共有限子集：剔除 {drop}/{len(idx_te)} 行 ({100*drop/len(idx_te):.3f}%)", flush=True)
        idx_te = idx_te[ok_rows]
        te = np.zeros_like(te); te[idx_te] = True

        y = X[te][:, gidx]
        xy = np.stack([a.obs["x_um"].values[te], a.obs["y_um"].values[te]], 1).astype(np.float64)
        W, deg = build_operator(xy)
        sig = calib_sigma(W, xy, cps)
        # 阶梯：真值扩散到 t 后与原真值的相关
        lad, cur, t = {}, y.copy(), 0
        for cp in cps:
            while t < cp:
                cur = W @ cur; t += 1
            lad[cp] = float(np.nanmean(pgp(cur, y)))
        pts = sorted((sig[c], lad[c]) for c in cps)
        print(f"\n[{sl}] test={int(te.sum())} 平均度={deg:.2f}  "
              f"σ({cps[0]})={sig[cps[0]]:.1f} σ({cps[-1]})={sig[cps[-1]]:.0f}µm  "
              f"阶梯 {lad[cps[0]]:.3f}→{lad[cps[-1]]:.3f}", flush=True)
        for m in methods:
            P = np.asarray(a.layers[m][te][:, gidx], np.float32)
            pc = float(np.nanmean(pgp(P, y)))
            eq, fl = float("nan"), "右删失"
            if pc >= pts[0][1]:
                eq, fl = pts[0][0], "低于最细档"
            else:
                for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
                    if v0 >= pc >= v1:
                        eq = s0 + (v0 - pc) / max(v0 - v1, 1e-12) * (s1 - s0); fl = "ok"; break
            per[m]["pcc"].append(pc); per[m]["sig"].append(eq); per[m]["flag"].append(fl)
            print(f"    {NICE.get(m, m):<18} PCC={pc:.4f}  eqσ={eq:7.1f}µm  {fl}", flush=True)

    # PCC 与 σ 必须用**同一批折**聚合。此前 PCC 取全部折、σ 只取 flag==ok 的折，
    # 于是某方法在一折上退化（PCC=nan）时，它的 PCC 是 nan 而 σ 仍有值，
    # 排序把 nan 排进去 ⇒ 名次变动是假象。实测 iStar 在 P2 上即如此。
    rows, dropped = [], []
    for m in methods:
        keep = [i for i, f in enumerate(per[m]["flag"])
                if f == "ok" and np.isfinite(per[m]["pcc"][i]) and np.isfinite(per[m]["sig"][i])]
        if len(keep) < len(per[m]["flag"]):
            dropped.append((NICE.get(m, m), len(keep), len(per[m]["flag"])))
        if not keep:
            continue
        rows.append((NICE.get(m, m),
                     float(np.mean([per[m]["pcc"][i] for i in keep])),
                     float(np.mean([per[m]["sig"][i] for i in keep])),
                     len(keep), len(per[m]["flag"])))
    if dropped:
        print("\n⚠ 折数不全的方法（PCC 与 σ 均只用其有效折，不与全折方法并列排名）:")
        for n, k, t in dropped:
            print(f"    {n}: {k}/{t} 折有效")
    print("\n" + "=" * 66)
    print("方法".ljust(20) + "PCC".rjust(9) + "等价σ".rjust(10) + "  ok/折")
    print("-" * 66)
    for n, p, s, k, tot in sorted(rows, key=lambda r: -r[1]):
        print(n.ljust(20) + f"{p:.4f}".rjust(9) + f"{s:.1f}".rjust(10) + f"  {k}/{tot}")

    full = [r for r in rows if r[3] == r[4]]        # 只用全折方法做排序检验
    if len(full) < len(rows):
        print(f"\n排序检验只用全折方法 {len(full)}/{len(rows)} 个")
    rows_all, rows = rows, full
    if len(rows) >= 4:
        pc = np.array([r[1] for r in rows]); sg = np.array([r[2] for r in rows])
        if not (np.all(np.isfinite(pc)) and np.all(np.isfinite(sg))):
            raise SystemExit("仍有非有限值，拒绝给出判读")
        rho = spearmanr(pc, -sg).statistic; tau = kendalltau(pc, -sg).statistic
        byp = sorted(rows, key=lambda r: -r[1]); bys = sorted(rows, key=lambda r: r[2])
        rp = {r[0]: i + 1 for i, r in enumerate(byp)}; rs = {r[0]: i + 1 for i, r in enumerate(bys)}
        print(f"\nPCC 排序 vs σ 排序: Spearman ρ = {rho:.3f}   Kendall τ = {tau:.3f}")
        print(f"  PCC 第一: {byp[0][0]}   σ 第一: {bys[0][0]}")
        print(f"  按 PCC 选出的第一名在 σ 上排第 {rs[byp[0][0]]}/{len(rows)}")
        mv = sorted(((abs(rs[n] - rp[n]), n, rp[n], rs[n]) for n, _, _, _, _ in rows), reverse=True)
        print(f"  最大名次变动 {mv[0][0]} 位: {mv[0][1]} (PCC 第{mv[0][2]} → σ 第{mv[0][3]})")
        print(f"  PCC 跨度 {pc.max()/pc.min():.3f}x   σ 跨度 {sg.max()/sg.min():.2f}x   "
              f"放大 {(sg.max()/sg.min())/(pc.max()/pc.min()):.2f}x")
        if not np.isfinite(rho):
            raise SystemExit("ρ 为 nan，拒绝判读")
        # 不再用单一阈值下结论 —— 该模板已连续三次误报（nan 一次、ρ 恰为 0.900 两次），
        # 每次都打印出「正是我期待看到的那句话」。判读改为只陈述事实，结论留给人。
        top_same = byp[0][0] == bys[0][0]
        big = [m for m in mv if m[0] >= 2]
        print(f"\n事实陈述（不下结论）:")
        print(f"  · ρ={rho:.3f}  τ={tau:.3f}")
        print(f"  · PCC 第一与 σ 第一{'相同' if top_same else '不同'}")
        print(f"  · 名次变动 ≥2 位的方法: {len(big)}/{len(rows)}"
              + (f"  ({', '.join(f'{n}:{a}→{b}' for _, n, a, b in big[:3])})" if big else ""))
        print(f"  · 换算：{100*(pc.max()-pc.min())/pc.min():.1f}% 的分数差 对应 "
              f"{sg.max()/sg.min():.2f} 倍的分辨率差")
        json.dump({"rows": rows, "rows_all": rows_all, "dropped": dropped, "spearman": float(rho), "kendall": float(tau),
                   "pcc_span": float(pc.max()/pc.min()), "sigma_span": float(sg.max()/sg.min())},
                  open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"已存 {args.out}")


if __name__ == "__main__":
    main()
