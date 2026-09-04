# -*- coding: utf-8 -*-
"""
主流程 —— 冻结编码器 + 轻量对齐 + 检索式 H&E→ST 预测。
核心实验（论文卖点）：同一管线，只换 ST 编码器 → 比较 per-gene PCC。

用法:
  python run.py --synthetic --aligner cca          # 本地端到端 smoke（无需 torch）
  python run.py --synthetic --aligner mlp --jepa 0.5
  python run.py --config config.yaml --aligner mlp # 真实数据（HPC）

划分:
  cross_slide / cross_patient : leave-one-out（推荐，防空间泄漏、最能体现 ST 编码器）
  within_slide                : 片内 A 训 / B 预测 + 空间排除（imputation 兜底）
"""
import argparse, json, os, numpy as np
import data as D, evaluate as E, retrieval as R
from align import build_aligner


def folds_leave_one_out(groups):
    for g in sorted(set(groups.tolist())):
        te = (groups == g)
        yield str(g), ~te, te


def run(dset, aligner_name, split, k=50, hvg=50, aligner_kw=None, seed=0):
    aligner_kw = aligner_kw or {}
    rng = np.random.default_rng(seed)
    img, expr, coords = dset["img"], dset["expr"], dset["coords"]
    st_dict = dset["st"]
    group = dset["slide"] if split in ("cross_slide", "within_slide") else dset["patient"]

    # 预先构造 fold 的 (train_mask, test_mask, exclude_fn)
    folds = []
    if split in ("cross_slide", "cross_patient"):
        for name, tr, te in folds_leave_one_out(group):
            folds.append(dict(name=name, tr=tr, te=te, exclude=None))
    elif split == "within_slide":
        # 每片随机切 A(参考/训练) / B(预测)，B 从本片 A 检索 + 空间排除
        A = np.zeros(len(img), bool)
        for g in sorted(set(dset["slide"].tolist())):
            gi = np.where(dset["slide"] == g)[0]
            a = rng.choice(gi, size=len(gi) // 2, replace=False)
            A[a] = True
        for g in sorted(set(dset["slide"].tolist())):
            te = (dset["slide"] == g) & (~A)
            ref = (dset["slide"] == g) & A
            folds.append(dict(name=str(g), tr=ref, te=te, exclude="spatial"))
        train_align_mask = A                      # 对齐全局在所有 A 上学
    else:
        raise ValueError(split)

    results = {enc: [] for enc in st_dict}
    floor = []
    for f in folds:
        tr, te = f["tr"], f["te"]
        # 对齐训练集：跨组用该 fold 的 train；within_slide 用全局 A
        atr = train_align_mask if split == "within_slide" else tr
        gidx = E.topk_hvg(expr[tr], hvg)           # HVG 在参考(训练)表达上选
        excl = None
        if f["exclude"] == "spatial":
            radius = 0.06
            excl = R.spatial_exclude_mask(coords[te], coords[tr], radius)

        # 图像地板（与 ST 编码器无关，每 fold 算一次）
        pf = R.image_floor(img[te], img[tr], expr[tr], k=k, exclude=excl)
        floor.append(E.per_gene_pcc(pf, expr[te], gidx).mean())

        # 逐 ST 编码器：对齐 → 跨模态检索 → PCC
        for enc, st in st_dict.items():
            al = build_aligner(aligner_name, **aligner_kw).fit(img[atr], st[atr])
            q = al.project_img(img[te])
            ref = al.project_st(st[tr])
            pred = R.retrieve_cross_modal(q, ref, expr[tr], k=k, exclude=excl)
            results[enc].append(E.per_gene_pcc(pred, expr[te], gidx).mean())

    summary = {enc: dict(mean=float(np.mean(v)), std=float(np.std(v)), folds=len(v))
               for enc, v in results.items()}
    summary["__image_floor__"] = dict(mean=float(np.mean(floor)), std=float(np.std(floor)),
                                      folds=len(floor))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--aligner", default="cca", choices=["cca", "mlp"])
    ap.add_argument("--split", default="cross_slide",
                    choices=["cross_slide", "cross_patient", "within_slide"])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--proj", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=512, help="MLP 隐层宽度；0=线性头(低容量,推荐做 ST 对比)")
    ap.add_argument("--jepa", type=float, default=0.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.synthetic:
        dset = D.make_synthetic()
        print(f"[synthetic] N={len(dset['img'])}, slides={len(set(dset['slide']))}, "
              f"ST 编码器={list(dset['st'])}")
    else:
        import yaml
        cfg = yaml.safe_load(open(args.config))
        dset = D.load_real(cfg["h5ad"], cfg["img_key"], cfg["st_keys"],
                           expr_layer=cfg.get("expr_layer"),
                           slide_col=cfg.get("slide_col", "slide_id"),
                           patient_col=cfg.get("patient_col"),
                           spatial_key=cfg.get("spatial_key", "spatial"))
        print(f"[real] N={len(dset['img'])}, slides={len(set(dset['slide']))}, "
              f"ST 编码器={list(dset['st'])}")

    akw = {}
    if args.proj: akw["n_components"] = args.proj
    if args.aligner == "mlp":
        akw.update(epochs=args.epochs, jepa_weight=args.jepa, hidden=args.hidden)

    summ = run(dset, args.aligner, args.split, k=args.k, hvg=args.hvg, aligner_kw=akw)

    print(f"\n=== per-gene PCC（top-{args.hvg} HVG, {args.split}, aligner={args.aligner}"
          f"{', jepa='+str(args.jepa) if args.aligner=='mlp' else ''}）===")
    rows = sorted([(k, v) for k, v in summ.items() if not k.startswith("__")],
                  key=lambda x: -x[1]["mean"])
    for enc, v in rows:
        print(f"  {enc:16s} {v['mean']:.4f} ± {v['std']:.4f}  ({v['folds']} folds)")
    fl = summ["__image_floor__"]
    print(f"  {'[image-floor]':16s} {fl['mean']:.4f} ± {fl['std']:.4f}  （不用 ST 编码器）")

    os.makedirs(args.out, exist_ok=True)
    op = os.path.join(args.out, f"pcc_{args.aligner}_{args.split}.json")
    json.dump(summ, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n结果已存 {op}")


if __name__ == "__main__":
    main()
