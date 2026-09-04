import re, sys, json, numpy as np
L = open(sys.argv[1]).read()
agg = {}
for b in re.split(r"\n(?=═══ )", L):
    m = re.match(r"═══ (\S+)\s+(\S+)/fold0.*?训练 (\d+) 片", b)
    if not m: continue
    enc, coh, ntr = m.group(1), m.group(2), int(m.group(3))
    rows = {r.group(1): (float(r.group(2)), float(r.group(3)))
            for r in re.finditer(r"^\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)", b, re.M)}
    LAST = sys.argv[2] if len(sys.argv) > 2 else "krr_rbf"
    if "ridge" not in rows or LAST not in rows: continue
    agg.setdefault(enc, {})[coh] = (ntr, rows)
order = ["ridge_a1e-1","ridge_a1","ridge_a10","ridge_a100","ridge_a1e3","ridge_a1e4","ridge_a1e5","mlp_a300","mlp_a1000","mlp_identity","mlp_a1e-4","mlp_a1e-2","mlp_a1e-1","mlp_a1","mlp_a3","mlp_a10","mlp_a30","mlp_a100","mlp_h64","mlp_h64_a1e-1","mlp_h64_a10","mlp_fixed100","mlp_groupval","krr_rbf"]
for enc, d in agg.items():
    cohs = sorted(d); R = np.array([d[c][1]["ridge"][1] for c in cohs])
    print("\n%s: 已完成 %d 个队列  ridge 测试均值 %.4f   训练片数: %s" % (enc, len(cohs), R.mean(), ",".join(str(d[c][0]) for c in cohs)))
    print("  %-14s %8s %11s %9s %8s" % ("变体", "测试均值", "Δ vs ridge", "高于ridge", "训练均值"))
    for v in order:
        cc = [c for c in cohs if v in d[c][1]]
        if not cc: continue
        te = np.array([d[c][1][v][1] for c in cc]); tr = np.array([d[c][1][v][0] for c in cc])
        Rv = np.array([d[c][1]["ridge"][1] for c in cc]); wins = int((te > Rv).sum())
        print("  %-14s %8.4f %+11.4f %5d/%-3d %8.4f%s" % (v, te.mean(), te.mean() - Rv.mean(), wins, len(cc), tr.mean(), "   (仅 ≥3 片的队列)" if v == "mlp_groupval" else ""))

# 冻结：逐 (编码器, 队列) 的训练切片数与各变体的 (训练PCC, 测试PCC)
if len(sys.argv) > 3:
    out = {enc: {coh: {"n_train_sections": d[coh][0], "variants": {v: {"train_pcc": tr, "test_pcc": te} for v, (tr, te) in d[coh][1].items()}}
                 for coh in d} for enc, d in agg.items()}
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    print("\n已存", sys.argv[3])
