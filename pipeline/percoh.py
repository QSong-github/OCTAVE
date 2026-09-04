import re, sys, numpy as np
L = open(sys.argv[1]).read()
print("MLP(α=1e-4) 与 KRR 相对 ridge 的亏损，按训练切片数排列（fold0）")
for enc in ("uni_v2", "hoptimus1", "ciga"):
    rows = []
    for b in re.split(r"\n(?=═══ )", L):
        m = re.match(r"═══ %s\s+(\S+)/fold0.*?训练 (\d+) 片 n=(\d+)" % enc, b)
        if not m: continue
        r = {x.group(1): float(x.group(3)) for x in re.finditer(r"^\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)", b, re.M)}
        if "krr_rbf" not in r: continue
        rows.append((int(m.group(2)), int(m.group(3)), m.group(1), r["ridge"], r["mlp_a1e-4"] - r["ridge"], r["mlp_a10"] - r["ridge"], r["krr_rbf"] - r["ridge"]))
    if not rows: continue
    rows.sort()
    print("\n%s   %-10s %5s %6s %8s %10s %9s %9s" % (enc, "队列", "切片", "n", "ridge", "Δmlp1e-4", "Δmlp10", "Δkrr"))
    for n, nn, c, rg, d1, d10, dk in rows:
        print("          %-10s %5d %6d %8.4f %+10.4f %+9.4f %+9.4f" % (c, n, nn, rg, d1, d10, dk))
    from scipy.stats import spearmanr
    ns = [r[0] for r in rows]
    for lab, i in (("Δmlp1e-4", 4), ("Δmlp10", 5), ("Δkrr", 6)):
        rho = spearmanr(ns, [r[i] for r in rows]).statistic
        print("          Spearman(训练切片数, %s) = %+.2f" % (lab, rho))
