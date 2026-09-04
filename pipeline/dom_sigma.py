import json, numpy as np
d = json.load(open("results/ruler_fold.json"))


def eqA(pcc, sig, sc):
    ts = sorted(int(k.split("_t")[1]) for k in sc if k.startswith("A_coarse_t"))
    s = np.array([sig[str(t)] for t in ts]); p = np.array([sc[f"A_coarse_t{t}"] for t in ts])
    o = np.argsort(-p); s, p = s[o], p[o]
    if pcc >= p[0]:
        return float(s[0]), "L"
    if pcc <= p[-1]:
        return None, "R"
    j = int(np.searchsorted(-p, -pcc))
    w = (p[j-1] - pcc) / (p[j-1] - p[j])
    return float(np.exp(np.log(s[j-1]) + w * (np.log(s[j]) - np.log(s[j-1])))), ""


KEYS = ["D_domImg_k5", "D_domImg_k10", "D_domImg_k20", "D_domImg_k50",
        "D_domImg_k100", "D_domImg_k200", "R_ridgeHEST", "R_imageKNN"]
folds = list(d)
print("predictor".ljust(15) + "".join(f"|{f:^21s}" for f in folds))
print("-" * (15 + 22 * len(folds)))
OUT = {}
for K in KEYS:
    row = K.ljust(15)
    for f in folds:
        sc = d[f]["scores"]
        if K not in sc:
            row += "|" + "--".center(21); continue
        e, fl = eqA(sc[K], d[f]["sigma_um"], sc)
        OUT.setdefault(K, {})[f] = {"pcc": sc[K], "sigma_um": e, "flag": fl}
        row += "|" + f" {sc[K]:.4f}  s={'  n/a' if e is None else f'{e:5.1f}'}{fl:1s}".ljust(21)
    print(row)
print()
for f in folds:
    r = d[f]["scores"].get("R_ridgeHEST")
    print(f"{f}: ridge PCC {r:.4f}", end="  ")
    for k in (5, 10, 20, 50, 100, 200):
        v = d[f]["scores"].get(f"D_domImg_k{k}")
        if v is not None:
            print(f"k{k}:{100*v/r:5.1f}%", end=" ")
    print()
json.dump(OUT, open("results/dom_sigma.json", "w"), indent=1)
