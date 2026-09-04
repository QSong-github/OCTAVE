"""指标是「不够精确」还是「在误导选择」？
测 PCC 排序与 σ 排序是否一致。一致 ⇒ 只是压缩；不一致 ⇒ 按 PCC 选模型会选错。"""
import json, glob, os
import numpy as np
from scipy.stats import spearmanr, kendalltau

def report(name, rows):
    rows = [r for r in rows if all(isinstance(x,(int,float)) and x==x for x in r[1:])]
    if len(rows) < 4: print(f"{name}: 样本不足"); return
    pc = np.array([r[1] for r in rows]); sg = np.array([r[2] for r in rows])
    rho = spearmanr(pc, -sg).statistic       # PCC 越高越好，σ 越小越好
    tau = kendalltau(pc, -sg).statistic
    by_p = sorted(rows, key=lambda r: -r[1])
    by_s = sorted(rows, key=lambda r: r[2])
    rp = {r[0]: i+1 for i, r in enumerate(by_p)}
    rs = {r[0]: i+1 for i, r in enumerate(by_s)}
    moves = sorted(((abs(rs[n]-rp[n]), n, rp[n], rs[n]) for n,_,_ in rows), reverse=True)
    print(f"\n=== {name}  n={len(rows)} ===")
    print(f"PCC 排序 vs σ 排序：Spearman ρ = {rho:.3f}   Kendall τ = {tau:.3f}")
    print(f"PCC 前三: {[r[0] for r in by_p[:3]]}")
    print(f"σ   前三: {[r[0] for r in by_s[:3]]}")
    print(f"最大名次变动 {moves[0][0]} 位：{moves[0][1]} (PCC 第{moves[0][2]} → σ 第{moves[0][3]})")
    big = [m for m in moves if m[0] >= 3]
    print(f"变动 ≥3 位的塔: {len(big)}/{len(rows)}" + (f"  例: " + ", ".join(f"{n}({a}→{b})" for _,n,a,b in big[:4]) if big else ""))
    top = by_p[0][0]
    print(f"按 PCC 选出的第一名 {top}，在 σ 上排第 {rs[top]}")

# 深度线 25 塔（估计量 A）
dp = []
for f in sorted(glob.glob("results/legacy8k/tower_*.json")):
    n = os.path.basename(f)[len("tower_"):-5]
    if any(x in n for x in ("grid", "ctx")): continue
    d = json.load(open(f))
    sl = [k for k in d if k.startswith("Visium")]
    sg = [d[k]["eq_sigma"] for k in sl if d[k].get("flag") == "ok"]
    pc = [d[k]["pcc"] for k in sl if d[k].get("flag") == "ok"]
    if sg: dp.append((n, float(np.mean(pc)), float(np.mean(sg))))
report("深度线 25 塔（估计量 A 阶梯匹配）", dp)

# 广度线 15 塔，两个估计量
brA, brB = [], []
for f in sorted(glob.glob("results/hest_effres2_*.json")):
    d = json.load(open(f)); p = (d.get("pcc_check") or {}).get("official")
    a = (d.get("eq_sigma_ladder") or {}).get("median_um")
    b = (d.get("eff_res_um") or {}).get("0.2")
    if p and a: brA.append((d["encoder"], p, a))
    if p and b: brB.append((d["encoder"], p, b))
report("广度线 15 塔（估计量 A）", brA)
report("广度线 15 塔（估计量 B τ=0.2）", brB)
