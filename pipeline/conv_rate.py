#!/usr/bin/env python
"""PCC → σ 的换算率：一个分数差在物理上值多少微米。

为什么不报「放大倍数」：PCC 有界于 [0,1] 而 σ 无界，有界量的相对跨度天然小于无界量，
故「σ 跨度 / PCC 跨度」这个比值有相当一部分只是这个数学事实，审稿人一句话即可打掉。

换算率没有这个问题：它不比较两个跨度，而是回答一个可解释的问题——
**报告分数每变化 0.01，等价分辨率变化百分之多少。**
拟合 log σ ~ PCC，斜率 b 给出 d(lnσ)/d(PCC)；每 +0.01 PCC 对应 σ 变化 (exp(0.01·b)−1)。

每条线的协议/基准不同，绝对值不可跨线并列，故分别拟合、分别报告。
"""
import json, glob, os
import numpy as np

def fit(name, rows, note=""):
    rows = [(n, p, s) for n, p, s in rows
            if all(isinstance(v, (int, float)) and v == v for v in (p, s)) and s > 0]
    if len(rows) < 4:
        print(f"\n=== {name} ===  样本不足 ({len(rows)})"); return None
    p = np.array([r[1] for r in rows]); s = np.log(np.array([r[2] for r in rows]))
    b, a = np.polyfit(p, s, 1)
    pred = a + b * p
    ss_res = ((s - pred) ** 2).sum(); ss_tot = ((s - s.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    per01 = (np.exp(0.01 * b) - 1) * 100
    lo, hi = min(rows, key=lambda r: r[1]), max(rows, key=lambda r: r[1])
    dp = hi[1] - lo[1]
    dr = lo[2] / hi[2]
    print(f"\n=== {name} ===  n={len(rows)}  {note}")
    print(f"  PCC 范围 {lo[1]:.4f}（{lo[0]}）→ {hi[1]:.4f}（{hi[0]}）  差 {dp:+.4f}（{100*dp/lo[1]:+.1f}%）")
    print(f"  对应 σ  {lo[2]:.1f} → {hi[2]:.1f} µm  改善 {dr:.2f}×")
    print(f"  换算率：PCC 每 +0.01 ⇒ σ 变化 {per01:+.1f}%     拟合 R² = {r2:.3f}")
    print(f"  一句话：{100*dp/lo[1]:.0f}% 的分数差 = {dr:.1f} 倍的分辨率差")
    return {"name": name, "n": len(rows), "slope_lnsigma_per_pcc": float(b),
            "pct_per_0.01pcc": float(per01), "r2": float(r2),
            "pcc_lo": lo[1], "pcc_hi": hi[1], "sigma_lo": lo[2], "sigma_hi": hi[2]}

out = []

# 深度线 25 塔（估计量 A，新基准）
dp = []
for f in sorted(glob.glob("results/legacy8k/tower_*.json")):
    n = os.path.basename(f)[len("tower_"):-5]
    if any(x in n for x in ("grid", "ctx")): continue
    d = json.load(open(f))
    sl = [k for k in d if k.startswith("Visium")]
    sg = [d[k]["eq_sigma"] for k in sl if d[k].get("flag") == "ok"]
    pc = [d[k]["pcc"] for k in sl if d[k].get("flag") == "ok"]
    if sg: dp.append((n, float(np.mean(pc)), float(np.mean(sg))))
r = fit("深度线 · 25 个图像编码器（跨片）", dp, "同一下游回归器，仅换编码器")
if r: out.append(r)

# 方法级（片内 / 跨片）
for tag, f in [("方法 · 片内块 CV", "results/method_rank_within.json"),
               ("方法 · 跨片留一", "results/method_rank_cross.json")]:
    if not os.path.exists(f): continue
    d = json.load(open(f))
    rows = [(x[0], x[1], x[2]) for x in d.get("rows", [])]
    r = fit(f"{tag}（旧 200 基因基准）", rows, "含 iStar 官方实现")
    if r: out.append(r)

# 多尺度分箱：同一模型，只改输出栅格 —— 方向相反，单独说明
ms = []
for f in glob.glob("results/xenium/*.json"):
    d = json.load(open(f))
    if d.get("eq_flag") == "ok": ms.append((d["name"] + "@16", d["pcc"], d["eq_sigma"]))
for f in glob.glob("results/xenium_multi/*.json"):
    d = json.load(open(f))
    if d.get("eq_flag") == "ok": ms.append((d["name"], d["pcc"], d["eq_sigma"]))
if ms:
    p = np.array([r[1] for r in ms]); s = np.array([r[2] for r in ms])
    b, _ = np.polyfit(p, np.log(s), 1)
    print(f"\n=== 对照 · 多尺度分箱（同一模型，只改输出栅格）===  n={len(ms)}")
    print(f"  斜率 d(lnσ)/d(PCC) = {b:+.2f}   —— 与上列各线**符号相反**")
    print(f"  即：编码器/方法变好时 PCC↑ σ↓；把栅格调粗时 PCC↑ 而 σ↑。")
    print(f"  ⇒ 同一指标在两种情形下指向相反方向，这是排行榜的真实漏洞。")
    out.append({"name": "多尺度分箱（对照）", "n": len(ms), "slope_lnsigma_per_pcc": float(b)})

json.dump(out, open("results/conversion_rate.json", "w"), indent=2, ensure_ascii=False)
print("\n已存 results/conversion_rate.json")
