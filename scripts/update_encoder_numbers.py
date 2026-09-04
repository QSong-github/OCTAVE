# -*- coding: utf-8 -*-
"""编码器集变化后，把正文/表 1 标题/附录里依赖编码器数的句子按 results/ 里的汇总重写。
用法：python3 scripts/update_encoder_numbers.py            （先打印将改的句子，不写）
      python3 scripts/update_encoder_numbers.py --write
每条替换都断言旧句子在 main.tex 中恰好出现一次；数字来源与 headline_numbers.py 同一口径（K=20，留一队列选 α）。"""
import json, os, re, sys, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S = json.load(open(f"{D}/results/hest_blocks_summary.json")); P = json.load(open(f"{D}/results/encoder_params.json"))
rows = [d for d in S if d["enc"] in P]; n = len(rows)
sh = np.array([d["blk_k20_sel"]["share"] for d in rows]); dm = np.array([d["blk_k20_sel"]["cohort_mean"] for d in rows])
t = np.array([d["blk_k20_sel"]["t"] for d in rows]); Pv = np.array([d["blk_k20_sel"]["P"] for d in rows])
prm = np.array([P[d["enc"]]["params"] for d in rows], float)
pos, t2, sig = int((dm > 0).sum()), int((np.abs(t) >= 2).sum()), int((Pv < 0.05).sum())
assert pos == t2 == sig, ("三个计数不一致，句子结构要改", pos, t2, sig)
med = int(round(np.median(sh))); med50 = int(round(np.median([d["blk_k50_sel"]["share"] for d in rows]))); med200 = int(round(np.median([d["blk_k200_sel"]["share"] for d in rows])))
small = int(round(np.median(sh[prm < 1e8]))); big = int(round(np.median(sh[prm > 6e8])))
if "--rho-p" in sys.argv:                      # 本机 scipy 坏时：把集群 headline_numbers.py 算出的 ρ 与 P 传进来
    i = sys.argv.index("--rho-p"); rho, pv = float(sys.argv[i + 1]), float(sys.argv[i + 2])
else:
    try:
        from scipy.stats import spearmanr; rho, pv = spearmanr(prm, sh)
    except Exception:
        sys.exit("需要 scipy 算 Spearman P；本机 scipy 坏则用 --rho-p <rho> <P>（取集群 headline_numbers.py 的输出）")
def sci(p):
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f"{m:.1f}\\times10^{{{e}}}" if p < 1e-3 else f"{p:.4f}"
words = {30: "thirty", 49: "forty-nine", 50: "fifty", 51: "fifty-one", 52: "fifty-two", 57: "fifty-seven", 58: "fifty-eight"}
tex = open(f"{D}/paper/main.tex").read()
# 当前正文里的数
cur = re.search(r"where \$(\d+)\$ frozen encoders,", tex); n0 = int(cur.group(1))
cur2 = re.search(r"trained model in \$(\d+)\$ of \$(\d+)\$ encoders, by a median of \$(\d+)\\%\$ of the model's\nscore, and the cohort-level sign test is significant in the same \$(\d+)\$\.", tex)
cur2b = re.search(r"Refining the partition raises\nthe bound further, to a median \$(\d+)\\%\$ at \$K=50\$ and \$(\d+)\\%\$ at \$K=200\$", tex)
p0, n0b, m0, p0b = map(int, cur2.groups()); m50_0, m200_0 = map(int, cur2b.groups())
cur3 = re.search(r"Spearman \$r_s = (-?[\d.]+)\$ against parameter count \(\$P = ([^$]+)\$\),\nfrom a median \$(\d+)\\%\$ below \$100\$M parameters to \$(\d+)\\%\$ above \$600\$M", tex)
rho0, p0s, small0, big0 = cur3.groups()
print(f"现有：n={n0}, 正 {p0}/{n0b}, 中位 {m0}%, K50 {m50_0}%, K200 {m200_0}%, ρ={rho0} P={p0s}, {small0}%/{big0}%")
print(f"新值：n={n}, 正 {pos}/{n}, 中位 {med-100}%, K50 {med50-100}%, K200 {med200-100}%, ρ={rho:.2f} P={sci(pv)}, {small}%/{big}%")
reps = [
 (f"benchmark of ${n0}$ frozen encoders, spanning two orders of magnitude in size, it scores above\nthe trained model in ${p0}$ of ${n0}$.",
  f"benchmark of ${n}$ frozen encoders, spanning two orders of magnitude in size, it scores above\nthe trained model in ${pos}$ of ${n}$."),
 (f"on a public benchmark of ${n0}$ frozen encoders, from an $11$M-parameter residual network to $1.1$B-parameter foundation models, it exceeds that score in ${p0}$ of ${n0}$.",
  f"on a public benchmark of ${n}$ frozen encoders, from an $11$M-parameter residual network to $1.1$B-parameter foundation models, it exceeds that score in ${pos}$ of ${n}$."),
 (f"the trained model on ${p0}$ of the ${n0}$ encoders we tested.", f"the trained model on ${pos}$ of the ${n}$ encoders we tested."),
 (f"scored under its own protocol with ${n0}$ frozen encoders.", f"scored under its own protocol with ${n}$ frozen encoders."),
 (f"We test Proposition~\\ref{{prop:proj}} on the full public benchmark, where ${n0}$ frozen encoders,",
  f"We test Proposition~\\ref{{prop:proj}} on the full public benchmark, where ${n}$ frozen encoders,"),
 (cur2.group(0), f"trained model in ${pos}$ of ${n}$ encoders, by a median of ${med-100}\\%$ of the model's\nscore, and the cohort-level sign test is significant in the same ${pos}$."),
 (cur2b.group(0), f"Refining the partition raises\nthe bound further, to a median ${med50-100}\\%$ at $K=50$ and ${med200-100}\\%$ at $K=200$"),
 (f"Over the ${n0}$ encoders the oracle's margin", f"Over the ${n}$ encoders the oracle's margin"),
 (cur3.group(0), f"Spearman $r_s = {rho:.2f}$ against parameter count ($P = {sci(pv)}$),\nfrom a median ${small}\\%$ below $100$M parameters to ${big}\\%$ above $600$M"),
 (f"$10$ cohorts, for each of ${n0}$ frozen encoders spanning", f"$10$ cohorts, for each of ${n}$ frozen encoders spanning"),
 (f"below every one of the {words[n0]} frozen encoders paired with ridge.", f"below every one of the {words[n]} frozen encoders paired with ridge."),
 (f"against the {words[n0]} frozen encoders paired with ridge. PCC is the mean over samples.", f"against the {words[n]} frozen encoders paired with ridge. PCC is the mean over samples."),
]
for a, b in reps:
    c = tex.count(a); assert c == 1, (c, a[:80])
    tex = tex.replace(a, b)
if "--write" in sys.argv:
    open(f"{D}/paper/main.tex", "w").write(tex); print(f"main.tex 已改 {len(reps)} 处；病理/通用骨干的构成句（Forty ... eight ... one）请手工核对")
else:
    print(f"（预览）将改 {len(reps)} 处；加 --write 写入")
