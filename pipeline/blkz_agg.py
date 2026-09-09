# -*- coding: utf-8 -*-
"""z-score 稳健性：对每个编码器比较 标准配方 vs PCA 前各维 z-score 的域预言机（K=20/50/200），分母同为留一队列选 α 的 ridge。
输出 results/hest_blocks_z_summary.json 与一张对照表。"""
import json, os, glob, numpy as np
R = "/path/to/systema4ST/results"
def cohort_stats(orc, mod, coh):
    by = {}
    for s in orc:
        if s in mod: by.setdefault(coh[s], []).append(orc[s] - mod[s])
    med = np.array([np.median(v) for v in by.values()]); won = int((med > 0).sum())
    return float(med.mean()), float(med.std(ddof=1) / np.sqrt(len(med))), won, len(med)
out = []
for f in sorted(glob.glob(f"{R}/hest_blocks_z_*.json")):
    e = os.path.basename(f)[len("hest_blocks_z_"):-5]
    if e == "summary": continue                      # 自己的汇总文件也匹配这个 glob
    Z = json.load(open(f))["samples"]; S = json.load(open(f"{R}/hest_blocks_{e}.json"))["samples"]
    mod = json.load(open(f"{R}/hest_rsel_ps_{e}.json"))["per_sample_pcc"]
    coh = {s: v["cohort"] for s, v in S.items()}
    rec = {"enc": e}
    for k in (20, 50, 200):
        o0 = {s: v[f"blk_k{k}"] for s, v in S.items()}; oz = {s: v[f"blk_k{k}"] for s, v in Z.items()}
        m = np.mean([mod[s] for s in o0 if s in mod])
        b0, bz = np.mean([o0[s] for s in o0 if s in mod]), np.mean([oz[s] for s in oz if s in mod])
        rec[f"k{k}"] = {"model": float(m), "orc": float(b0), "orc_z": float(bz), "share": float(100 * b0 / m), "share_z": float(100 * bz / m),
                        "std": cohort_stats(o0, mod, coh), "zs": cohort_stats(oz, mod, coh)}
    out.append(rec)
json.dump(out, open(f"{R}/hest_blocks_z_summary.json", "w"), indent=1)
print("%-15s | %6s %6s %6s %7s | %6s %6s %6s %7s" % ("encoder", "orc20", "orcZ20", "Δ", "won→wonZ", "orc200", "orcZ200", "Δ", "won→wonZ"))
d20 = []
for r in sorted(out, key=lambda r: r["k20"]["orc_z"] - r["k20"]["orc"]):
    a, b = r["k20"], r["k200"]; d20.append(a["share_z"] - a["share"])
    print("%-15s | %6.3f %6.3f %+6.3f %3d→%-3d | %6.3f %6.3f %+6.3f %3d→%-3d" % (r["enc"], a["orc"], a["orc_z"], a["orc_z"] - a["orc"], a["std"][2], a["zs"][2],
                                                                        b["orc"], b["orc_z"], b["orc_z"] - b["orc"], b["std"][2], b["zs"][2]))
d20 = np.array(d20); shz = np.array([r["k20"]["share_z"] for r in out]); pos = sum(r["k20"]["zs"][0] > 0 for r in out); won = sum(r["k20"]["zs"][2] >= 8 for r in out)
print(f"\nK=20 z-score 后：占比中位 {np.median(shz):.0f}% ({shz.min():.0f}–{shz.max():.0f})；配对差为正 {pos}/{len(out)}；占比变化中位 {np.median(np.abs(d20)):.1f} 个百分点，最大 {np.abs(d20).max():.1f}（{out[int(np.abs(d20).argmax())]['enc']}）")
