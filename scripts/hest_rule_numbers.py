# -*- coding: utf-8 -*-
"""HEST 上全部数字按同一汇总规则计算。
规则：每个数先逐样本得到（PCC 值、比值或差值都一样，比值与差值先在单个样本上算），
再在队列内取中位数，再跨 10 个队列取中位数，最后跨 57 个编码器取中位数。
队列计数：该队列内逐样本比值的中位数大于 1，算该队列里前者高于后者。
输入：hest_blocks_{e}、hest_blocks_z_{e}、hest_controls_{e}、hest_inductive_{e}、hest_trainonly_coord、hest_rsel_ps_{e}（模型：留一队列选 λ 的 ridge）、
hest_oracle_bands_{e} 与 hest_effres_ps_selfull_{e}（最细带 β1）、encoder_params，以及端到端方法的逐样本文件。输出 results/hest_rule_numbers.json。"""
import json, glob, os, collections, numpy as np

R = os.environ.get("OCTAVE_RESULTS", "results")
encs = sorted(f.split("hest_controls_")[1][:-5] for f in glob.glob(f"{R}/hest_controls_*.json") if "summary" not in f)
PAR = json.load(open(f"{R}/encoder_params.json"))
TOC = json.load(open(f"{R}/hest_trainonly_coord.json"))["samples"]
DISTILLED = {"litevirchow2", "h0_mini", "litefm", "litefm_l", "litefm_s", "distillpath_is16", "distillpath_ks16", "pathryoshka_b"}
GROUPS = {"uni": ["uni_v1", "uni_v2"], "virchow": ["virchow", "virchow2", "litevirchow2"],
          "hoptimus": ["hoptimus0", "hoptimus1", "h0_mini"], "phikon": ["phikon", "phikon_v2"],
          "conch": ["conch_v1", "conch_v15"], "gigapath": ["gigapath", "gigapath_flash"], "hibou": ["hibou_b", "hibou_l"],
          "kaiko": ["kaiko_vitb16", "kaiko_vitb8", "kaiko_vitl14", "kaiko_vits16", "midnight12k", "openmidnight"],
          "lunit": ["lunit_r50_bt", "lunit_r50_moco", "lunit_r50_swav", "lunit_vits16", "lunit_vits8"],
          "dinov2": ["dinov2_base", "dinov2_large", "dinov2_giant"], "dinov3": ["dinov3_vitb16", "dinov3_vith16", "dinov3_vitl16"],
          "litefm": ["litefm", "litefm_l", "litefm_s"], "distillpath": ["distillpath_is16", "distillpath_ks16"]}
METHODS = [("deepspot_hoptimus0_steps500", "deepspot_hest_hoptimus0_steps500.json"), ("deepspot_hoptimus0", "deepspot_hest_hoptimus0.json"),
           ("deepspot_uni_steps500", "deepspot_hest_uni_v1_steps500.json"), ("deepspot_uni", "deepspot_hest_uni_v1.json"),
           ("bleep", "bleep_hest.json"), ("histogene", "histogene_matched.json"), ("hggep", "hggep_hest.json"),
           ("heclip", "heclip_hest.json"), ("hist2st", "hist2st_matched_all.json"), ("thitogene", "thitogene_hest.json")]


def cohorts(sids, coh_of):
    c = collections.defaultdict(list)
    for i, s in enumerate(sids): c[coh_of[s]].append(i)
    return [c[k] for k in sorted(c)]


def val(x, coh):                       # PCC 值：样本 → 队列中位数 → 跨队列中位数
    return float(np.median([np.median(x[ix]) for ix in coh]))


def ratio(num, den, coh):              # 比值：逐样本比值 → 队列中位数 → 跨队列中位数；另给高于 1 的队列数
    cm = np.array([np.median(num[ix] / den[ix]) for ix in coh])
    return float(np.median(cm)), int((cm > 1).sum())


def diff(a, b, coh):                   # 差值：逐样本差 → 队列中位数 → 跨队列中位数
    return float(np.median([np.median(a[ix] - b[ix]) for ix in coh]))


SIDS = None; COH_OF = None; RIDGE = {}
def per_encoder(e):
    global SIDS, COH_OF
    C = json.load(open(f"{R}/hest_controls_{e}.json"))["samples"]
    B = json.load(open(f"{R}/hest_blocks_{e}.json"))["samples"]
    Z = json.load(open(f"{R}/hest_blocks_z_{e}.json"))["samples"]
    P = json.load(open(f"{R}/hest_rsel_ps_{e}.json"))["per_sample_pcc"]
    I = json.load(open(f"{R}/hest_inductive_{e}.json"))["samples"]
    O = json.load(open(f"{R}/hest_oracle_bands_{e}.json"))["samples"]
    sel = f"{R}/hest_effres_ps_selfull_{e}.json"
    F = json.load(open(sel)) if os.path.exists(sel) else json.load(open(f"{R}/hest_effres_ps_full_{e}.json"))
    sids = sorted(C); coh_of = {s: C[s]["cohort"] for s in sids}
    if SIDS is None: SIDS, COH_OF = sids, coh_of
    assert sids == SIDS and coh_of == COH_OF
    coh = cohorts(sids, coh_of)
    v = lambda f: np.array([f(s) for s in sids], float)
    model = v(lambda s: P[s]); RIDGE[e] = model
    X = {"oracle": v(lambda s: B[s]["blk_k20"]), "oracle_k50": v(lambda s: B[s]["blk_k50"]), "oracle_k200": v(lambda s: B[s]["blk_k200"]),
         "oracle_z": v(lambda s: Z[s]["blk_k20"]), "coord": v(lambda s: C[s]["oracle_coord"]), "random": v(lambda s: C[s]["oracle_random"]),
         "train": v(lambda s: C[s]["trainonly_image"]), "inductive": v(lambda s: I[s]["inductive"]), "train_coord": v(lambda s: TOC[s]["trainonly_coord"])}
    b1m = v(lambda s: F["per_sample_band_pcc"][s]["1"]); b1o = v(lambda s: O[s]["band_pcc_oracle"]["1"])
    out = dict(enc=e, n=len(sids), n_coh=len(coh), params_M=PAR[e]["params"] / 1e6,
               ridge_check=float(np.max(np.abs(v(lambda s: F["per_sample_pcc"][s]) - model))),
               model_pcc=val(model, coh), b1_model=val(b1m, coh), b1_oracle=val(b1o, coh))
    for k, x in X.items():
        out[k + "_pcc"] = val(x, coh)
        out[k + "_ratio"], out[k + "_coh"] = ratio(x, model, coh)
    out["fine_ratio"], out["fine_coh"] = ratio(b1o, b1m, coh)
    out["coord_over_oracle"], _ = ratio(X["coord"], X["oracle"], coh)
    out["random_over_oracle"], _ = ratio(X["random"], X["oracle"], coh)
    out["oracle_minus_model"] = diff(X["oracle"], model, coh)
    out["oracle_z_minus_oracle"] = out["oracle_z_pcc"] - out["oracle_pcc"]
    return out


def _rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a))
    d = collections.defaultdict(list)
    for i, x in enumerate(a): d[x].append(i)
    for x, ix in d.items():
        if len(ix) > 1:
            m = float(np.mean([r[i] for i in ix]))
            for i in ix: r[i] = m
    return r


def spearman(x, y):
    rx, ry = _rank(x), _rank(y); rx = rx - rx.mean(); ry = ry - ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def permP(x, y, n=20000, seed=0):
    rho = spearman(x, y); rng = np.random.default_rng(seed); y = np.asarray(y, float)
    hit = sum(1 for _ in range(n) if abs(spearman(x, rng.permutation(y))) >= abs(rho) - 1e-12)
    return rho, hit / n


E = [per_encoder(e) for e in encs]
by = {o["enc"]: o for o in E}
COH = cohorts(SIDS, COH_OF)
med = lambda k, es=None: float(np.median([by[e][k] for e in (es or encs)]))
rng_ = lambda k, es=None: (float(min(by[e][k] for e in (es or encs))), float(max(by[e][k] for e in (es or encs))))
S = dict(n_enc=len(E), ridge_check_max=max(o["ridge_check"] for o in E))
for k in ["model_pcc", "oracle_pcc", "train_pcc", "inductive_pcc", "coord_pcc", "random_pcc", "train_coord_pcc", "oracle_z_pcc", "b1_model", "b1_oracle"]:
    S[k] = dict(median=med(k), range=rng_(k))
for k in ["oracle", "train", "inductive", "coord", "random", "train_coord", "oracle_k50", "oracle_k200", "oracle_z", "fine"]:
    S[k + "_ratio"] = dict(median=med(k + "_ratio"), range=rng_(k + "_ratio"))
    S[k + "_coh"] = dict(median=med(k + "_coh"), range=rng_(k + "_coh"), n_all10=sum(o[k + "_coh"] == 10 for o in E),
                         n_majority=sum(o[k + "_coh"] >= 6 for o in E), not_all10=[o["enc"] for o in E if o[k + "_coh"] < 10])
S["coord_over_oracle"] = med("coord_over_oracle"); S["random_over_oracle"] = med("random_over_oracle")
S["oracle_minus_model"] = med("oracle_minus_model")
S["n_fine_oracle_ahead"] = sum(o["fine_ratio"] > 1 for o in E)
S["oracle_ratio_min"] = min(o["oracle_ratio"] for o in E)
# 参数量与 oracle 比值
FAM = {e: f for f, es in GROUPS.items() for e in es}
for e in encs: FAM.setdefault(e, e)
fam = collections.defaultdict(list)
for e in encs: fam[FAM[e]].append(e)
dfam = {f for f, es in fam.items() if all(e in DISTILLED for e in es)}
par = {e: by[e]["params_M"] for e in encs}; sh = {e: by[e]["oracle_ratio"] for e in encs}
def famvals(drop=()):
    ks = [f for f in sorted(fam) if f not in drop]
    return [float(np.median([par[e] for e in fam[f]])) for f in ks], [float(np.median([sh[e] for e in fam[f]])) for f in ks], ks
sp = {}
r, p = permP([par[e] for e in encs], [sh[e] for e in encs]); sp["enc_all"] = dict(n=len(encs), rho=r, P=p)
nd = [e for e in encs if e not in DISTILLED]
r, p = permP([par[e] for e in nd], [sh[e] for e in nd]); sp["enc_not_distilled"] = dict(n=len(nd), rho=r, P=p)
x, y, ks = famvals(); r, p = permP(x, y); sp["fam_all"] = dict(n=len(ks), rho=r, P=p)
x, y, ks = famvals(dfam); r, p = permP(x, y); sp["fam_not_distilled"] = dict(n=len(ks), rho=r, P=p)
S["spearman_params"] = sp
grp = {"all": encs, "lt100": [e for e in encs if par[e] < 100], "100to600": [e for e in encs if 100 <= par[e] <= 600],
       "gt600": [e for e in encs if par[e] > 600], "distilled": [e for e in encs if e in DISTILLED],
       "scratch_lt100": [e for e in encs if par[e] < 100 and e not in DISTILLED]}
S["groups"] = {g: dict(n=len(es), model_pcc=med("model_pcc", es), oracle_pcc=med("oracle_pcc", es), ratio=med("oracle_ratio", es),
                       ratio_range=rng_("oracle_ratio", es), coh_median=med("oracle_coh", es), n_all10=sum(by[e]["oracle_coh"] == 10 for e in es))
               for g, es in grp.items()}
# 端到端方法：同样的规则；最强编码器 = ridge 值最高的编码器
best = max(encs, key=lambda e: by[e]["model_pcc"]); S["strongest_encoder"] = best; S["strongest_pcc"] = by[best]["model_pcc"]
M = {}
for name, f in METHODS:
    D = json.load(open(f"{R}/{f}")); assert sorted(D) == SIDS, name
    x = np.array([D[s]["pcc"] for s in SIDS], float)
    pv = val(x, COH); rb, cb = ratio(x, RIDGE[best], COH)
    M[name] = dict(pcc=pv, enc_above=sum(by[e]["model_pcc"] > pv for e in encs), margin=diff(x, RIDGE[best], COH),
                   ratio_to_best=rb, coh_above_best=cb, ratio_to_oracle_median=pv / S["oracle_pcc"]["median"])
ds = [M[k] for k in M if k.startswith("deepspot")]
xs = np.max(np.vstack([np.array([json.load(open(f"{R}/{f}"))[s]["pcc"] for s in SIDS]) for k, f in METHODS if k.startswith("deepspot")]), axis=0)
M["deepspot_best_of_4"] = dict(pcc=val(xs, COH), margin=diff(xs, RIDGE[best], COH), coh_above_best=ratio(xs, RIDGE[best], COH)[1],
                               ratio_to_best=ratio(xs, RIDGE[best], COH)[0])
S["methods"] = M
json.dump(dict(summary=S, encoders=E), open(f"{R}/hest_rule_numbers.json", "w"), indent=1)

pc = lambda x: f"{100 * x:.1f}%"
print(f"编码器 {S['n_enc']}；两份 ridge 逐样本最大差 {S['ridge_check_max']:.2e}")
for k in ["model", "oracle", "train", "inductive", "coord", "random", "train_coord", "oracle_z"]:
    line = f"{k:12s} PCC {S[k + '_pcc']['median']:.3f} [{S[k + '_pcc']['range'][0]:.3f}, {S[k + '_pcc']['range'][1]:.3f}]"
    if k != "model": line += (f"  比值 {pc(S[k + '_ratio']['median'])} [{pc(S[k + '_ratio']['range'][0])}, {pc(S[k + '_ratio']['range'][1])}]  "
                              f"队列中位 {S[k + '_coh']['median']:.0f}/10  全 10/10 {S[k + '_coh']['n_all10']}  ≥6/10 {S[k + '_coh']['n_majority']}")
    print(line)
print(f"K=50 比值 {pc(S['oracle_k50_ratio']['median'])}，K=200 {pc(S['oracle_k200_ratio']['median'])}；oracle 减模型 {S['oracle_minus_model']:+.3f}；比值最小 {pc(S['oracle_ratio_min'])}")
print(f"未全 10/10：{S['oracle_coh']['not_all10']}；标准化后未全 10/10：{S['oracle_z_coh']['not_all10']}")
print(f"最细带：β1 模型 {S['b1_model']['median']:.3f} oracle {S['b1_oracle']['median']:.3f}，比值 {pc(S['fine_ratio']['median'])} [{pc(S['fine_ratio']['range'][0])}, {pc(S['fine_ratio']['range'][1])}]，"
      f"队列中位 {S['fine_coh']['median']:.0f}/10 {S['fine_coh']['range']}，oracle 领先的编码器 {S['n_fine_oracle_ahead']}，多数队列 {S['fine_coh']['n_majority']}，全 10 {S['fine_coh']['n_all10']}")
print(f"坐标/oracle {pc(S['coord_over_oracle'])}，随机/oracle {pc(S['random_over_oracle'])}")
for k, v in sp.items(): print(f"Spearman {k:18s} n={v['n']:2d} r_s={v['rho']:+.4f} P={v['P']:.4f}")
for g, v in S["groups"].items(): print(f"组 {g:13s} n={v['n']:2d} 模型 {v['model_pcc']:.3f} oracle {v['oracle_pcc']:.3f} 比值 {pc(v['ratio'])} [{pc(v['ratio_range'][0])}, {pc(v['ratio_range'][1])}] 队列 {v['coh_median']:.0f}/10 全 10 {v['n_all10']}")
print(f"最强编码器 {best} {S['strongest_pcc']:.4f}")
for k, v in M.items(): print(f"方法 {k:28s} PCC {v['pcc']:.4f}  " + " ".join(f"{kk} {vv:.4f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in v.items() if kk != 'pcc'))
om = by["openmidnight"]; others = [o for o in E if o["enc"] != "openmidnight"]
print(f"OpenMidnight：模型 {om['model_pcc']:.3f} oracle {om['oracle_pcc']:.3f} 比值 {pc(om['oracle_ratio'])} 队列 {om['oracle_coh']}；K=50 队列 {om['oracle_k50_coh']} K=200 {om['oracle_k200_coh']}；"
      f"标准化 oracle {om['oracle_z_pcc']:.3f} 比值 {pc(om['oracle_z_ratio'])} 队列 {om['oracle_z_coh']}；差值 {om['oracle_minus_model']:+.4f}")
print(f"其他编码器标准化后：oracle 最大变化 {max(abs(o['oracle_z_minus_oracle']) for o in others):.4f}，比值最大变化 {100 * max(abs(o['oracle_z_ratio'] - o['oracle_ratio']) for o in others):.1f} 点，"
      f"队列数变化 {[o['enc'] for o in others if o['oracle_z_coh'] != o['oracle_coh']]}")
