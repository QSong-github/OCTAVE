#!/usr/bin/env python
"""Source Data：六张主图每个面板一份 CSV，与 make_figs.py 读同一批 JSON、同一套过滤规则。

投稿要求图中每个数据点可追溯。这里不重算任何统计量的定义——
定义只写在 make_figs.py 里一次，本脚本复用它，避免两处漂移。
"""
import csv, glob, json, os, re
import numpy as np
import make_figs as F   # 复用 BINS / COH / jload / PROTO_EN，保证口径一致

OUT = "source_data"
RES = F.RES


def specimen(name):
    """15 个切片区域来自 7 个独立样本：乳腺 S1–S4 各含 Top/Mid/Bot 三个子区域。

    所有「k/15」计数与符号检验若按区域计会构成伪重复，故 source data 必须
    带上样本列，让审稿人能自行按样本层级重算。
    """
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", name)
    return m.group(1) if m else name


def dump(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  → {OUT}/{name}.csv  ({len(rows)} 行)", flush=True)


def sd_fig1():
    d = F.jload(f"{RES}/effres_hibou_l_hvg50_t2048.json")
    if not d:
        return
    sig = {int(k): v for k, v in d["sigma_um"].items()}
    tp = {int(k): v for k, v in d["truth_power"].items()}
    cps = sorted(sig)
    dump("Fig1a_sigma_calibration", ["diffusion_steps_t", "measured_sigma_um"],
         [[c, sig[c]] for c in cps])
    rows = []
    for m in sorted(d["methods"], key=lambda m: -m["pcc"]):
        cur = {int(k): v for k, v in m["curve"].items()}
        for c in cps:
            rows.append([m["method"], c, sig[c], cur[c]])
    dump("Fig1b_band_pcc", ["method", "t", "band_centre_sigma_um", "band_pcc"], rows)
    dump("Fig1b_effective_resolution",
         ["method", "aggregate_pcc"] + [f"ER_tau_{t}_um" for t in ("0.1", "0.2", "0.3")],
         [[m["method"], m["pcc"]] + [m["eff_res_um"][t] for t in ("0.1", "0.2", "0.3")]
          for m in sorted(d["methods"], key=lambda m: -m["pcc"])])
    tot = sum(tp.values())
    best = sorted(d["methods"], key=lambda m: -m["pcc"])[0]
    bc = {int(k): v for k, v in best["curve"].items()}
    dump("Fig1c_band_power",
         ["t", "band_centre_sigma_um", "raw_power_fraction",
          "share_of_band_passed_power_pct", f"band_pcc_{best['method']}"],
         [[c, sig[c], tp[c], 100 * tp[c] / tot, bc[c]] for c in cps])


def sd_fig2():
    rows = []
    for f in sorted(glob.glob(f"{RES}/legacy8k/tower_*.json")):
        n = os.path.basename(f)[len("tower_"):-5]
        if any(x in n for x in ("grid", "ctx")):
            continue
        d = F.jload(f)
        if not d:
            continue
        sl = [k for k in d if k.startswith("Visium")]
        s = [d[k]["eq_sigma"] for k in sl if d[k].get("flag") == "ok"]
        p = [d[k]["pcc"] for k in sl if d[k].get("flag") == "ok"]
        if s:
            rows.append([n, len(s), float(np.mean(p)), float(np.mean(s))])
    dump("Fig2a_encoder_pcc_sigma",
         ["encoder", "n_sections_ok", "mean_pcc", "mean_equivalent_sigma_um"], rows)
    conv = F.jload(f"{RES}/conversion_rate.json") or []
    dump("Fig2b_exchange_rate",
         ["protocol_en", "protocol_original", "pct_sigma_per_0.01_pcc"],
         [[F.PROTO_EN.get(c["name"], c["name"]).replace("\n", " "), c["name"],
           c.get("pct_per_0.01pcc")] for c in conv])


def sd_fig3():
    rows = []
    for f in sorted(glob.glob(f"{RES}/downstream_*.json")):
        n = os.path.basename(f)[len("downstream_"):-5]
        d = F.jload(f)
        if not d or not all(str(b) in d.get("scales", {}) for b in F.BINS):
            continue
        for b in F.BINS:
            p = f"{RES}/xenium_multi/{n}_bin{b}.json" if b != 16 else f"{RES}/xenium/{n}.json"
            q = F.jload(p)
            s = d["scales"][str(b)]
            rows.append([n, specimen(n), b, q["pcc"] if q else "", s["pcc"],
                         (q or {}).get("eq_sigma", ""), (q or {}).get("eq_flag", ""),
                         s.get("n_ok", ""), d.get("n_bin16", ""),
                         round(s["n_ok"] / d["n_bin16"], 6)
                         if s.get("n_ok") and d.get("n_bin16") else ""])
    dump("Fig3_evaluation_grid",
         ["section", "specimen", "prediction_bin_um", "pcc_grid_follows",
          "pcc_grid_fixed_16um", "equivalent_sigma_um", "sigma_flag",
          "n_bins_scored", "n_bins_total", "coverage_fraction"], rows)


def sd_fig4():
    def allcfg(pat, label):
        out = []
        for f in glob.glob(pat):
            cfg = os.path.basename(f)[:-5]
            for sid, v in (F.jload(f) or {}).items():
                if isinstance(v, dict) and isinstance(v.get("pcc"), (int, float)):
                    out.append([label, cfg, sid, v.get("cohort", "?"), v["pcc"]])
        return out
    rows = allcfg(f"{RES}/histogene_*.json", "HisToGene") + \
           allcfg(f"{RES}/h2st2_[mp]*.json", "Hist2ST")
    rd = F.jload(f"{RES}/hest_reported_pcc_phikon_v2.json") or {}
    rows += [["Ridge + phikon-v2", "official_protocol", k, v["cohort"], v["pcc"]]
             for k, v in rd.items() if "pcc" in v]
    dump("Fig4_method_comparison",
         ["method", "config", "sample_id", "cohort", "per_gene_pcc"], rows)


def sd_fig5():
    GR = [16, 8, 4, 2]
    rows = []
    for g in GR:
        for f in glob.glob(f"{RES}/proto_g{g}/*.json"):
            j = F.jload(f)
            if j:
                nm = j["name"].replace(f"_g{g}", "")
                rows.append([nm, specimen(nm), f"{g}x{g}", j["pcc"],
                             j.get("eq_sigma", ""), j.get("eq_flag", "")])
    dump("Fig5_protocol_sensitivity",
         ["section", "specimen", "block_grid", "pcc", "equivalent_sigma_um",
          "sigma_flag"], rows)


def sd_fig6():
    rows, rec = [], []
    SZ = ["<50µm", "50-100µm", "100-200µm", "≥200µm"]
    for f in glob.glob(f"{RES}/downstream_*.json"):
        d = F.jload(f)
        if not d or not all(str(b) in d.get("scales", {}) for b in F.BINS):
            continue
        for b in F.BINS:
            s = d["scales"][str(b)]
            for k in ("ari", "edge_ratio"):
                rows.append([d["name"], specimen(d["name"]), b, k, s.get(k, ""),
                             s.get("n_ok", "")])
    for f in glob.glob(f"{RES}/downstream2_*.json"):
        d = F.jload(f)
        if not d or not all(str(b) in d.get("scales", {}) for b in F.BINS):
            continue
        for b in F.BINS:
            s = d["scales"][str(b)]
            for k in ("coloc_preserve", "hotspot_jaccard", "boundary_shift_um"):
                rows.append([d["name"], specimen(d["name"]), b, k, s.get(k, ""),
                             s.get("n_ok", "")])
            hr = s.get("hotspot_recall_by_size") or {}
            for k in SZ:
                if k in hr:
                    rec.append([d["name"], specimen(d["name"]), b, k, hr[k][0], hr[k][1]])
    dump("Fig6a-e_downstream_metrics",
         ["section", "specimen", "prediction_bin_um", "metric", "value",
          "n_bins_scored"], rows)
    dump("Fig6f_hotspot_recall_by_size",
         ["section", "specimen", "prediction_bin_um", "size_class", "recall",
          "n_hotspots"], rec)


if __name__ == "__main__":
    for fn in (sd_fig1, sd_fig2, sd_fig3, sd_fig4, sd_fig5, sd_fig6):
        print(fn.__name__, flush=True)
        fn()
    print("完成")
