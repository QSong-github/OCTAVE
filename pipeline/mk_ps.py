src = open("src/hest_effres.py").read()

a = "    curve, sig_all, meta, own_pcc, eq_ladder = {}, {}, {}, {}, []"
assert a in src
src = src.replace(a, a + "\n    ps_band, ps_sigma = {}, {}", 1)

b = ("                    r = per_gene_pcc(band[:, G:], band[:, :G])\n"
     "                    curve.setdefault(cp, []).append(float(np.nanmean(r)))")
assert b in src
src = src.replace(b, b + (
    "\n                    ps_band.setdefault(s, {})[str(cp)] = float(np.nanmean(r))"
    "\n                    ps_sigma.setdefault(s, {})[str(cp)] = float(sig[cp])"), 1)

c = '    res = {"encoder": args.encoder, "pcc_check": check, "eq_sigma_ladder": eq_stats,'
assert c in src
src = src.replace(c, (
    '    res = {"per_sample_band_pcc": ps_band, "per_sample_sigma_um": ps_sigma,\n'
    '           "per_sample_pcc": own_pcc,\n'
    '           "encoder": args.encoder, "pcc_check": check, "eq_sigma_ladder": eq_stats,'), 1)

open("src/hest_effres_ps.py", "w").write(src)
import ast; ast.parse(src); print("src/hest_effres_ps.py 已生成")
