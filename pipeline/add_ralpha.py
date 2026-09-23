import ast
p = "/path/to/project/src/hest_effres_ps.py"; s = open(p).read()
if "--ridge_alpha" not in s:
    a = '    ap.add_argument("--mlp_alpha", type=float, default=1e-4'
    assert s.count(a) == 1
    s = s.replace(a, '    ap.add_argument("--ridge_alpha", type=float, default=None, help="仅 ridge 用：L2 强度；缺省为官方公式 100/(D·G)")\n' + a, 1)
    b = '                reg = Ridge(solver="lsqr", alpha=100.0 / (Ztr.shape[1] * Ytr.shape[1]),\n'
    assert s.count(b) == 1
    s = s.replace(b, '                reg = Ridge(solver="lsqr", alpha=(args.ridge_alpha if args.ridge_alpha is not None else 100.0 / (Ztr.shape[1] * Ytr.shape[1])),\n', 1)
    c = '"mlp_alpha": args.mlp_alpha,'
    assert s.count(c) == 1
    s = s.replace(c, c + ' "ridge_alpha": args.ridge_alpha,', 1)
    ast.parse(s); open(p, "w").write(s); print("已加 --ridge_alpha（缺省 = 官方公式）")
else:
    print("--ridge_alpha 已存在")
