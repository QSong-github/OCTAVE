import ast
p = "/path/to/project/src/hest_effres_ps.py"; s = open(p).read()
if "--mlp_alpha" not in s:
    a = '    ap.add_argument("--seed", type=int, default=0, help="仅 mlp 用：初始化与早停验证集划分")\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '    ap.add_argument("--mlp_alpha", type=float, default=1e-4, help="仅 mlp 用：L2 强度；1e-4 为首轮全量所用，10 为扫描所得最优")\n', 1)
    b = '                                  alpha=1e-4, batch_size=256, learning_rate_init=1e-3,\n'
    assert s.count(b) == 1
    s = s.replace(b, '                                  alpha=args.mlp_alpha, batch_size=256, learning_rate_init=1e-3,\n', 1)
    c = '"head": args.head, "seed": args.seed,'
    assert s.count(c) == 1
    s = s.replace(c, '"head": args.head, "seed": args.seed, "mlp_alpha": args.mlp_alpha,', 1)
    ast.parse(s); open(p, "w").write(s); print("已加 --mlp_alpha（默认 1e-4，与首轮一致）")
else:
    print("--mlp_alpha 已存在")
