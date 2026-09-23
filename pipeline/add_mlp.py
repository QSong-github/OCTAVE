# -*- coding: utf-8 -*-
"""给 hest_effres_ps.py 加 --head {ridge,mlp} 与 --seed。
只改回归器这一处；岭回归分支一个字符不动，只是被包进 else。
MLP：sklearn MLPRegressor，256→512→50，ReLU，Adam，L2=1e-4，早停用**训练集**切出的 10%
（validation_fraction，随机划分由 seed 决定），从不碰测试集。目标按训练集均值/标准差
标准化后再拟合，预测时逆变换——PCC 对逐基因仿射不变，此举只为优化稳定，不改度量。
超参对全部编码器固定，不逐个调。"""
import ast, shutil
p = "/path/to/project/src/hest_effres_ps.py"
shutil.copy(p, p + ".bak_pre_mlp")
s = open(p).read()

a = '    ap.add_argument("--skip_sigma", action="store_true", help="只算 PCC，跳过图扩散（用于排序对照）")\n'
assert s.count(a) == 1
s = s.replace(a, a +
    '    ap.add_argument("--head", default="ridge", choices=["ridge", "mlp"],\n'
    '                    help="ridge=官方线性头；mlp=两层 MLP（256→512→50），其余管线完全相同")\n'
    '    ap.add_argument("--seed", type=int, default=0, help="仅 mlp 用：初始化与早停验证集划分")\n', 1)

b = ('            reg = Ridge(solver="lsqr", alpha=100.0 / (Ztr.shape[1] * Ytr.shape[1]),\n'
     '                        random_state=0, fit_intercept=False, max_iter=1000).fit(Ztr, Ytr)\n')
assert s.count(b) == 1
new = ('            if args.head == "mlp":\n'
       '                # 非线性头。早停验证集从训练集内切（validation_fraction），测试集不参与。\n'
       '                from sklearn.neural_network import MLPRegressor\n'
       '                mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6\n'
       '                _m = MLPRegressor(hidden_layer_sizes=(512,), activation="relu", solver="adam",\n'
       '                                  alpha=1e-4, batch_size=256, learning_rate_init=1e-3,\n'
       '                                  max_iter=200, early_stopping=True, validation_fraction=0.1,\n'
       '                                  n_iter_no_change=10, random_state=args.seed)\n'
       '                _m.fit(Ztr, (Ytr - mu) / sd)\n'
       '                print(f"  [{c} fold{k}] mlp 训练 n={Ztr.shape[0]} 轮数={_m.n_iter_} "\n'
       '                      f"最佳验证={_m.best_validation_score_:.3f}", flush=True)\n'
       '                class _Wrap:\n'
       '                    def predict(self, Z, _m=_m, mu=mu, sd=sd):\n'
       '                        return _m.predict(Z) * sd + mu\n'
       '                reg = _Wrap()\n'
       '            else:\n'
       '                reg = Ridge(solver="lsqr", alpha=100.0 / (Ztr.shape[1] * Ytr.shape[1]),\n'
       '                            random_state=0, fit_intercept=False, max_iter=1000).fit(Ztr, Ytr)\n')
s = s.replace(b, new, 1)

c = '        json.dump({"encoder": args.encoder, "target": args.target,\n'
assert s.count(c) == 1
s = s.replace(c, '        json.dump({"encoder": args.encoder, "target": args.target, "head": args.head, "seed": args.seed,\n', 1)
ast.parse(s); open(p, "w").write(s)
print("已加 --head/--seed；岭回归分支原样保留")
