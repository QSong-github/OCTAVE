# -*- coding: utf-8 -*-
"""由已调试好的 HGGEP/HEST 适配器（含 --fold）生成 THItoGene 的对应版本。
THItoGene 与 Hist2ST 同源：forward(patch, center, adj)、calcADJ(k=4)、training_step 解包 (patch, center, exp, adj)。
三处不同，逐一替换：① forward 只返回 pred；② 损失为纯 MSE（无 ZINB、无自蒸馏）；③ configure_optimizers 为 Adam，无 StepLR。
超参取作者 train.py 的 HER2+ 设置：lr=1e-5, route_dim=64, caps=20, heads=[16,8], n_layers=4；邻接用作者 dataset.py 的 calcADJ(k=4, pruneTag='NA')。"""
import os, re, ast
B = "/path/to/systema4ST"
s = open(os.path.join(B, "src/hggep_hest.py")).read()
def rep(a, b, cnt=1):
    global s
    c = s.count(a); assert c == cnt, "锚点 %r 命中 %d（期望 %d）" % (a[:60], c, cnt); s = s.replace(a, b)
rep("/path/to/systema4ST/methods/HGGEP", "/path/to/systema4ST/methods/THItoGene")
rep("from HGGEP import HGGEP", "from vis_model import THItoGene")
rep("from NB_module import ZINB_loss                          # 作者的 ZINB 损失，原样\n", "")
# 构造：整段替换
m = re.search(r"model = HGGEP\(n_genes=len\(genes\).*?\)\.to\(dev\)", s, re.S); assert m
s = s[:m.start()] + ("model = THItoGene(n_genes=len(genes), n_pos=npos, learning_rate=a_.lr,\n"
                     "                              route_dim=64, caps=20, heads=[16, 8], n_layers=4).to(dev)") + s[m.end():]
# 调度：作者无 StepLR
rep("sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.9)", "sch = None   # THItoGene 的 configure_optimizers 只有 Adam，无调度")
rep("                sch.step()\n", "                if sch is not None: sch.step()\n")
# 前向与损失
rep("pred, extra, h = model(pt, ctt, adt)", "pred = model(pt, ctt, adt)")
m = re.search(r"\n\s*# 作者 training_step 的完整损失.*?model\.distillation\(model\.aug\(pt, ctt, adt\)\), pred\)\n", s, re.S); assert m, "损失块锚点缺失"
s = s[:m.start()] + "\n" + s[m.end():]
rep("adj.to(dev))[0].squeeze(0).cpu().numpy()", "adj.to(dev)).squeeze(0).cpu().numpy()")
# 邻接：作者默认 pruneTag='NA'
rep('calcADJ(ctr.astype(np.float32), k=4, pruneTag="Grid")', 'calcADJ(ctr.astype(np.float32), k=4, pruneTag="NA")')
rep('default="results/hggep_hest.json"', 'default="results/thitogene_hest.json"')
rep("=== HGGEP: {len(res)} ", "=== THItoGene: {len(res)} ")
s = re.sub(r'\A#!/usr/bin/env python\n""".*?"""\n', '#!/usr/bin/env python\n"""THItoGene（Brief Bioinform 2024）在 HEST-benchmark 上的运行 —— 模型原样用作者代码，只换数据源。\n由 src/hggep_hest.py 生成（mk_thitogene.py），替换点见生成器注释。"""\n', s, count=1, flags=re.S)
s = "\n".join(l for l in s.split("\n") if not re.search(r"import HGGEP as _HGmod|_HGmod\.|hg_fast|build_adj_hypergraph", l))  # HGGEP 专用补丁
ast.parse(s)
open(os.path.join(B, "src/thitogene_hest.py"), "w").write(s)
print("已生成 src/thitogene_hest.py；残留检查 HGGEP=%d ZINB=%d extra=%d" % (s.count("HGGEP"), s.count("ZINB"), s.count("extra")))
