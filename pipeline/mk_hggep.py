# -*- coding: utf-8 -*-
"""由已调试好的 Hist2ST/HEST 适配器生成 HGGEP 的对应版本，只替换模型与路径。"""
import os
B = "/path/to/project"
s = open(os.path.join(B, "src/hist2st_hest.py")).read()
sub = [
    ("/path/to/project/methods/Hist2ST",
     "/path/to/project/methods/HGGEP"),
    ("from HIST2ST import Hist2ST", "from HGGEP import HGGEP"),
    ("model = Hist2ST(n_genes=", "model = HGGEP(n_genes="),
    ('default="results/hist2st_hest.json"', 'default="results/hggep_hest.json"'),
    ("=== Hist2ST: {len(res)} ", "=== HGGEP: {len(res)} "),
]
for a, b in sub:
    assert a in s, "锚点缺失: " + a
    s = s.replace(a, b)
HDR = '''#!/usr/bin/env python
"""HGGEP 在 HEST-benchmark 上的运行 —— 模型原样用作者代码，只换数据源。

HGGEP 建在 Hist2ST 的代码库上。已逐项核对：构造签名相同；
forward(patches, centers, adj) -> (x, extra, h) 相同；training_step 的损失组合
（mse + zinb*ZINB + lamb*自蒸馏）与 Hist2ST 逐行等价，只少了 logging；
configure_optimizers 同为 StepLR(step_size=50, gamma=0.9)。

因此本文件由 src/hist2st_hest.py 改成，只替换模型类与作者代码路径，
训练循环一字未动——包括 §32 查出的三处修正：孤立点补边、StepLR、完整损失。
calcADJ 用 HGGEP 自己那份（与 Hist2ST 的差别只是多一个默认关闭的 all_conn）。
"""
'''
s = HDR + s[s.index("import os, sys, glob"):]
p = os.path.join(B, "src/hggep_hest.py")
open(p, "w").write(s)
import ast
ast.parse(s)
print("已生成 %s，语法 OK" % p)
for ln in s.split("\n"):
    if "HGGEP" in ln and not ln.strip().startswith("#"):
        print("   ", ln.strip()[:96])
