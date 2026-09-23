# -*- coding: utf-8 -*-
"""由已调试好的 BLEEP/HEST 适配器生成 HECLIP 版本。HECLIP 的文件头自称
   "developed based on the BLEEP"，逐项核对后只有三处实质差异，均在下方处理。"""
import os, ast
B = "/path/to/project"
s = open(os.path.join(B, "src/bleep_hest.py")).read()
sub = [
 # ① 作者代码路径与模型类
 ('BLEEP = "/path/to/project/methods/BLEEP"\nsys.path.insert(0, BLEEP)',
  'HEC = "/path/to/project/methods/HECLIP/code"\nsys.path.insert(0, HEC)'),
 ('import config as CFG                                   # BLEEP 的超参\n'
  'from models import CLIPModel                           # BLEEP 的模型, 原样',
  'from models_hvg import HECLIPModel                      # HECLIP 的模型, 原样'),
 # ② spot 侧不过投影头：HECLIP 的 forward 里 spot_embeddings = spot_features（原始表达）
 ('        I.append(model.image_projection(model.image_encoder(img)).cpu())\n'
  '        Sp.append(model.spot_projection(expr).cpu())',
  '        I.append(model.image_projection(model.image_encoder(img)).cpu())\n'
  '        # HECLIP 没有 spot 投影头：作者 forward 里 spot_embeddings = spot_features，\n'
  '        # 即原始表达本身充当参考侧嵌入。因此 projection_dim 必须等于基因数。\n'
  '        Sp.append(expr.cpu())'),
 # ③ 模型构造与超参（作者默认 lr=1e-3, wd=1e-3, epochs=15, batch=128）
 ('            model = CLIPModel(spot_embedding=len(genes)).to(dev)   # config 硬编码 3467, 显式覆盖\n'
  '            opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)',
  '            # 作者 init() 按数据集把 projection_dim 设成该数据集的基因数；这里是 50。\n'
  '            cfg = {"projection_dim": len(genes), "temperature": 1.0,\n'
  '                   "embedding_dim": 2048, "model_name": "resnet50", "dropout": 0.1}\n'
  '            model = HECLIPModel(cfg).to(dev)\n'
  '            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)'),
 ('    ap.add_argument("--epochs", type=int, default=40)\n'
  '    ap.add_argument("--batch_size", type=int, default=256)',
  '    ap.add_argument("--epochs", type=int, default=15)      # 作者 max_epochs\n'
  '    ap.add_argument("--batch_size", type=int, default=128) # 作者 batch_size\n'
  '    ap.add_argument("--lr", type=float, default=1e-3)      # 作者 lr\n'
  '    ap.add_argument("--wd", type=float, default=1e-3)      # 作者 wd'),
 ('default="results/bleep_hest.json"', 'default="results/heclip_hest.json"'),
 ("=== BLEEP: {len(out)} ", "=== HECLIP: {len(out)} "),
]
for a, b in sub:
    assert a in s, "锚点缺失:\n" + a[:120]
    s = s.replace(a, b, 1)
HDR = '''#!/usr/bin/env python
"""HECLIP 在 HEST-benchmark 上的运行 —— 模型原样用作者代码，只换数据源。

HECLIP 的文件头自称 "developed based on the BLEEP"。逐项核对后，实质差异只有三处，
本文件据此从 src/bleep_hest.py 改成：
  ① 无 spot 投影头。作者 forward 里 spot_embeddings = spot_features，参考侧嵌入
     就是原始表达，所以 projection_dim 必须等于基因数（本协议 50）。漏掉这条
     模型维度就对不上，或退化成无意义的投影。
  ② 只有 image-centric 单向损失，targets 只由 image 相似度构造（BLEEP 是双向、
     targets 取两侧相似度均值）。这一条在作者模型内部，本文件不介入。
  ③ 超参用作者默认：lr=1e-3, wd=1e-3, max_epochs=15, batch=128, temperature=1.0。

推断方式与 BLEEP 相同且已核对：作者 infer.py 的 find_matches 对两侧做 L2 归一化后
取 top-50 点积近邻，再对其真值表达做均匀平均——与 src/bleep_hest.py 的 retrieve() 一致。

协议与 bleep_hest.py 完全相同：同 HEST 官方划分、同 50 基因、同 load_adata。
"""
'''
s = HDR + s[s.index("import os, sys, glob"):]
ast.parse(s)
p = os.path.join(B, "src/heclip_hest.py")
open(p, "w").write(s)
print("已生成 %s，语法 OK" % p)
for ln in s.split("\n"):
    if ("HECLIP" in ln or "Sp.append" in ln or "projection_dim" in ln) and "\"\"\"" not in ln:
        print("   ", ln.strip()[:100])
