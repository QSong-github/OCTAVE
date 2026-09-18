# -*- coding: utf-8 -*-
"""接入 Phaet 与 Mascaret（2026-06，wearewaiv）。
config.json 里 auto_map 指向仓库自带的 modeling 代码，走 trust_remote_code 即可;
pixel_mean/std 为 ImageNet 统计量、image_size=224、DINOv2 架构（取 [CLS]），
与脚本里既有的 "hf" 分支完全一致，因此只需登记仓库名。
Phaet 是 Phikon-v2 的微调版，Mascaret 是 Midnight-12k 的微调版。"""
import ast
p = "/path/to/systema4ST/src/hest_embed_v2.py"
s = open(p).read()
assert "phaet" not in s
a = '"dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m"}'
assert a in s
s = s.replace(a, '"dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",\n'
                 '            "phaet": "wearewaiv/phaet", "mascaret": "wearewaiv/mascaret"}', 1)
ast.parse(s); open(p, "w").write(s)
print("Phaet / Mascaret 已登记")
