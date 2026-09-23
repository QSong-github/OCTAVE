# -*- coding: utf-8 -*-
"""omiclip_raw 真正取未归一化特征。

上一版只是去掉了我自己那句 F.normalize，但 CoCa 的 encode_image 默认
normalize=True，所以两个变体拿到的是同一批单位模长特征（实测行范数都是
1.0000，逐元素差 3e-08，纯浮点噪声）。必须显式传 normalize=False。
探针确认：normalize=False 时行范数 12.2112。"""
import ast
p = "/path/to/project/src/hest_embed_v2.py"
s = open(p).read()
a = '''                f = model.encode_image(xb).float()
                if kind[2] == "omiclip":
                    f = _F.normalize(f, p=2, dim=-1)      # 作者 encode_images 的做法（为检索）
                # omiclip_raw 不归一化，与其余 25 个编码器同口径'''
assert a in s, "锚点缺失"
s = s.replace(a, '''                # CoCa 的 encode_image 默认 normalize=True。作者在 encode_images 里
                # 也做 L2，那是为检索（余弦相似度）。本文用途是岭回归，因此另跑一个
                # normalize=False 的变体，与其余编码器同口径，取对该模型有利的报。
                f = model.encode_image(xb, normalize=(kind[2] == "omiclip")).float()''', 1)
ast.parse(s)
open(p, "w").write(s)
print("omiclip_raw 已改为 normalize=False")
