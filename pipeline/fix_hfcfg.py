# -*- coding: utf-8 -*-
"""hf 分支改为读取模型 config 自带的 pixel_mean/pixel_std。
Waiv 的两个模型把各自的统计量写在 config 里，且两者并不相同：
Phaet 用 ImageNet（与旧行为一致，改动对它是恒等），
Mascaret 继承 Midnight 的 0.5/0.5/0.5 —— 套 ImageNet 会静默压低它的表现。
与 trident / bioptimus / CLIP 各分支同理：绝不默认套 ImageNet。"""
import ast
p = "/path/to/systema4ST/src/hest_embed_v2.py"
s = open(p).read()

a = '    m = AutoModel.from_pretrained(HF_REPOS[name], trust_remote_code=True).eval().to(dev)\n    return m, "hf"'
assert s.count(a) == 1
s = s.replace(a,
    '    m = AutoModel.from_pretrained(HF_REPOS[name], trust_remote_code=True).eval().to(dev)\n'
    '    pm, ps = getattr(m.config, "pixel_mean", None), getattr(m.config, "pixel_std", None)\n'
    '    if pm is None or ps is None:\n'
    '        return m, "hf"\n'
    '    print("  %s 自带统计量 mean=%s std=%s" % (name, pm, ps), flush=True)\n'
    '    v = lambda t: torch.tensor(t, dtype=torch.float32, device=dev).view(1, 3, 1, 1)\n'
    '    return m, ("hfcfg", v(pm), v(ps))', 1)

b = '            elif kind == "hf":'
assert s.count(b) == 1
s = s.replace(b,
    '            elif isinstance(kind, tuple) and kind[0] == "hfcfg":\n'
    '                o = model(pixel_values=(x - kind[1]) / kind[2])\n'
    '                f = o.last_hidden_state[:, 0] if hasattr(o, "last_hidden_state") else o.pooler_output\n'
    '            elif kind == "hf":', 1)

ast.parse(s); open(p, "w").write(s)
print("hf 分支已改为读 config 统计量")
