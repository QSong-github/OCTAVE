# -*- coding: utf-8 -*-
"""接入 MUSK（Ruijiang Li 组，Nature 2025）。加载与前向全部照作者 README：
   384x384、Inception mean/std、fp16、
   with_head=False, out_norm=False, ms_aug=True, return_global=True
   （作者指明这组用于线性探针/MIL，正是本文岭回归的用法）。
   该仓库没有 config.json 也没有 timm 配置，所以 AutoModel 与 timm 的通用路径都
   认不出来 —— 必须用作者的 musk 包注册模型再手动载权重。"""
import ast
p = "/path/to/systema4ST/src/hest_embed_v2.py"
s = open(p).read()
assert "musk" not in s
s = s.replace('NEW_ENC = list(NEW_TIMM) + list(NEW_CLIP) + list(NEW_HF2) + ["omiclip", "omiclip_raw"]',
              'NEW_ENC = list(NEW_TIMM) + list(NEW_CLIP) + list(NEW_HF2) + ["omiclip", "omiclip_raw", "musk"]', 1)
s = s.replace('TIMM_SIZE = {"kaiko_vitl14": 518}', 'TIMM_SIZE = {"kaiko_vitl14": 518, "musk": 384}', 1)
c = '    if name in ("omiclip", "omiclip_raw"):'
assert c in s
s = s.replace(c, '''    if name == "musk":
        import sys as _sys, torchvision as _tv
        _sys.path.insert(0, "/path/to/methods/MUSK")
        from musk import utils as _mu, modeling as _mm      # noqa: F401  注册 timm 模型
        from timm.models import create_model as _cm
        from timm.data.constants import IMAGENET_INCEPTION_MEAN as _M, IMAGENET_INCEPTION_STD as _S
        mk = _cm("musk_large_patch16_384")
        _mu.load_model_and_may_interpolate("hf_hub:xiangjx/musk", mk, "model|module", "")
        mk.to(device=dev, dtype=torch.float16).eval()
        tf = _tv.transforms.Compose([
            _tv.transforms.Resize(384, interpolation=3, antialias=True),
            _tv.transforms.CenterCrop((384, 384)),
            _tv.transforms.ToTensor(),
            _tv.transforms.Normalize(mean=_M, std=_S)])
        return mk, ("musk", tf, "musk")
''' + c, 1)
d = '            if isinstance(kind, tuple) and kind[0] == "omiclip":'
assert d in s
s = s.replace(d, '''            if isinstance(kind, tuple) and kind[0] == "musk":
                from PIL import Image
                arr = np.asarray(h["img"][i:i + bs])[..., :3].astype(np.uint8)
                xb = torch.stack([kind[1](Image.fromarray(a_)) for a_ in arr]).to(dev, dtype=torch.float16)
                with torch.inference_mode():
                    f = model(image=xb, with_head=False, out_norm=False,
                              ms_aug=True, return_global=True)[0]
                f = f.float()
            elif isinstance(kind, tuple) and kind[0] == "omiclip":''', 1)
ast.parse(s)
open(p, "w").write(s)
print("MUSK 已接入")
