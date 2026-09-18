import torch, numpy as np, inspect
from open_clip import create_model_from_pretrained
p = "/path/to/systema4ST/methods/OmiCLIP/checkpoint.pt"
m, pre = create_model_from_pretrained("coca_ViT-L-14", device="cpu", pretrained=p, weights_only=False)
x = torch.randn(2, 3, 224, 224)
print("visual 类型:", type(m.visual).__name__)
print("encode_image 参数:", list(inspect.signature(m.encode_image).parameters))
with torch.no_grad():
    a = m.encode_image(x)
    print("encode_image        ->", tuple(a.shape), " 行范数 %.4f" % float(a.norm(dim=1).median()))
    try:
        b = m.encode_image(x, normalize=False)
        print("encode_image(norm=F)->", tuple(b.shape), " 行范数 %.4f" % float(b.norm(dim=1).median()))
    except Exception as e:
        print("encode_image(norm=F) 不支持:", str(e)[:70])
    for attr in ("trunk", "proj", "head"):
        print("  visual 有 %s:" % attr, hasattr(m.visual, attr))
    try:
        t = m.visual.trunk(x)
        print("visual.trunk        ->", tuple(t.shape), " 行范数 %.4f" % float(t.reshape(2,-1).norm(dim=1).median()))
    except Exception as e:
        print("visual.trunk 取不到:", str(e)[:70])
    try:
        v = m.visual(x)
        v0 = v[0] if isinstance(v, (tuple, list)) else v
        print("visual(x)           ->", tuple(v0.shape))
    except Exception as e:
        print("visual(x) 失败:", str(e)[:70])
