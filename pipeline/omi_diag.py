import numpy as np, torch, zipfile, traceback, inspect
p = "methods/OmiCLIP/checkpoint.pt"
print("numpy", np.__version__, "torch", torch.__version__)
print("zip ok:", zipfile.is_zipfile(p))
import open_clip
from open_clip import factory
print("open_clip", open_clip.__version__)
sig = inspect.signature(factory.create_model_from_pretrained)
print("create_model_from_pretrained 参数:", list(sig.parameters))
print("load_checkpoint 参数:", list(inspect.signature(factory.load_checkpoint).parameters))

print("\n[1] torch.load(weights_only=False)")
try:
    sd = torch.load(p, map_location="cpu", weights_only=False)
    k = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    print("    OK  顶层键:", list(sd)[:5] if isinstance(sd, dict) else type(sd))
    print("    state_dict 条目:", len(k) if hasattr(k, "__len__") else "?")
    ks = list(k)[:3] if hasattr(k, "keys") else []
    print("    前几个权重名:", ks)
except Exception as e:
    print("    失败:", str(e).split("\n")[0][:110]); traceback.print_exc()

print("\n[2] create_model + 手动 load_state_dict")
try:
    m = open_clip.create_model("coca_ViT-L-14", pretrained=None)
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {kk.replace("module.", ""): vv for kk, vv in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print("    OK  缺失 %d 个, 多余 %d 个" % (len(miss), len(unexp)))
    print("    缺失前 3:", list(miss)[:3], " 多余前 3:", list(unexp)[:3])
    with torch.no_grad():
        f = m.encode_image(torch.randn(1, 3, 224, 224))
    print("    encode_image 输出", tuple(f.shape))
except Exception as e:
    print("    失败:", str(e).split("\n")[0][:110]); traceback.print_exc()
