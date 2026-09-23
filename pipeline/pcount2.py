import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/path/to/project/src")
import torch
from hest_embed_v2 import encoder as load_encoder
P = "/path/to/project/results/encoder_params.json"
out = json.load(open(P))
for e in ["dinov2_large", "uni_v1", "uni_v2", "virchow", "virchow2"]:
    try:
        m, kind = load_encoder(e, "cpu")
        n = sum(p.numel() for p in m.parameters())
        out[e] = {"params": int(n), "hidden": getattr(getattr(m, "config", None), "hidden_size", None)}
        print("  %-14s %9.1f M  %s" % (e, n/1e6, type(m).__name__), flush=True); del m
    except Exception as ex:
        print("  %-14s 仍失败: %s" % (e, str(ex)[:70]), flush=True)
json.dump(out, open(P, "w"), indent=1)
print("\n共 %d 个" % len(out))
