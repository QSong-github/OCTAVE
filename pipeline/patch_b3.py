# -*- coding: utf-8 -*-
"""接入 hibou_b（HF + 自带 image_mean/std）与 gigapath_flash（手工注册 DINOv2-S SwiGLU 架构）。"""
import ast, shutil
p = "/path/to/systema4ST/src/hest_embed_v2.py"; s = open(p).read()
if "GIGA_FLASH" in s:
    print("已接入"); raise SystemExit
shutil.copy(p, p + ".bak_pre_b3")
# 1) 字典：放在 ALL_ENC 定义之前
anchor = 'ALL_ENC = ["resnet50", "ciga"] + list(HF_REPOS) + list(TIMM_REPOS) + TRIDENT_ENC + NEW_ENC'
assert s.count(anchor) == 1
s = s.replace(anchor, '''# 2026-09-03 第二批。hibou_b 的 config 没有 pixel_mean，但 preprocessor_config 里的 image_mean/std
# 是 (0.7068,0.5755,0.722)/(0.195,0.2316,0.1816)，不是 ImageNet 的 —— 走 HF_REPOS 的兜底会套错统计量，
# 故单独一支：AutoModel + AutoImageProcessor 的统计量。
HF_PROC = {"hibou_b": "histai/hibou-b"}
# gigapath_flash：权重是 timm 风格 (config.json + pytorch_model.bin)，但架构名 gigapath_tile_enc_dinov2s
# 只在作者的 gigapath 包里注册（要求 timm>=1.0.3）。这里按作者 gigapath/tile_encoder.py 的 _TILE_ENC_ARGS
# 直接构造 timm VisionTransformer 并严格加载 state_dict（缺/多任何键都报错）。预处理同 gigapath：ImageNet、224。
GIGA_FLASH = {"gigapath_flash": "prov-gigapath/prov-gigapath-flash"}
ALL_ENC = ["resnet50", "ciga"] + list(HF_REPOS) + list(TIMM_REPOS) + TRIDENT_ENC + NEW_ENC + list(GIGA_FLASH)''', 1)
# 2) 分支：插在 "if name in TRIDENT_ENC:" 之前
anchor2 = '    if name in TRIDENT_ENC:\n'
assert s.count(anchor2) == 1
s = s.replace(anchor2, '''    if name in HF_PROC:
        from transformers import AutoModel, AutoImageProcessor
        m = AutoModel.from_pretrained(HF_PROC[name], trust_remote_code=True).eval().to(dev)
        pr = AutoImageProcessor.from_pretrained(HF_PROC[name], trust_remote_code=True)
        pm, ps = list(pr.image_mean), list(pr.image_std)
        print("  %s 自带统计量 mean=%s std=%s" % (name, pm, ps), flush=True)
        v = lambda t: torch.tensor(t, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        return m, ("hfcfg", v(pm), v(ps))
    if name in GIGA_FLASH:
        import torch.nn as nn, torchvision.transforms as _T
        from timm.layers import SwiGLUPacked
        from timm.models.vision_transformer import VisionTransformer
        from huggingface_hub import hf_hub_download
        m = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
                              mlp_ratio=2048 / 384.0, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU,
                              init_values=1e-5, num_classes=0, global_pool="token",
                              class_token=True, reg_tokens=0)
        sd = torch.load(hf_hub_download(GIGA_FLASH[name], "pytorch_model.bin"), map_location="cpu")
        if isinstance(sd, dict) and "patch_embed.proj.weight" not in sd and "model" in sd:
            sd = sd["model"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, ("gigapath_flash 权重键不对应", missing[:5], unexpected[:5])
        print("  gigapath_flash: %d 个张量严格对上" % len(sd), flush=True)
        tf = _T.Compose([_T.Resize(224), _T.CenterCrop(224), _T.ToTensor(),
                         _T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return m.eval().to(dev), ("own", tf, "auto")
''' + anchor2, 1)
ast.parse(s); open(p, "w").write(s); print("已接入 hibou_b（HF 自带统计量）、gigapath_flash（手工注册）")
