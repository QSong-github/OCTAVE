# -*- coding: utf-8 -*-
"""接入扩集的 16 个编码器（15 个探针成功 + DINOv3 ViT-H+ 试用现有 token）。
原则不变：每个模型走自己官方的预处理（timm 的 data config / HF processor / open_clip 自带 / trident 自带），绝不默认套 ImageNet。"""
import ast, shutil
p = "/path/to/project/src/hest_embed_v2.py"
shutil.copy(p, p + ".bak_pre_new15")
s = open(p).read()
assert "NEW_TIMM2" not in s
# ① 字典
a = 'NEW_ENC = list(NEW_TIMM) + list(NEW_CLIP) + list(NEW_HF2) + ["omiclip", "omiclip_raw", "musk"]'
assert s.count(a) == 1
s = s.replace(a, '''# 2026-09-03 扩集：timm hf-hub（各自 data config 预处理）、SigLIP2、BiomedCLIP、RetCCL、更多 HF/CLIP 通用基线
NEW_TIMM2 = {"kaiko_vitb8": "hf-hub:1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms",
             "lunit_vits16": "hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
             "lunit_r50_swav": "hf-hub:1aurent/resnet50.lunit_swav",
             "lunit_r50_bt": "hf-hub:1aurent/resnet50.lunit_bt",
             "lunit_r50_moco": "hf-hub:1aurent/resnet50.lunit_mocov2"}
NEW_SIGLIP = {"siglip2": "google/siglip2-so400m-patch14-384"}
NEW_OPENCLIP = {"biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"}
RETCCL = {"retccl": ("jamesdolezal/RetCCL", "retccl_torchscript.pth")}
NEW_ENC = (list(NEW_TIMM) + list(NEW_CLIP) + list(NEW_HF2) + ["omiclip", "omiclip_raw", "musk"]
           + list(NEW_TIMM2) + list(NEW_SIGLIP) + list(NEW_OPENCLIP) + list(RETCCL))''', 1)
# ② 现有字典追加
b = '"phaet": "wearewaiv/phaet", "mascaret": "wearewaiv/mascaret"}'
assert s.count(b) == 1
s = s.replace(b, '"phaet": "wearewaiv/phaet", "mascaret": "wearewaiv/mascaret",\n            "dinov2_base": "facebook/dinov2-base", "dinov2_giant": "facebook/dinov2-giant",\n            "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m", "dinov3_vith16": "facebook/dinov3-vith16plus-pretrain-lvd1689m"}', 1)
c = 'NEW_CLIP = {"plip": "vinid/plip", "quiltnet": "wisdomik/QuiltNet-B-32"}'
assert s.count(c) == 1
s = s.replace(c, 'NEW_CLIP = {"plip": "vinid/plip", "quiltnet": "wisdomik/QuiltNet-B-32",\n            "clip_vitl14": "openai/clip-vit-large-patch14", "pathgen_clip": "jamessyx/pathgenclip-vit-large-patch14-hf"}', 1)
d = 'TRIDENT_ENC = ["uni_v1", "uni_v2", "virchow", "virchow2", "gigapath", "conch_v1",\n               "conch_v15", "hoptimus0", "keep", "midnight12k", "openmidnight", "hibou_l"]'
assert s.count(d) == 1
s = s.replace(d, 'TRIDENT_ENC = ["uni_v1", "uni_v2", "virchow", "virchow2", "gigapath", "conch_v1",\n               "conch_v15", "hoptimus0", "keep", "midnight12k", "openmidnight", "hibou_l",\n               "ctranspath", "gpfm"]', 1)
# ③ 加载分支（插在 NEW_CLIP 分支之前）
e = '    if name in NEW_CLIP:\n        from transformers import CLIPModel, CLIPImageProcessor'
assert s.count(e) == 1
s = s.replace(e, '''    if name in NEW_TIMM2:
        # 普通 timm hf-hub：num_classes=0 得到池化特征；预处理取模型自带 data config（Kaiko/Lunit 的 mean/std 各不相同）
        import timm
        from timm.data import resolve_model_data_config, create_transform
        m = timm.create_model(NEW_TIMM2[name], pretrained=True, num_classes=0).eval().to(dev)
        tf = create_transform(**resolve_model_data_config(m), is_training=False)
        return m, ("own", tf, "auto")
    if name in NEW_SIGLIP:
        # SiglipModel 的 get_image_features / SiglipImageProcessor 接口与 CLIP 相同，复用 clip 前向
        from transformers import SiglipModel, SiglipImageProcessor
        m = SiglipModel.from_pretrained(NEW_SIGLIP[name]).eval().to(dev)
        pr = SiglipImageProcessor.from_pretrained(NEW_SIGLIP[name])
        return m, ("clip", pr, "clip")
    if name in NEW_OPENCLIP:
        # open_clip hf-hub，自带 preprocess；encode_image 走 omiclip 前向，normalize=False（同 omiclip_raw 口径）
        from open_clip import create_model_from_pretrained
        m, pre = create_model_from_pretrained(NEW_OPENCLIP[name], device=dev)
        return m.eval().to(dev), ("omiclip", pre, name)
    if name in RETCCL:
        # 作者移植的 torchscript ResNet50；官方 RetCCL 用 ImageNet 归一化、224 输入
        from huggingface_hub import hf_hub_download
        import torch.nn as nn, torchvision.transforms as _T
        repo, fn = RETCCL[name]
        ts = torch.jit.load(hf_hub_download(repo, fn), map_location=dev).eval()
        class _W(nn.Module):
            def __init__(self, ts): super().__init__(); self.ts = ts
            def forward(self, x):
                o = self.ts(x); return o[0] if isinstance(o, (tuple, list)) else o
        tf = _T.Compose([_T.Resize(224), _T.CenterCrop(224), _T.ToTensor(),
                         _T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return _W(ts).eval().to(dev), ("own", tf, "auto")
''' + e, 1)
ast.parse(s); open(p, "w").write(s)
print("hest_embed_v2.py 已接入 16 个：", "kaiko_vitb8 lunit_vits16 lunit_r50_swav lunit_r50_bt lunit_r50_moco ctranspath gpfm retccl dinov2_base dinov2_giant dinov3_vitb16 dinov3_vith16 clip_vitl14 pathgen_clip siglip2 biomedclip")

# ridge_loco.py：没有地板目录时，从 hest_blocks_<e>.json 取队列归属
q = "/path/to/project/ridge_loco.py"; t = open(q).read()
old = '''    coh = {}
    for f in glob.glob(f"{R}/hest_floor_{e}/*.json"):
        d = json.load(open(f))
        for s in d["samples"]: coh[s] = d["cohort"]'''
assert t.count(old) == 1
t = t.replace(old, '''    coh = {}
    for f in glob.glob(f"{R}/hest_floor_{e}/*.json"):
        d = json.load(open(f))
        for s in d["samples"]: coh[s] = d["cohort"]
    if not coh:   # 扩集编码器不跑检索地板；队列归属从块预言机文件取
        for s, v in json.load(open(f"{R}/hest_blocks_{e}.json"))["samples"].items(): coh[s] = v["cohort"]''', 1)
ast.parse(t); open(q, "w").write(t); print("ridge_loco.py 已加队列归属回退")
