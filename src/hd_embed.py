# -*- coding: utf-8 -*-
"""
新片的塔嵌入 —— 严格复刻母项目 extract_wsi_emb.py 的协议，保证与 P2/P5 可比。

协议（不可改动，改了等价 σ 就不可比）:
  · openslide 读 level 0
  · 以 obsm['pxl'] 为中心切 ctx_px=224 的方块 → 61.4 µm 物理视野（对上 FOV 律的基座档）
  · grid=1（单 patch，不做子块池化）
  · trident encoder_factory(name).eval_transforms 做预处理

--validate 模式: 用母项目自己的 P2 金字塔 tiff 跑一遍，与母项目的 emb_hibou_l_P2.npy 对比。
若逐 bin 余弦相似度 ≈1，则证明本脚本与母项目管线等价，新片的嵌入可以直接并列使用。
"""
import os, glob, argparse, numpy as np, torch, anndata as ad

def build(name, device):
    from trident.patch_encoder_models.load import encoder_factory
    e = encoder_factory(name).to(device).eval()
    return e, e.eval_transforms, getattr(e, "precision", torch.float32)

def embed(model, tf, prec, tiff, pxl, ctx_px, device, bs=256):
    import openslide
    from PIL import Image
    sl = openslide.OpenSlide(tiff); half = ctx_px // 2
    out = []
    for i in range(0, len(pxl), bs):
        pb = pxl[i:i+bs]
        tiles = [Image.fromarray(np.asarray(
                    sl.read_region((int(x-half), int(y-half)), 0, (ctx_px, ctx_px)).convert("RGB")))
                 for x, y in pb]
        xb = torch.stack([tf(t) for t in tiles]).to(device)
        use_ac = prec in (torch.float16, torch.bfloat16)
        with torch.inference_mode(), torch.autocast("cuda", dtype=prec, enabled=use_ac):
            e = model(xb)
        out.append(e.float().cpu().numpy().astype(np.float32))
        if (i // bs) % 100 == 0: print(f"    {i+len(pb)}/{len(pxl)}", flush=True)
    return np.concatenate(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad"); ap.add_argument("--tiff"); ap.add_argument("--out")
    ap.add_argument("--encoder", default="hibou_l"); ap.add_argument("--ctx_px", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--validate", action="store_true",
                    help="用母项目 P2 的 tiff+h5ad 复现其 emb_hibou_l_P2.npy")
    ap.add_argument("--val_tiff", default=None,
                    help="验证时改用指定 tiff（用于检验 vips 转换是否引入偏差）")
    a_ = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a_.validate:
        PA = "/path/to/spatial2exp/he2st_align"
        import h5py
        with h5py.File(f"{PA}/data/binned_16um.h5ad", "r") as h:
            sid = h["obs"]["slide_id"]
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in sid["categories"][:]]
            names = np.array(cats)[sid["codes"][:]]
            pxl = h["obsm"]["pxl"][:]
        m = np.char.endswith(names.astype(str), "P2")
        pxl = pxl[m][:4000]                                   # 前 4000 个 bin 足以判定等价性
        tiff = a_.val_tiff or glob.glob(f"{PA}/data/virtualST/*P2/*PYRAMIDAL*.tif*")[0]
        print(f"  用图: {tiff}", flush=True)
        model, tf, prec = build(a_.encoder, dev)
        mine = embed(model, tf, prec, tiff, pxl, a_.ctx_px, dev, a_.batch)
        ref = np.load(f"{PA}/results/emb_{a_.encoder}_P2.npy")[:4000].astype(np.float32)
        assert mine.shape == ref.shape, f"形状不符 {mine.shape} vs {ref.shape}"
        cos = (mine*ref).sum(1) / (np.linalg.norm(mine,axis=1)*np.linalg.norm(ref,axis=1) + 1e-9)
        d = np.abs(mine - ref).max()
        print(f"\n=== 验证 {a_.encoder} / P2 前 {len(pxl)} bin ===")
        print(f"  余弦相似度  中位={np.median(cos):.6f}  最小={cos.min():.6f}  >0.999 占比={np.mean(cos>0.999):.4f}")
        print(f"  逐元素最大绝对差 = {d:.6f}")
        print("  ⇒ " + ("等价" if np.median(cos) > 0.999
                        else "不等价，需排查后再用"))
        raise SystemExit(0)

    a = ad.read_h5ad(a_.h5ad, backed="r")
    pxl = np.asarray(a.obsm["pxl"], np.float64)
    print(f"[{os.path.basename(a_.h5ad)}] {len(pxl)} bin  enc={a_.encoder} ctx={a_.ctx_px} {dev}", flush=True)
    model, tf, prec = build(a_.encoder, dev)
    E = embed(model, tf, prec, a_.tiff, pxl, a_.ctx_px, dev, a_.batch)
    os.makedirs(os.path.dirname(a_.out), exist_ok=True)
    np.save(a_.out, E)
    print(f"已存 {a_.out}  {E.shape}")
