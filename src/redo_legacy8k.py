# -*- coding: utf-8 -*-
"""
把四个旧实验统一到新表达基准上重跑。

为什么必须重跑：旧线用母项目 binned_16um.h5ad（200 基因面板 + 不同归一化），
新线用 10x 原始计数（18,085 基因，log1p only）。实测同一张片同一协议
PCC 从 0.6976 变成 0.8257 —— 两套数字不在同一基准上，并列即是本文批判的错误。

为什么不必重提嵌入：图像来源受控对照已证明 σ 不受图像管线影响（28 vs 29 µm，
而嵌入余弦仅 0.888）。且母项目 binned_16um.h5ad 与 st_bench 的 adata_16um.h5ad
是同一批 bin、同序（fulltx 已断言验证），故上游 25 套塔嵌入可直接接到原始计数上。

协议：跨片（P5 训 P2 / P2 训 P5），训练片选 top-50 HVG，alpha=1e4，天花板用 √c。
"""
import os, sys, json, glob, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import sparse
import evaluate as E, retrieval as R
from baselines import ridge_predict
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES, SEB

RAW = "/path/to/spatial2exp/he2st_align/st_bench/data/{s}/adata_16um.h5ad"
RES = "/path/to/systema4ST/results/legacy8k"
EXCLUDE_BASE = ("res2um","res4um","res8um","res16um","res32um","test_")
EXCLUDE_VAR  = ("_ctx","_grid")

def towers(mode="base"):
    out=[]
    for p in sorted(glob.glob(os.path.join(SEB.EMBDIR,"emb_*_P2.npy"))):
        t=os.path.basename(p)[4:-7]
        if any(x in t for x in EXCLUDE_BASE): continue
        if (mode=="base")==any(x in t for x in EXCLUDE_VAR): continue
        if os.path.exists(os.path.join(SEB.EMBDIR,f"emb_{t}_P5.npy")): out.append(t)
    return out

ap=argparse.ArgumentParser()
ap.add_argument("--tower"); ap.add_argument("--idx",type=int,default=-1)
ap.add_argument("--mode",default="base"); ap.add_argument("--list",action="store_true")
ap.add_argument("--hvg",type=int,default=50); ap.add_argument("--tmax",type=int,default=2048)
ap.add_argument("--bracket",action="store_true",help="三方夹逼（含 oracle_self / oracle_cv）")
ap.add_argument("--beta",action="store_true",help="B_blend(β) 换算表")
a_=ap.parse_args()
T=towers(a_.mode)
if a_.list: print(f"{len(T)} 塔:"," ".join(T)); raise SystemExit
cps=[1]
while cps[-1]<a_.tmax: cps.append(cps[-1]*2)

# ── 载入：坐标取自 binned（obsm['pxl']），表达取自 10x 原始计数 ──
b=ad.read_h5ad(SEB.H5AD)
bslide=b.obs["slide_id"].astype(str).values; bpxl=np.asarray(b.obsm["pxl"],np.float64)
D={}
for s in SLIDES:
    r=ad.read_h5ad(RAW.format(s=s)); C=sparse.csr_matrix(r.X)
    m=bslide==s
    assert C.shape[0]==m.sum(), f"{s}: 原始 {C.shape[0]} vs binned {m.sum()} —— bin 集不一致"
    D[s]=dict(C=C, Y=np.log1p(np.asarray(C.todense(),np.float32)), xy=bpxl[m]/PX_PER_UM[s])
    del r
print(f"表达基准 = 10x 原始计数 {D[SLIDES[0]]['Y'].shape[1]} 基因（旧线为 200 面板）",flush=True)

def ladder_eq(W,sig,y,val):
    lad={}; cur,t=y.copy(),0
    for cp in cps:
        while t<cp: cur=W@cur; t+=1
        lad[cp]=float(np.nanmean(E.per_gene_pcc(cur,y)))
    pts=sorted((sig[c],lad[c]) for c in cps)
    if val>=pts[0][1]: return pts[0][0],"低于最细档",lad
    for (s0,v0),(s1,v1) in zip(pts,pts[1:]):
        if v0>=val>=v1: return s0+(v0-val)/max(v0-v1,1e-12)*(s1-s0),"ok",lad
    return float("nan"),f">{pts[-1][0]:.0f}µm 右删失",lad

def ceiling(C,gidx,reps=2):
    Cg=C[:,gidx].tocsr(); rng=np.random.default_rng(0); ch=[]
    for _ in range(reps):
        h=rng.binomial(Cg.data.astype(np.int64),0.5).astype(np.float32)
        f=lambda d: np.log1p(np.asarray(sparse.csr_matrix((d,Cg.indices,Cg.indptr),shape=Cg.shape).todense(),np.float32))
        ch.append(E.per_gene_pcc(f(h),f(Cg.data-h)))
    c2=float(np.nanmean(np.mean(ch,0))); return float(np.sqrt(max(2*c2/(1+c2),0)))

os.makedirs(RES,exist_ok=True)

if a_.beta:
    # β 换算表：完美粗结构 + 比例 β 的真实细结构，看报告 PCC
    out={}
    for s in SLIDES:
        W=build_operator(D[s]["xy"],k=8,cut_um=29.0); sig=calibrate_sigma(W,D[s]["xy"],cps)
        tr=[x for x in SLIDES if x!=s][0]
        gidx=E.topk_hvg(D[tr]["Y"],a_.hvg); y=D[s]["Y"][:,gidx]
        cur,t=y.copy(),0
        while t<128: cur=W@cur; t+=1                      # σ≈110µm 的粗结构
        coarse=cur; fine=y-coarse
        row={}
        for beta in (0,0.01,0.025,0.05,0.10,0.20,0.50,1.0):
            P=coarse+beta*fine
            v=float(np.nanmean(E.per_gene_pcc(P,y)))
            eq,fl,_=ladder_eq(W,sig,y,v); row[str(beta)]=dict(pcc=v,eq=float(eq),flag=fl)
            print(f"  [{s[-2:]}] β={beta:<5} PCC={v:.4f}  等价σ={eq:.0f}µm",flush=True)
        out[s]=row
    json.dump(out,open(f"{RES}/beta_table.json","w"),indent=2,ensure_ascii=False)
    print("已存 beta_table.json"); raise SystemExit

if a_.bracket:
    out={}
    for s in SLIDES:
        tr=[x for x in SLIDES if x!=s][0]
        W=build_operator(D[s]["xy"],k=8,cut_um=29.0); sig=calibrate_sigma(W,D[s]["xy"],cps)
        gidx=E.topk_hvg(D[tr]["Y"],a_.hvg); y=D[s]["Y"][:,gidx]
        X=np.nan_to_num(np.load(os.path.join(SEB.EMBDIR,f"emb_hibou_l_{s[-2:]}.npy")).astype(np.float32))
        Xtr=np.nan_to_num(np.load(os.path.join(SEB.EMBDIR,f"emb_hibou_l_{tr[-2:]}.npy")).astype(np.float32))
        M={}
        M["oracle_self"]=ridge_predict(X,y,X,1e4)                       # 用测试片自己拟合（上界）
        n=len(y); rng=np.random.default_rng(0); fold=rng.integers(0,5,n)
        P=np.full_like(y,np.nan)
        for f in range(5):
            te=fold==f; P[te]=ridge_predict(X[~te],y[~te],X[te],1e4)
        M["oracle_cv"]=P
        M["ridge_xs"]=ridge_predict(Xtr,D[tr]["Y"][:,gidx],X,1e4)       # 跨片
        M["imageKNN"]=R.image_floor(X,Xtr,D[tr]["Y"],k=800)[:,gidx]
        row={}
        for k,Pk in M.items():
            v=float(np.nanmean(E.per_gene_pcc(Pk,y))); eq,fl,_=ladder_eq(W,sig,y,v)
            row[k]=dict(pcc=v,eq=float(eq),flag=fl)
            print(f"  [{s[-2:]}] {k:12s} PCC={v:.4f}  等价σ={eq:.0f}µm ({fl})",flush=True)
        out[s]=dict(methods=row,ceiling=ceiling(D[s]["C"],gidx))
    json.dump(out,open(f"{RES}/bracket.json","w"),indent=2,ensure_ascii=False)
    print("已存 bracket.json"); raise SystemExit

# ── 塔扫描 / FOV 律 ──
tw=a_.tower or T[a_.idx]
print(f"塔 = {tw}（{a_.mode} 清单共 {len(T)}）",flush=True)
out={}
for s in SLIDES:
    tr=[x for x in SLIDES if x!=s][0]
    W=build_operator(D[s]["xy"],k=8,cut_um=29.0); sig=calibrate_sigma(W,D[s]["xy"],cps)
    gidx=E.topk_hvg(D[tr]["Y"],a_.hvg); y=D[s]["Y"][:,gidx]
    X=np.nan_to_num(np.load(os.path.join(SEB.EMBDIR,f"emb_{tw}_{s[-2:]}.npy")).astype(np.float32))
    Xtr=np.nan_to_num(np.load(os.path.join(SEB.EMBDIR,f"emb_{tw}_{tr[-2:]}.npy")).astype(np.float32))
    assert X.shape[0]==len(y), f"{tw}/{s}: 嵌入 {X.shape[0]} vs 表达 {len(y)}"
    P=ridge_predict(Xtr,D[tr]["Y"][:,gidx],X,1e4)
    v=float(np.nanmean(E.per_gene_pcc(P,y))); eq,fl,lad=ladder_eq(W,sig,y,v)
    out[s]=dict(pcc=v,eq_sigma=float(eq),flag=fl,dim=int(X.shape[1]),
                ceiling=ceiling(D[s]["C"],gidx),
                sigma_um={str(c):sig[c] for c in cps},ladder={str(c):lad[c] for c in cps})
    print(f"  [{s[-2:]}] PCC={v:.4f}  等价σ={eq:.0f}µm ({fl})",flush=True)
out["_tower"]=tw
json.dump(out,open(f"{RES}/tower_{tw}.json","w"),indent=2,ensure_ascii=False)
print(f"已存 tower_{tw}.json")
