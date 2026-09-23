#!/bin/bash
cd /path/to/project
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 fix_raw.py
rm -f results/hest_emb/*_omiclip_raw.npz results/hest_effres_ps_omiclip_raw.json
rm -rf results/hest_floor_omiclip_raw results/hest_floor_k*_omiclip_raw
say "阶段1 重抽 omiclip_raw 嵌入"
sbatch jobs/fixup.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n fixup | wc -l)" -gt 0 ]; do sleep 60; done
N=$(ls results/hest_emb/*_omiclip_raw.npz 2>/dev/null | wc -l); say "  嵌入 $N/72"
python3 -c "
import numpy as np,glob
a=np.load(sorted(glob.glob('results/hest_emb/*_omiclip.npz'))[0])['X']
b=np.load(sorted(glob.glob('results/hest_emb/*_omiclip_raw.npz'))[0])['X']
print('  omiclip 行范数 %.4f  omiclip_raw 行范数 %.4f  最大差 %.4f'%(
  np.median(np.linalg.norm(a,axis=1)), np.median(np.linalg.norm(b,axis=1)), np.abs(a-b).max()))
"
[ "$N" -ge 72 ] || { say "嵌入未齐，中止"; exit 1; }
say "阶段2 发下游"
sbatch jobs/orps.sh >/dev/null; sbatch jobs/orfloor.sh >/dev/null
while [ "$(squeue -u $USER -r -h | wc -l)" -gt 0 ]; do sleep 60; done
say "阶段3 汇总"
sbatch jobs/agg27.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n agg27 | wc -l)" -gt 0 ]; do sleep 20; done
cat "$(ls -t logs/agg27_*.out | head -1)"
