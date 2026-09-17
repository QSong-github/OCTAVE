# -*- coding: utf-8 -*-
"""重建 MANIFEST.txt：对项目内所有会被发布的文件取 SHA-256。
覆盖 results/ 与 results_new/ 全部结果文件、scripts/、jobs/、paper/ 的源与产出，
以及顶层说明文件。排除中间产物与缓存。用法：python3 scripts/mk_manifest.py"""
import hashlib, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIR = {".git", "__pycache__", ".ipynb_checkpoints"}
SKIP_EXT = {".aux", ".blg", ".out", ".fls", ".fdb_latexmk", ".pyc", ".zip"}
SKIP_NAME = {"MANIFEST.txt", ".DS_Store"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

rows = []
for base, dirs, files in os.walk(ROOT):
    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR)
    for f in sorted(files):
        if f in SKIP_NAME or os.path.splitext(f)[1] in SKIP_EXT: continue
        if ".bak" in f or f.endswith("~"): continue
        p = os.path.join(base, f)
        rows.append((sha(p), "./" + os.path.relpath(p, ROOT)))
with open(os.path.join(ROOT, "MANIFEST.txt"), "w") as fh:
    for h, rel in sorted(rows, key=lambda r: r[1]): fh.write(f"{h}  {rel}\n")
print(f"MANIFEST.txt: {len(rows)} 个文件")
