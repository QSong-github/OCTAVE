# -*- coding: utf-8 -*-
"""Compare two result JSONs leaf by leaf . Prints the largest absolute difference and the leaves that differ by more than 1e-4."""
import json, math, sys


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, f"{p}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield p, float(x)


A = dict(leaves(json.load(open(sys.argv[1])))); B = dict(leaves(json.load(open(sys.argv[2]))))
common = sorted(set(A) & set(B))
print(f"numeric leaves: old {len(A)}, new {len(B)}, shared {len(common)}")
for k in sorted(set(A) - set(B))[:10]: print("  only in old:", k)
for k in sorted(set(B) - set(A))[:10]: print("  only in new:", k)
d = []
for k in common:
    a, b = A[k], B[k]
    if math.isnan(a) and math.isnan(b): continue
    d.append((math.inf if math.isnan(a) or math.isnan(b) else abs(a - b), k))
d.sort(reverse=True)
print("largest abs diff:", d[0] if d else None)
bad = [x for x in d if x[0] > 1e-4]
print(f"{len(bad)} of {len(d)} leaves differ by more than 1e-4")
for x in bad[:40]: print(f"  {x[1]}: old {A[x[1]]:.6f} new {B[x[1]]:.6f}")
