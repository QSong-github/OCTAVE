# -*- coding: utf-8 -*-
"""给两个适配器加 --fold，使数组作业能逐折并行；每个任务独立落盘，避免并发写同一文件。"""
import os, ast, re
B = "/blue/qsong1/wang.qing/systema4ST"
SPEC = [("src/hggep_hest.py", "a_",
         '        for kf in range(len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))):\n'),
        ("src/heclip_hest.py", "a",
         '        for kf in range(nsp):\n')]
for f, var, anchor in SPEC:
    p = os.path.join(B, f)
    s = open(p).read()
    if "--fold" in s:
        print(f, "已有 --fold，跳过"); continue
    key = '    ap.add_argument("--cohorts"'
    assert key in s, "cohorts 锚点缺失 " + f
    s = s.replace(key,
                  '    ap.add_argument("--fold", type=int, default=-1,\n'
                  '                    help="只跑该折；-1 为全部。数组作业逐折并行时用，"\n'
                  '                         "每任务写自己的 --out，避免并发写同一文件互相覆盖")\n' + key, 1)
    assert anchor in s, "折循环锚点缺失 " + f
    s = s.replace(anchor, anchor + "            if %s.fold >= 0 and kf != %s.fold:\n                continue\n" % (var, var), 1)
    ast.parse(s)
    open(p, "w").write(s)
    print(f, "已加 --fold（变量名 %s）" % var)
