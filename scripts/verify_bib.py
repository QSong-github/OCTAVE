# -*- coding: utf-8 -*-
"""逐条核验 refs.bib：优先 arXiv id（批量）、其次 Crossref DOI、再 Crossref 标题检索、最后 arXiv 标题检索。
输出 results/bib_verification.json 与进度日志；标题相似度 ≥0.85 记为核验通过。"""
import re, json, time, sys, urllib.request, urllib.parse, difflib, xml.etree.ElementTree as ET
bib = open("paper/refs.bib").read(); LOG = open("results/bib_verification.log", "w")
def log(*a): print(*a, file=LOG, flush=True)
chunks = [c for c in re.split(r"\n(?=@)", "\n" + bib) if c.strip().startswith("@")]
entries = []
for c in chunks:
    m = re.match(r"@(\w+)\{([^,\s]+),", c.strip())
    if m and m.group(1).lower() not in ("comment", "preamble", "string"): entries.append((m.group(1), m.group(2), c))
def field(body, name):
    m = re.search(r"\b" + name + r"\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)", body, re.I)
    if not m: return ""
    v = m.group(1).strip(); v = v[1:-1] if v[:1] in "{\"" else v; return re.sub(r"\s+", " ", v).strip()
def norm(t): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\\.|[{}]", "", t).lower()).strip()
def sim(a, b): return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
UA = {"User-Agent": "octave-bib-check/1.0 (mailto:wangqingai2481@gmail.com)"}
def get(url, timeout=40, tries=4):
    for k in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503): time.sleep(10 * (k + 1)); continue
            raise
        except Exception: time.sleep(3 * (k + 1))
    raise RuntimeError("retries exhausted")
recs = []
for typ, key, body in entries:
    dm = re.search(r"10\.\d{4,9}/[^\s}\"]+", body); arx = re.search(r"(\d{4}\.\d{4,5})", body)
    recs.append(dict(key=key, type=typ, title=field(body, "title"), year=field(body, "year"), doi=(field(body, "doi") or (dm.group(0) if dm else "")).rstrip("."), arxiv=arx.group(1) if arx else "", url=field(body, "url"), verified=None, found_title="", found_year="", how=""))
log("条目", len(recs))
ns = {"a": "http://www.w3.org/2005/Atom"}; found = {}
ids = [r["arxiv"] for r in recs if r["arxiv"]]
for i in range(0, len(ids), 20):
    try:
        x = ET.fromstring(get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": ",".join(ids[i:i + 20]), "max_results": 25})))
        for e in x.findall("a:entry", ns):
            aid = re.sub(r"v\d+$", "", e.find("a:id", ns).text.split("/abs/")[-1]); found[aid] = (re.sub(r"\s+", " ", e.find("a:title", ns).text.strip()), e.find("a:published", ns).text[:4])
        log("arXiv 批", i, "→", len(found))
    except Exception as ex: log("arXiv 批量失败:", ex)
    time.sleep(4)
for r in recs:
    if r["arxiv"] in found:
        t, y = found[r["arxiv"]]; r["found_title"], r["found_year"], r["how"] = t, y, "arXiv id"; r["verified"] = sim(r["title"], t) >= 0.85
for r in recs:
    if r["verified"] is None and r["doi"]:
        try:
            m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(r["doi"])))["message"]; t = (m.get("title") or [""])[0]; y = str((m.get("issued") or {}).get("date-parts", [[None]])[0][0])
            r["found_title"], r["found_year"], r["how"] = t, y, "Crossref DOI"; r["verified"] = sim(r["title"], t) >= 0.85
        except Exception: r["how"] = "DOI 查询失败"
        log("DOI", r["key"], r["verified"]); time.sleep(0.5)
for r in recs:
    if not r["verified"]:
        try:
            m = json.loads(get("https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": r["title"], "rows": 3, "select": "title,issued,DOI"})))["message"]["items"]
            best = max(m, key=lambda it: sim(r["title"], (it.get("title") or [""])[0]), default=None)
            if best and sim(r["title"], best["title"][0]) >= 0.85: r["found_title"], r["found_year"], r["how"] = best["title"][0], str((best.get("issued") or {}).get("date-parts", [[None]])[0][0]), "Crossref title"; r["verified"] = True
        except Exception: pass
        log("Crossref 标题", r["key"], r["verified"]); time.sleep(0.5)
for r in recs:
    if not r["verified"]:
        try:
            q = re.sub(r"[^A-Za-z0-9 ]", " ", re.sub(r"\\.|[{}]", "", r["title"]))[:150]
            x = ET.fromstring(get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": 'ti:"%s"' % q, "max_results": 3})))
            for e in x.findall("a:entry", ns):
                t = re.sub(r"\s+", " ", e.find("a:title", ns).text.strip())
                if sim(r["title"], t) >= 0.85: r["found_title"], r["found_year"], r["how"] = t, e.find("a:published", ns).text[:4], "arXiv title"; r["verified"] = True; break
        except Exception as ex: log("arXiv 标题失败", r["key"], ex)
        if not r["verified"]: r["how"] = r["how"] or "未检索到"
        log("arXiv 标题", r["key"], r["verified"]); time.sleep(4)
json.dump(recs, open("results/bib_verification.json", "w"), indent=1, ensure_ascii=False)
ok = [r for r in recs if r["verified"]]; bad = [r for r in recs if not r["verified"]]
print(f"已核验 {len(ok)}/{len(recs)}：", {h: sum(1 for r in ok if r['how'] == h) for h in set(r['how'] for r in ok)})
print("年份与来源不一致:", [(r["key"], r["year"], r["found_year"]) for r in ok if r["found_year"] and r["year"] and r["found_year"] != r["year"]])
print("标题相似度 <0.95（请人工看）:", [(r["key"], round(sim(r["title"], r["found_title"]), 2)) for r in ok if sim(r["title"], r["found_title"]) < 0.95])
print("\n未能自动核验（需人工）:")
for r in bad: print(f"  [{r['key']}] {r['title'][:85]} | {r['year']} | doi={r['doi'] or '-'} arxiv={r['arxiv'] or '-'} url={r['url'][:70] or '-'} | {r['how']}")
