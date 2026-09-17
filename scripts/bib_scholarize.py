# -*- coding: utf-8 -*-
"""把 refs.bib 统一成 Google Scholar「引用 → BibTeX」的形态，元数据全部取自 Crossref / arXiv，不手写。
规则（与 Scholar 导出一致）：
  · 已有正式发表版本（Crossref 上非预印本 DOI）→ @article{journal, volume, number, pages, year, publisher} 或 @inproceedings{booktitle, pages, year, organization/publisher}
  · 仅 arXiv → @article{..., journal={arXiv preprint arXiv:ID}, year}
  · 作者写成 "Last, First and Last, First"
  · 模型发布（@misc，无论文）保持不变
匹配已发表版本的条件：标题相似度 ≥0.93 且第一作者姓氏一致；否则保留原条目形态。输出 results/bib_scholarize_report.json。"""
import re, json, time, sys, urllib.request, urllib.parse, difflib, xml.etree.ElementTree as ET
SRC, OUT = "paper/refs.bib", "paper/refs.bib"
bib = open(SRC).read(); LOG = open("results/bib_scholarize.log", "w")
def log(*a): print(*a, file=LOG, flush=True)
UA = {"User-Agent": "octave-bib/1.0 (mailto:anonymous@example.com)"}
def get(url, tries=4):
    for k in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503): time.sleep(10 * (k + 1)); continue
            if e.code == 404: return None
            raise
        except Exception: time.sleep(3 * (k + 1))
    return None
def field(body, name):
    m = re.search(r"\b" + name + r"\s*=\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|\"[^\"]*\"|[^,\n]+)", body, re.I)
    if not m: return ""
    v = m.group(1).strip(); v = v[1:-1] if v[:1] in "{\"" else v; return re.sub(r"\s+", " ", v).strip()
def norm(t): return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\\.|[{}]", "", t).lower()).strip()
def sim(a, b): return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
def bibtex_escape(s): return s.replace("&", r"\&").replace("%", r"\%") if s else s
def fmt_author_cr(a): return (a.get("family", "") + (", " + a.get("given", "") if a.get("given") else "")).strip(", ") if "family" in a else a.get("name", "")
def fmt_author_arxiv(name):
    p = name.strip().split(); return (p[-1] + ", " + " ".join(p[:-1])) if len(p) > 1 else name
def first_surname_arxiv(name): return name.strip().split()[-1].lower() if name.strip() else ""
PREPRINT = ("10.48550", "10.1101", "10.21203", "10.31219", "10.2139", "10.20944")
chunks = [c for c in re.split(r"\n(?=@)", "\n" + bib) if c.strip().startswith("@")]
header = bib.split(chunks[0].strip())[0] if chunks else ""
report, out_entries = [], []
def crossref_entry(key, m, note_extra=""):
    typ = m.get("type", ""); title = (m.get("title") or [""])[0]; authors = " and ".join(fmt_author_cr(a) for a in m.get("author", []))
    year = str(((m.get("issued") or {}).get("date-parts") or [[None]])[0][0] or ((m.get("published") or {}).get("date-parts") or [[None]])[0][0] or "")
    cont = (m.get("container-title") or [""])[0]; vol = m.get("volume", ""); num = m.get("issue", ""); pages = m.get("page", "").replace("-", "--") if m.get("page") else ""; pub = m.get("publisher", ""); doi = m.get("DOI", "")
    if typ in ("proceedings-article",) or ("conference" in cont.lower() or "proceedings" in cont.lower() or "advances in neural" in cont.lower() or "international conference" in cont.lower()):
        f = [("title", title), ("author", authors), ("booktitle", cont), ("pages", pages), ("year", year), ("organization" if "IEEE" in pub or "ACM" in pub else "publisher", pub), ("doi", doi)]
        t = "inproceedings"
    else:
        f = [("title", title), ("author", authors), ("journal", cont), ("volume", vol), ("number", num), ("pages", pages), ("year", year), ("publisher", pub), ("doi", doi)]
        t = "article"
    if note_extra: f.append(("note", note_extra))
    body = ",\n".join(f"  {k}={{{bibtex_escape(v)}}}" for k, v in f if v)
    return f"@{t}{{{key},\n{body}\n}}", dict(type=t, venue=cont, year=year, doi=doi)
def arxiv_entry(key, aid, title, authors, year, note_extra=""):
    f = [("title", title), ("author", " and ".join(fmt_author_arxiv(a) for a in authors)), ("journal", f"arXiv preprint arXiv:{aid}"), ("year", year)]
    if note_extra: f.append(("note", note_extra))
    return f"@article{{{key},\n" + ",\n".join(f"  {k}={{{bibtex_escape(v)}}}" for k, v in f if v) + "\n}", dict(type="article", venue=f"arXiv preprint arXiv:{aid}", year=year, doi="")
ns = {"a": "http://www.w3.org/2005/Atom"}
def arxiv_meta(aid):
    x = get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": aid, "max_results": 1})); time.sleep(3.2)
    if not x: return None
    e = ET.fromstring(x).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None: return None
    return dict(title=re.sub(r"\s+", " ", e.find("a:title", ns).text.strip()), authors=[a.find("a:name", ns).text for a in e.findall("a:author", ns)], year=e.find("a:published", ns).text[:4])
def crossref_by_doi(doi):
    r = get("https://api.crossref.org/works/" + urllib.parse.quote(doi)); time.sleep(0.5)
    return json.loads(r)["message"] if r else None
def crossref_search(title, surname):
    r = get("https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": title, "rows": 5})); time.sleep(0.5)
    if not r: return None
    best = None
    for it in json.loads(r)["message"]["items"]:
        t = (it.get("title") or [""])[0]; doi = it.get("DOI", "")
        if sim(title, t) < 0.93 or doi.startswith(PREPRINT) or it.get("type") in ("posted-content",): continue
        fams = [a.get("family", "").lower() for a in it.get("author", [])]
        if surname and fams and surname not in fams: continue
        if best is None or sim(title, t) > sim(title, (best.get("title") or [""])[0]): best = it
    return best
for c in chunks:
    c = c.strip(); m = re.match(r"@(\w+)\{([^,\s]+),", c)
    if not m: continue
    typ, key = m.group(1).lower(), m.group(2)
    title = field(c, "title"); year = field(c, "year"); doi = field(c, "doi") or ((re.search(r"10\.\d{4,9}/[^\s}\"]+", c) or [None]) and (re.search(r"10\.\d{4,9}/[^\s}\"]+", c).group(0).rstrip(".") if re.search(r"10\.\d{4,9}/[^\s}\"]+", c) else "")); arx = re.search(r"(\d{4}\.\d{4,5})", c); aid = arx.group(1) if arx else ""
    rec = dict(key=key, old_type=typ, old_venue=field(c, "journal") or field(c, "booktitle") or field(c, "howpublished"), old_year=year, action="kept", new_venue="", new_year="")
    if typ == "misc" and not aid and not doi:  # 模型发布
        out_entries.append(c); rec["action"] = "kept (model release)"; report.append(rec); continue
    new = None; note_extra = "arXiv:%s" % aid if (aid and doi and not doi.startswith(PREPRINT)) else ""
    try:
        if doi and not doi.startswith(PREPRINT):
            mm = crossref_by_doi(doi)
            if mm and sim(title, (mm.get("title") or [""])[0]) >= 0.85: new, info = crossref_entry(key, mm, note_extra); rec["action"] = "published (DOI)"
        if new is None:
            am = arxiv_meta(aid) if aid else None
            surname = first_surname_arxiv(am["authors"][0]) if am and am["authors"] else (re.split(r"\s+and\s+", field(c, "author"))[0].split(",")[0].strip().split()[-1].lower() if field(c, "author") else "")
            pub = crossref_search(title, surname)
            if pub:
                new, info = crossref_entry(key, pub, ("arXiv:%s" % aid) if aid else ""); rec["action"] = "published (title match)"
            elif am and sim(title, am["title"]) >= 0.85:
                new, info = arxiv_entry(key, aid, am["title"], am["authors"], am["year"]); rec["action"] = "arXiv preprint (Scholar form)"
    except Exception as ex:
        log("异常", key, ex)
    if new is None:
        out_entries.append(c); rec["action"] = rec["action"] if rec["action"] != "kept" else "kept (no retrievable record)"
    else:
        out_entries.append(new); rec["new_venue"], rec["new_year"] = info["venue"], info["year"]
    report.append(rec); log(key, rec["action"], rec["new_venue"], rec["new_year"])
open(OUT, "w").write(header.rstrip() + ("\n\n" if header.strip() else "") + "\n\n".join(out_entries) + "\n")
json.dump(report, open("results/bib_scholarize_report.json", "w"), indent=1, ensure_ascii=False)
from collections import Counter
print(Counter(r["action"] for r in report))
print("\n年份变化:", [(r["key"], r["old_year"], r["new_year"]) for r in report if r["new_year"] and r["new_year"] != r["old_year"]])
print("\n改为正式发表版本的条目:")
for r in report:
    if r["action"].startswith("published"): print(f"  {r['key']:22s} {r['old_venue'][:38]:38s} → {r['new_venue'][:60]} ({r['new_year']})")
