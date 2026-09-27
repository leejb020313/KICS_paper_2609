"""Verify every reference claim of the KICS draft against the cited sources, then write reference-dossier.md.

Checks
  1. every quote in references.json is found verbatim in its cached PDF (pdfs/); its page is recorded
  2. every [n] cited in the draft has an entry, and every entry marked as cited is cited
  3. every reference-list line of the draft equals the 'bib' string of its entry
  4. (--online) title, authors and year still match the arXiv API

Usage: uv run --no-project --with pymupdf python verify_references.py <draft.pdf> [--online]
Exit code 1 if anything fails.
"""
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = json.load(open(os.path.join(HERE, "references.json"), encoding="utf-8"))


def norm(s):
    """Collapse whitespace, unify quote characters, and undo end-of-line hyphenation."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("­", "")
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"(\w)- (\w)", r"\1\2", s).strip()


def dehyphen(s):
    """Fallback for a real hyphen that falls at a line break ('strong-model- only'): ignore hyphens entirely."""
    return norm(s).replace("-", "")


def find_pages(pdf, quote):
    doc = pymupdf.open(os.path.join(HERE, "pdfs", pdf))
    texts = [p.get_text() for p in doc]
    for f in (norm, dehyphen):
        q, full = f(quote), [f(t) for t in texts]
        pages = [i + 1 for i, t in enumerate(full) if q in t]
        if not pages:  # quote spanning a page break
            pages = [i + 1 for i in range(len(full) - 1) if q in full[i] + " " + full[i + 1]]
        if pages:
            return pages
    return []


def check_draft(draft):
    text = norm("".join(p.get_text() for p in pymupdf.open(draft)))
    body, _, reflist = text.partition("참 고 문 헌")
    # reference markers only: skip math intervals such as s(x)∈[0,1]
    cited = {int(n) for grp in re.findall(r"(?<!∈)\[(\d+(?:\s*,\s*\d+)*)\]", body) for n in re.split(r"\s*,\s*", grp)}
    entries = {r["n"]: r for r in REFS["references"] if r.get("cited_in_draft", True)}
    errors = []
    if cited != set(entries):
        errors.append(f"cited in draft {sorted(cited)} != dossier entries {sorted(entries)}")
    for n, r in entries.items():
        nxt = re.escape(f"[{n + 1}]") if n + 1 in entries else "$"
        m = re.search(re.escape(f"[{n}]") + r"\s*(.*?)\s*(?=" + nxt + ")", reflist)
        # hyphen-insensitive: a hyphen at a line break ("Fine-" + newline + "Tuned") looks like a soft break
        if not m or dehyphen(m.group(1)).replace(" ", "") != dehyphen(r["bib"]).replace(" ", ""):
            errors.append(f"[{n}] reference list differs from dossier:\n      draft:   {m.group(1) if m else None}\n      dossier: {r['bib']}")
    return errors


def check_online():
    ns = {"a": "http://www.w3.org/2005/Atom"}
    live = {}
    for aid in [r["arxiv"] for r in REFS["references"] if r.get("arxiv")]:
        # curl, not urllib: export.arxiv.org answers HTTP 406 to Python's client regardless of User-Agent
        out = subprocess.run(["curl", "-s", "--fail", f"https://export.arxiv.org/api/query?id_list={aid}"],
                             capture_output=True, timeout=90)
        if out.returncode != 0:
            return [f"[{aid}] arXiv API unavailable (curl exit {out.returncode}); online check not completed"]
        e = ET.fromstring(out.stdout).find("a:entry", ns)
        live[aid] = ([a.find("a:name", ns).text for a in e.findall("a:author", ns)], " ".join(e.find("a:title", ns).text.split()),
                     e.find("a:published", ns).text[:4])
        time.sleep(3)  # arXiv API etiquette
    errors = []
    for r in REFS["references"]:
        if not r.get("arxiv"):
            continue
        authors, title, year = live[r["arxiv"]]
        if authors != r["authors_official"]:
            errors.append(f"[{r['n']}] authors changed on arXiv: {authors}")
        if norm(title).lower() not in norm(r["bib"]).lower():
            errors.append(f"[{r['n']}] title mismatch: arXiv '{title}'")
        if year not in r["bib"]:
            errors.append(f"[{r['n']}] year mismatch: arXiv {year}")
    return errors


def write_dossier(pages):
    L = ["# 참고문헌 검증 자료 (cascade_kics_draft)", "",
         "`verify_references.py`가 생성합니다. 모든 인용문은 `pdfs/`의 원문 PDF에서 그대로 찾은 문장이며, 쪽수는 PDF 쪽 번호입니다. "
         "`support: inference`는 원문에 그대로 쓰인 문장이 아니라 원문을 근거로 한 **우리의 해석**이라는 뜻입니다.", ""]
    claims = [(r, c) for r in REFS["references"] for c in r["claims"]]
    inf = [f"{c['id']} ({c['paper'][:40]}…)" for r, c in claims if c["support"] == "inference"]
    L += ["## 요약", "",
          f"- 참고문헌 {sum(1 for r in REFS['references'] if r.get('cited_in_draft', True))}편, 근거 {len(claims)}건 "
          f"(원문 직접 확인 {sum(1 for _, c in claims if c['support'] == 'direct')}건, 해석 {len(inf)}건)",
          f"- 원문 PDF에서 찾은 인용문 {sum(1 for v in pages.values() if v)}/{len(pages)}건 (유료 원문 {len(claims) - len(pages)}건은 공개 초록으로 확인)",
          "- 해석에 해당하는 근거 (심사에서 질문받으면 note의 논리로 답해야 함): " + ", ".join(inf)]
    L += [f"- 권고 ([{r['n']}]): {r['recommendation']}" for r in REFS["references"] if r.get("recommendation")]
    L += [""]
    for r in REFS["references"]:
        tag = f"[{r['n']}]" if r.get("cited_in_draft", True) else f"({r['n']}, 본문 미인용)"
        L += [f"## {tag} {r['bib']}", ""]
        ident = f"arXiv:{r['arxiv']} ({r['version_read']} 확인)" if r.get("arxiv") else (f"DOI {r['doi']}" if r.get("doi") else r.get("url", ""))
        L += [f"- 식별자: {ident} · 공개일 {r['published']} · {r['venue']}",
              f"- 공식 저자: {', '.join(r['authors_official'])}",
              f"- 원문 PDF: {'`pdfs/' + r['pdf'] + '`' if r.get('pdf') else '없음 (유료 원문, 공개 초록으로 확인)'}", ""]
        if r.get("recommendation"):
            L += [f"> **권고:** {r['recommendation']}", ""]
        for c in r["claims"]:
            pg = pages.get(c["id"])
            where = f"p. {', '.join(map(str, pg))}" if pg else ("공개 초록" if not r.get("pdf") else "**찾지 못함**")
            L += [f"**{c['id']}** ({c['where']}) — 논문: {c['paper']}", "",
                  f"> \"{c['quote']}\" ({where})", "",
                  f"- 근거 유형: {c['support']}" + (f" — {c['note']}" if c.get("note") else ""), ""]
    L += ["## 검토했으나 인용하지 않은 논문", "", "| arXiv | 제목 | 확인 깊이 | 판단 |", "|---|---|---|---|"]
    L += [f"| {s['arxiv']} | {s['title']} | {s['depth']} | {s['verdict']} |" for s in REFS["screened_not_cited"]]
    L += ["", "## 검색 기록", "", "| 날짜 | 소스 | 질의 | 결과 |", "|---|---|---|---|"]
    L += [f"| {s['date']} | {s['source']} | {s['query']} | {s.get('result', '')} |" for s in REFS["search_log"]]
    open(os.path.join(HERE, "reference-dossier.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    errors, pages = [], {}
    for r in REFS["references"]:
        for c in r["claims"]:
            if not r.get("pdf"):
                print(f"SKIP  {c['id']:4s} no open PDF ({r['key']})")
                continue
            pg = find_pages(r["pdf"], c["quote"])
            pages[c["id"]] = pg
            print(f"{'OK  ' if pg else 'FAIL'}  {c['id']:4s} p.{pg}  {c['quote'][:70]}")
            if not pg:
                errors.append(f"quote {c['id']} not found in {r['pdf']}")
    errors += check_draft(sys.argv[1])
    if "--online" in sys.argv:
        errors += check_online()
    write_dossier(pages)
    print("\n" + ("ALL REFERENCES VERIFIED" if not errors else "FAILURES:\n  " + "\n  ".join(errors)))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
