"""Check that every result number and significance claim printed in the JEV paper PDF matches the result files.

Numbers come from results/jev_strict_full.json (scripts/jev_strict.py), data sizes from results/final_results_full.json,
the wording pilot from results/jev/{short,nnppi}/*cal.jsonl.

Usage: python scripts/verify_jev_paper_numbers.py <paper.pdf>
"""
import json
import os
import re
import sys

import pymupdf
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "jev_strict_full.json"), encoding="utf-8"))
M = json.load(open(os.path.join(ROOT, "results", "final_results_full.json"), encoding="utf-8"))
text = re.sub(r"\s+", " ", "".join(p.get_text() for p in pymupdf.open(sys.argv[1])).replace("​", ""))
acc = lambda d, n: R[d]["metrics"][n]["acc"][0]
t = lambda d, c: R[d]["tests"][c]
ns = lambda d, c: t(d, c)["n_sig"]
pos = lambda d, c: all(x > 0 for x in t(d, c)["diff"])
neg = lambda d, c: all(x < 0 for x in t(d, c)["diff"])
pct = lambda b: f"{round(100 * b)}%"
rate = lambda d, v="jevfuse": R[d]["rate"][v]
F = {d: f"jevfuse_{rate(d)}" for d in ("clef", "cb")}
S = {d: f"fuse_{rate(d, 'fuse')}" for d in ("clef", "cb")}
P = {d: f"jevrep_{rate(d)}" for d in ("clef", "cb")}
best = {d: R[d]["best_full"] for d in ("clef", "cb")}
gain = {d: f"{100 * (acc(d, 'jev_thr') - acc(d, 'svm')):.1f}%p" for d in ("clef", "cb")}
peak = max((.05, .1, .2, .3), key=lambda b: acc("cb", f"jevfuse_{b}"))

checks = [("clef n_calib", f"{M['clef']['n_calib_frontier']:,}문장"), ("cb n_calib", f"{M['cb']['n_calib_frontier']:,}문장"),
          ("clef n_test", f"{R['clef']['n_test']:,}문장"), ("cb n_test", f"{R['cb']['n_test']:,}문장"),
          ("abstract gains", f"{gain['clef']}, {gain['cb']} 높았다"),
          ("abstract rates", f"LLM 호출률은 CLEF {pct(rate('clef'))}, ClaimBuster {pct(rate('cb'))}였으며"),
          ("abstract accuracies", f"정확도({acc('clef', F['clef']):.3f}, {acc('cb', F['cb']):.3f})"),
          ("results gains", f"CLEF에서 {gain['clef']}, ClaimBuster에서 {gain['cb']} 높았다(5회 모두 유의)"),
          ("jev vs best full CLEF n_sig", f"CLEF에서는 5회 중 {ns('clef', 'jev_thr_vs_best_full')}회 유의하게 낮았으나"),
          ("clef proposed vs all-call", f"{pct(rate('clef'))} 호출로 {acc('clef', F['clef']):.3f}를 달성하여 LLM 전량 호출({acc('clef', best['clef']):.3f})"),
          ("clef svm cascade", f"{pct(rate('clef', 'fuse'))} 호출로 얻은 정확도({acc('clef', S['clef']):.3f})"),
          ("cb svm cascade", f"임베딩 SVM 결합형({pct(rate('cb', 'fuse'))} 호출, {acc('cb', S['cb']):.3f})"),
          ("cb test peak", f"{pct(peak)} 호출 시 {acc('cb', f'jevfuse_{peak}'):.3f}까지"),
          ("cb replacement drop", f"정확도가 {acc('cb', 'jevrep_0.2'):.3f}에서 {acc('cb', 'jevrep_1.0'):.3f}까지"),
          ("conclusion rates", f"LLM 호출률을 CLEF에서 {pct(rate('clef'))}, ClaimBuster에서 {pct(rate('cb'))}로"),
          ("pilot size", "학습 세트 200문장"), ("JEV fee", "1,000건당 0.044달러")]
RATES = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)
maj = {d: [b for b in RATES if ns(d, f"jevfuse_{b}_vs_jevrep_{b}") >= 3] for d in ("clef", "cb")}
rng = lambda r: f"{pct(r[0])} 이상" if r[-1] == 1.0 else f"{pct(r[0])[:-1]}~{pct(r[-1])}"
checks.append(("fusion > replacement ranges", f"CLEF {rng(maj['clef'])}, ClaimBuster {rng(maj['cb'])}의 호출률에서 5회 중 3회 이상 유의"))
# whole Table 1 rows; † = the proposed row is significantly higher on >= 3 of 5 runs
dag = lambda d, n: f"{acc(d, n):.3f}" + ("†" if f"table:{n}" in R[d]["tests"] and ns(d, f"table:{n}") >= 3 else "")
rr = lambda v: f"{pct(rate('clef', v))}/{pct(rate('cb', v))}"
for label, rt, (nc, nb) in [("임베딩 SVM", "0%", ("svm", "svm")), ("JEV (임계값 조정)", "0%", ("jev_thr", "jev_thr")),
                            ("LLM 전량 호출", "100%", ("sonnet_raw", "sonnet_raw")), ("+ 임계값 조정", "100%", ("sonnet_thr", "sonnet_thr")),
                            ("+ NN-PPI [1]", "100%", ("sonnet_nnppi", "sonnet_nnppi")), ("SVM→LLM 결합형", rr("fuse"), (S["clef"], S["cb"])),
                            ("JEV→LLM 교체형", rr("jevfuse"), (P["clef"], P["cb"])),
                            ("JEV→LLM 결합형(제안)", rr("jevfuse"), (F["clef"], F["cb"]))]:
    checks.append((f"table row {label}", f"{label} {rt} {dag('clef', nc)} {dag('cb', nb)}"))

# the wording pilot: 200 learning-set sentences, the chosen wording (nnppi) has the higher AUC on both
cal = json.load(open(os.path.join(ROOT, "data", "frontier", "calib_set.json"), encoding="utf-8"))
auc_ok, n_pilot = True, 0
for k in ("clefcal", "cbcal"):
    rows = {v: {json.loads(line)["id"]: json.loads(line)["noul"] for line in open(os.path.join(ROOT, "results", "jev", v, f"{k}.jsonl"))}
            for v in ("short", "nnppi")}
    ids = sorted(rows["short"])
    n_pilot += len(ids)
    y = [cal[k][i]["label"] for i in ids]
    auc_ok &= roc_auc_score(y, [rows["nnppi"][i] for i in ids]) > roc_auc_score(y, [rows["short"][i] for i in ids])

directional = {
    "pilot has 200 sentences and nnppi wording wins on both": n_pilot == 200 and auc_ok,
    "rates: CLEF > 0, ClaimBuster = 0": rate("clef") > 0 and rate("cb") == 0,
    "JEV > SVM on both, 5/5": all(ns(d, "jev_thr_vs_svm") == 5 and pos(d, "jev_thr_vs_svm") for d in ("clef", "cb")),
    "JEV alone < all-call on CLEF, > on CB (5/5)": neg("clef", "jev_thr_vs_best_full") and pos("cb", "jev_thr_vs_best_full")
                                                   and ns("cb", "jev_thr_vs_best_full") == 5,
    "CLEF proposed vs all-call n.s. 5/5": ns("clef", "jevfuse_sel_vs_best_full") == 0,
    "proposed vs SVM cascade n.s. on both": ns("clef", f"table:{S['clef']}") == 0 and ns("cb", f"table:{S['cb']}") == 0,
    "CB proposed > all-call 5/5": ns("cb", f"table:{best['cb']}") == 5 and pos("cb", f"table:{best['cb']}"),
    "proposed > all-call + NN-PPI 5/5 on both": all(ns(d, "jevfuse_sel_vs_sonnet_nnppi") == 5 and pos(d, "jevfuse_sel_vs_sonnet_nnppi")
                                                    for d in ("clef", "cb")),
    "fused >= replacement mean at every rate > 0": all(acc(d, f"jevfuse_{b}") >= acc(d, f"jevrep_{b}") for d in ("clef", "cb") for b in RATES),
    "CB test peak above the chosen 0% point": acc("cb", f"jevfuse_{peak}") > acc("cb", F["cb"]),
}
bad = 0
for name, s in checks:
    ok = s in text or s.replace(" ", "") in text.replace(" ", "")
    bad += not ok
    print(f"{'OK ' if ok else 'MISSING'}  {name:32s} {s}")
for name, ok in directional.items():
    bad += not ok
    print(f"{'OK ' if ok else 'FALSE  '}  {name}")
print("ALL MATCH" if bad == 0 else f"{bad} mismatches")
