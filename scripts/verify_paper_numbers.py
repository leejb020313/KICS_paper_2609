"""Check that every result number printed in the paper PDF matches the result files.

The paper reports the full held-out evaluation (results/final_results_full.json); only the CLEF
NN-PPI reproduction F1 is on the original
paper's dev-test split (results/final_results.json).

Usage: python scripts/verify_paper_numbers.py <paper.pdf>
"""
import json
import math
import os
import re
import sys

import pymupdf

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
R = json.load(open(os.path.join(RESULTS, "final_results_full.json"), encoding="utf-8"))
R0 = json.load(open(os.path.join(RESULTS, "final_results.json"), encoding="utf-8"))
S = json.load(open(os.path.join(RESULTS, "streaming_results_full.json"), encoding="utf-8"))
EX = json.load(open(os.path.join(RESULTS, "example_case.json"), encoding="utf-8"))
# the paper puts zero-width spaces between Hangul syllables (line-break opportunities); drop them
text = re.sub(r"\s+", " ", "".join(p.get_text() for p in pymupdf.open(sys.argv[1])).replace("​", ""))
acc = lambda d, n: R[d]["metrics"][n]["acc"][0]
m = lambda d, n, k: R[d]["metrics"][n][k][0]
cal = lambda d, n: R[d]["calib_curve"][n][0]
pv = lambda d, n: R[d]["contrasts"][n]["mcnemar_p"]
pp = lambda d, a, b: f"{100 * (acc(d, a) - acc(d, b)):.1f}%p"
best = {d: max(("sonnet_raw", "sonnet_thr"), key=lambda n: acc(d, n)) for d in ("clef", "cb")}


def pf(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def pup(pairs):
    return f"p≤{math.ceil(1000 * max(pv(d, k) for d, k in pairs)) / 1000:.3f}"


checks = [("clef n_test", f"{R['clef']['n_test']:,}문장"), ("cb n_test", f"{R['cb']['n_test']:,}문장"),
          ("clef n_calib", f"{R['clef']['n_calib_gemma']:,}문장"), ("cb n_calib", f"{R['cb']['n_calib_gemma']:,}문장"),
          ("clef NN-PPI wF1 (dev-test)", f"{R0['clef']['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}"),
          ("cb NN-PPI wF1 (2016 full)", f"{m('cb', 'gemma_nnppi_sel', 'wf1'):.3f}"),
          ("k_sel", f"k={R['clef']['k_sel']}")]
for d in ("clef", "cb"):
    for n in ["gemma_nnppi_sel", "svm", "sonnet_raw", "sonnet_thr", "sonnet_nnppi", "replace_0.5", "fuse_0.5"]:
        checks.append((f"{d} acc {n}", f"{acc(d, n):.3f}"))
    checks += [(f"{d} p svm vs nnppi", pf(pv(d, 'svm_vs_gemma_nnppi_sel'))),
               (f"{d} recall sonnet_raw", f"{m(d, 'sonnet_raw', 'rec1'):.2f}"), (f"{d} prec sonnet_raw", f"{m(d, 'sonnet_raw', 'prec1'):.2f}"),
               (f"{d} recall sonnet_thr", f"{m(d, 'sonnet_thr', 'rec1'):.2f}"),
               (f"{d} p fuse50 vs best full", ("CLEF " if d == "clef" else "ClaimBuster ") + pf(pv(d, f'fuse_0.5_vs_{best[d]}'))),
               (f"{d} diff fuse100 - fuse50", ("CLEF " if d == "clef" else "ClaimBuster ") + pp(d, 'fuse_1.0', 'fuse_0.5')),
               (f"{d} stream acc@50", f"{S[d]['acc']['stream_global_0.5'][0]:.3f}"),
               (f"{d} stream rate@50", f"{100 * S[d]['test_call_rate']['0.5'][0]:.0f}%")]
checks += [("best full-call acc clef", f"{acc('clef', best['clef']):.3f}"), ("best full-call acc cb", f"{acc('cb', best['cb']):.3f}"),
           ("p fuse50 vs sonnet+nnppi (max)", pf(max(pv(d, 'fuse_0.5_vs_sonnet_nnppi') for d in ("clef", "cb")))),
           ("cb p fuse50 vs fuse100 (seed 0)", pf(pv('cb', 'fuse_0.5_vs_fuse_1.0_seed0'))),
           ("cb raw - thr accuracy drop", pp('cb', 'sonnet_raw', 'sonnet_thr')),
           ("cb test positive share", f"{100 * sum(v['n_pos'] for v in R['cb']['by_part'].values()) / R['cb']['n_test']:.0f}%"),
           ("calib gap <=0.7pp", "0.7%p"),
           ("fuse vs replace p bound", pup([("cb", f"fuse_{b}_vs_replace_{b}") for b in (0.3, 0.4, 0.5)] +
                                           [("clef", f"fuse_{b}_vs_replace_{b}") for b in (0.2, 0.3)])),
           ("batch vs single AUC", "0.991 대 0.990"),
           ("example text (quoted prefix)", "I'm going to give them $5,000 to take with them …"), ("example d", f"d={EX['example']['d']:.2f}".replace("-", "−")),
           ("example s", f"s={EX['example']['s']:.2f}"), ("example LLM threshold", f"t={EX['llm_threshold']:.2f}"),
           ("example bar", f"기준이 {EX['example']['bar']:.2f}로 올라가"),
           ("example counts", f"결합형만 옳은 문장은 {EX['n_queried_fuse_right_replace_wrong']}개, 교체형만 옳은 문장은 "
                              f"{EX['n_queried_replace_right_fuse_wrong']}개"),
           ("calib gain after 50%", f"{100 * max(cal(d, f'fuse_{b}') - cal(d, 'fuse_0.5') for d in ('clef', 'cb') for b in (.6, .7, .8, .9, 1.0)):.1f}%p")]
# the text names which full-call setting is the stronger one on each dataset
BF_NAME = {"sonnet_thr": "임계값 조정", "sonnet_raw": "조정 전"}
checks.append(("stronger full-call setting named", f"CLEF는 {BF_NAME[best['clef']]}, ClaimBuster는 {BF_NAME[best['cb']]}"))
# Table 1 daggers: a baseline cell carries † exactly when fused@50% is significantly higher (McNemar p<0.05, seed 0)
DAG = {"sonnet_raw": "fuse_0.5_vs_sonnet_raw", "sonnet_thr": "fuse_0.5_vs_sonnet_thr",
       "sonnet_nnppi": "fuse_0.5_vs_sonnet_nnppi", "replace_0.5": "fuse_0.5_vs_replace_0.5",
       "nnppi_0.5": "fuse_0.5_vs_nnppi_0.5"}
mark = lambda d, n: f"{acc(d, n):.3f}" + ("†" if n in DAG and pv(d, DAG[n]) < 0.05 else "")
# whole Table 1 rows (label, call rate, CLEF, ClaimBuster), so a † can only match in its own row
for label, rate, n in [("Gemma 3 4B + NN-PPI [1]", "0%", "gemma_nnppi_sel"),
                       ("임베딩 SVM", "0%", "svm"), ("LLM 전량 호출", "100%", "sonnet_raw"), ("+ 임계값 조정", "100%", "sonnet_thr"),
                       ("+ NN-PPI [1]", "100%", "sonnet_nnppi"), ("교체형 캐스케이드", "50%", "replace_0.5"),
                       ("결합형 캐스케이드 (제안)", "50%", "fuse_0.5")]:
    checks.append((f"table row {n}", f"{label} {rate} {mark('clef', n)} {mark('cb', n)}"))
gap = max(cal(d, "sonnet_thr_oof") - cal(d, "fuse_0.5") for d in ("clef", "cb"))
# claims whose direction the text asserts: "fused > replacement at every rho > 0" and "significant" / "not significant"
directional = {
    "fuse >= replace at every rho>0": all(acc(d, f"fuse_{b}") >= acc(d, f"replace_{b}") for d in ("clef", "cb")
                                          for b in (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)),
    "fuse50 vs best full-call n.s.": all(pv(d, f"fuse_0.5_vs_{best[d]}") >= 0.05 for d in ("clef", "cb")),
    "fuse50 > sonnet+nnppi significant": all(pv(d, "fuse_0.5_vs_sonnet_nnppi") < 0.05 for d in ("clef", "cb")),
    "svm > nnppi significant on CLEF only": pv("clef", "svm_vs_gemma_nnppi_sel") < 0.05 <= pv("cb", "svm_vs_gemma_nnppi_sel")
                                            and acc("clef", "svm") > acc("clef", "gemma_nnppi_sel"),
    "calib gap within 0.7pp": gap <= 0.007 + 1e-9,
    "quoted pledge text is a prefix of the example": EX["example"]["text"].startswith("I'm going to give them $5,000 to take with them"),
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
