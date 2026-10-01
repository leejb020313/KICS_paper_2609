"""Check that every result number printed in the paper PDF matches the result files.

The paper reports the full held-out evaluation (results/paper/final_results_full.json); only the CLEF
NN-PPI reproduction F1 is on the original
paper's dev-test split (results/paper/final_results.json).

Usage: python scripts/verify_paper_numbers.py <paper.pdf>
"""
import json
import math
import os
import re
import sys

import pymupdf

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "paper")
R = json.load(open(os.path.join(RESULTS, "final_results_full.json"), encoding="utf-8"))
R0 = json.load(open(os.path.join(RESULTS, "final_results.json"), encoding="utf-8"))
S = json.load(open(os.path.join(RESULTS, "streaming_results_full.json"), encoding="utf-8"))
EX = json.load(open(os.path.join(RESULTS, "example_case.json"), encoding="utf-8"))
SR = json.load(open(os.path.join(RESULTS, "seed_robustness_full.json"), encoding="utf-8"))  # tests on each of the 5 runs
CT = json.load(open(os.path.join(RESULTS, "cost_time.json"), encoding="utf-8"))  # LLM cost and time per 1,000 sentences
C = CT["per_1000_sentences"]
DR = json.load(open(os.path.join(os.path.dirname(RESULTS), "analysis", "drift_remedy.json"), encoding="utf-8"))  # drift warning/remedy
HK = json.load(open(os.path.join(RESULTS, "seed_robustness_full_haiku45.json"), encoding="utf-8"))  # second LLM (Claude Haiku 4.5)
hk = lambda d, k: f"{HK[d]['mean_acc'][k]:.3f}"
EQ = json.load(open(os.path.join(RESULTS, "equivalence.json"), encoding="utf-8"))  # non-inferiority CIs
eqlo = lambda llm, d, k: min(EQ[llm][d]["rows"][k]["ci_lo"])
NI = math.ceil(-100 * min(eqlo("sonnet5", d, "fuse_0.5") for d in ("clef", "cb")) * 10) / 10
NI_HK = math.ceil(-100 * min(eqlo("haiku45", d, "fuse_0.5") for d in ("clef", "cb")) * 10) / 10
mstr = lambda x: f"{x:.1f}".replace("-", "−")
ns = lambda d, c: SR[d]["per_seed"][c]["n_sig"]
ps = lambda pairs: [p for d, c in pairs for p in SR[d]["per_seed"][c]["p"]]
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
    checks += [(f"{d} recall sonnet_raw", f"{m(d, 'sonnet_raw', 'rec1'):.2f}"), 
]
checks += [("best full-call acc clef", f"{acc('clef', best['clef']):.3f}"), ("best full-call acc cb", f"{acc('cb', best['cb']):.3f}"),
           ("fuse50 vs sonnet+nnppi: 5/5, p bound", f"5회 모두 유의하게 높았다(p≤{math.ceil(1000 * max(ps([(d, 'fuse_0.5_vs_sonnet_nnppi') for d in ('clef', 'cb')]))) / 1000:.3f})"),
           ("non-inferiority (abstract)", f"5회 반복 모두 차이의 95% 신뢰구간 하한이 −{NI:.1f}%p 이상이었다"),
           ("non-inferiority (results)", f"5회 모두 차이의 95% 신뢰구간 하한이 −{NI:.1f}%p 이상이었다. 반면"),
           ("mean diff to best all-call", "정확도가 평균 " + ", ".join(f"{100 * sum(EQ['sonnet5'][d]['rows']['fuse_0.5']['diff']) / 5:.1f}%p" for d in ("clef", "cb")) + " 높았고"),
           ("abstract mean gain", "평균적으로 오히려 " + "~".join(f"{100 * sum(EQ['sonnet5'][d]['rows']['fuse_0.5']['diff']) / 5:.1f}" for d in ("clef", "cb")) + "%p 높았고"),
           ("replacement CB lower bound", f"ClaimBuster에서 하한이 {mstr(100 * eqlo('sonnet5', 'cb', 'replace_0.5'))}%p까지"),
           ("haiku non-inferiority", f"결합형(50%)의 신뢰구간 하한은 5회 모두 −{NI_HK:.1f}%p 이상이었으나"),
           ("haiku replacement lower bound", f"CLEF에서 하한이 {mstr(100 * eqlo('haiku45', 'clef', 'replace_0.5'))}%p까지"),
           ("bootstrap", "부트스트랩(2,000회)"),
           ("reach rates (results)", f"CLEF {round(100 * SR['clef']['first_rate_reaching_best_full']['fuse'])}%, "
                                     f"ClaimBuster {round(100 * SR['cb']['first_rate_reaching_best_full']['fuse'])}%에서 이르렀다"),
           ("cb raw - thr accuracy drop", pp('cb', 'sonnet_raw', 'sonnet_thr')),
           ("example d", f"d={EX['example']['d']:.2f}".replace("-", "−")),
           ("example s", f"s={EX['example']['s']:.2f}"), ("example LLM threshold", f"t={EX['llm_threshold']:.2f}"),
           ("example bar", f"판정 기준이 {EX['example']['bar']:.2f}로 높아져"),
           ("svm vs nnppi: CLEF n_sig of 5", f"NN-PPI보다 CLEF에서 5회 중 {ns('clef', 'svm_vs_gemma_nnppi_sel')}회 유의하게 높았으며"),
           ("dagger definition", "†: 5회 중 3회 이상 제안보다 유의하게 낮음"),
           ("cost sample size", f"테스트 배치 {CT['n_calls']}개"),
           ("cost per 1,000", f"전량 호출 ${C['1.0']['usd']:.3f}, 결합형 50% ${C['0.5']['usd']:.3f}였고"),
           ("time per 1,000", f"시간은 임베딩 SVM을 포함하여 {C['1.0']['seconds']:.0f}초, {C['0.5']['seconds']:.0f}초로 추정되었다"),
           ("abstract cost/time (expected)", f"이를 통해 LLM 비용은 약 {100 * (1 - C['0.5']['usd'] / C['1.0']['usd']):.0f}%, 처리 시간은 약 "
                                             f"{100 * (1 - C['0.5']['seconds'] / C['1.0']['seconds']):.0f}% 줄일 수 있을 것으로 기대된다"),
           ("haiku nnppi", f"NN-PPI를 적용한 전량 호출({hk('clef', 'sonnet_nnppi')}, {hk('cb', 'sonnet_nnppi')})보다 유의하게 높지 않았다"),
           ("conclusion cost/time (expected)", f"이에 따라 LLM 비용은 약 {100 * (1 - C['0.5']['usd'] / C['1.0']['usd']):.0f}%, 처리 시간은 약 "
                                               f"{100 * (1 - C['0.5']['seconds'] / C['1.0']['seconds']):.0f}% 줄일 수 있을 것으로 기대된다"),
           ("conclusion recall limitation", f"분류기의 재현율이 {m('cb', 'svm', 'rec1'):.2f}에 그쳐(CLEF {m('clef', 'svm', 'rec1'):.2f})"),
           ("drift warning", f"(학습 데이터 {100 * DR['warning']['cb_learning_oof']:.1f}% → 테스트 {100 * DR['warning']['cb_test']:.1f}%, CLEF는 증가 없음)"),
           ("drift remedy", f"300문장의 라벨을 학습 세트에 추가하면 나머지 문장에서 재현율이 {100 * (DR['remedy_cb']['300']['fused']['recall'] - DR['remedy_cb']['0']['fused']['recall']):.1f}%p 올랐다"),
           ("conclusion recall numbers", f"(재현율 {m('cb', 'fuse_0.5', 'rec1'):.2f} 대 {m('cb', 'sonnet_thr', 'rec1'):.2f})")]
# the text names which full-call setting is the stronger one on each dataset
BF_NAME = {"sonnet_thr": "임계값 조정", "sonnet_raw": "조정 전"}
checks.append(("stronger full-call setting named", f"CLEF는 {BF_NAME[best['clef']]}, ClaimBuster는 {BF_NAME[best['cb']]}"))
# Table 1 daggers: a baseline cell carries † exactly when fused@50% is significantly higher on >= 3 of the 5 runs
DAG = {n: f"fuse_0.5_vs_{n}" for n in ("sonnet_raw", "sonnet_thr", "sonnet_nnppi", "replace_0.5")}
mark = lambda d, n: f"{acc(d, n):.3f}" + ("†" if n in DAG and ns(d, DAG[n]) >= 3 else "")
# whole Table 1 rows (label, call rate, CLEF, ClaimBuster), so a † can only match in its own row
for label, rate, n in [("Gemma+NN-PPI [1]", "0%", "gemma_nnppi_sel"),
                       ("임베딩 SVM", "0%", "svm"), ("LLM 전량 호출", "100%", "sonnet_raw"), ("+ 임계값 조정", "100%", "sonnet_thr"),
                       ("+ NN-PPI [1]", "100%", "sonnet_nnppi"), ("교체형 캐스케이드", "50%", "replace_0.5"),
                       ("결합형 캐스케이드 (제안)", "50%", "fuse_0.5")]:
    checks.append((f"table row {n}", f"{label} {rate} {mark('clef', n)} {mark('cb', n)}"))
gap = max(cal(d, "sonnet_thr_oof") - cal(d, "fuse_0.5") for d in ("clef", "cb"))
# claims whose direction the text asserts: "fused > replacement at every rho > 0" and "significant" / "not significant"
directional = {
    "fuse >= replace at every rho>0": all(acc(d, f"fuse_{b}") >= acc(d, f"replace_{b}") for d in ("clef", "cb")
                                          for b in (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)),
    "fuse50 vs best full-call n.s. in all 5 runs": all(ns(d, f"fuse_0.5_vs_best_full({best[d]})") == 0 for d in ("clef", "cb")),
    "fuse50 > sonnet+nnppi significant in all 5 runs": all(ns(d, "fuse_0.5_vs_sonnet_nnppi") == 5 for d in ("clef", "cb")),
    "svm vs nnppi: CB n.s. in all 5 runs, CLEF higher": ns("cb", "svm_vs_gemma_nnppi_sel") == 0 and acc("clef", "svm") > acc("clef", "gemma_nnppi_sel"),
    "replacement first reaches all-call accuracy at 50%": all(SR[d]["first_rate_reaching_best_full"]["replace"] == 0.5 for d in ("clef", "cb")),
    "calib gap within 0.7pp": gap <= 0.007 + 1e-9,
    "drift: CLEF disagreement does not rise; CB rises; retraining keeps accuracy":
        DR["warning"]["clef_test"] <= DR["warning"]["clef_learning_oof"] and DR["warning"]["cb_test"] > DR["warning"]["cb_learning_oof"]
        and DR["remedy_cb"]["300"]["fused"]["acc"] >= DR["remedy_cb"]["0"]["fused"]["acc"],
    "CB recall: fused50 below threshold-tuned all-call; SVM recall lower on CB than CLEF":
        m("cb", "fuse_0.5", "rec1") < m("cb", "sonnet_thr", "rec1") and m("cb", "svm", "rec1") < m("clef", "svm", "rec1"),
    "haiku: stronger all-call = raw on both": all(HK[d]["best_full"] == "sonnet_raw" for d in ("clef", "cb")),
    "haiku: fuse50 vs all-call n.s. in all 5 runs": all(HK[d]["per_seed"]["fuse_0.5_vs_best_full(sonnet_raw)"]["n_sig"] == 0 for d in ("clef", "cb")),
    "fused50 non-inferior at NI in all 5 runs, both datasets": all(l > -NI / 100 for d in ("clef", "cb") for l in EQ["sonnet5"][d]["rows"]["fuse_0.5"]["ci_lo"]),
    "replacement50 non-inferior at NI on CLEF (conclusion names ClaimBuster only)": all(l > -NI / 100 for l in EQ["sonnet5"]["clef"]["rows"]["replace_0.5"]["ci_lo"]),
    "replacement50 not non-inferior at NI on CB in some run": any(l <= -NI / 100 for l in EQ["sonnet5"]["cb"]["rows"]["replace_0.5"]["ci_lo"]),
    "haiku fused50 non-inferior at NI_HK in all runs": all(l > -NI_HK / 100 for d in ("clef", "cb") for l in EQ["haiku45"][d]["rows"]["fuse_0.5"]["ci_lo"]),
    "haiku replacement50 not non-inferior at NI_HK on CLEF": any(l <= -NI_HK / 100 for l in EQ["haiku45"]["clef"]["rows"]["replace_0.5"]["ci_lo"]),
    "haiku: replace50 below all-call on CLEF >= 3/5, all negative": HK["clef"]["per_seed"]["replace_0.5_vs_best_full(sonnet_raw)"]["n_sig"] >= 3
        and all(x < 0 for x in HK["clef"]["per_seed"]["replace_0.5_vs_best_full(sonnet_raw)"]["diff"]),
    "haiku: fuse50 vs all-call+NN-PPI not significant on a majority": all(HK[d]["per_seed"]["fuse_0.5_vs_sonnet_nnppi"]["n_sig"] < 3 for d in ("clef", "cb")),
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
