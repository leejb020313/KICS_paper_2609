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
SR = json.load(open(os.path.join(RESULTS, "seed_robustness_full.json"), encoding="utf-8"))  # tests on each of the 5 runs
CT = json.load(open(os.path.join(RESULTS, "cost_time.json"), encoding="utf-8"))  # LLM cost and time per 1,000 sentences
C = CT["per_1000_sentences"]
HK = json.load(open(os.path.join(RESULTS, "seed_robustness_full_haiku.json"), encoding="utf-8"))  # second LLM (Claude Haiku 4.5)
hk = lambda d, k: f"{HK[d]['mean_acc'][k]:.3f}"
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
    checks += [(f"{d} recall sonnet_raw", f"{m(d, 'sonnet_raw', 'rec1'):.2f}"), (f"{d} prec sonnet_raw", f"{m(d, 'sonnet_raw', 'prec1'):.2f}"),
               (f"{d} recall sonnet_thr", f"{m(d, 'sonnet_thr', 'rec1'):.2f}"),
               (f"{d} diff fuse100 - fuse50", ("CLEF " if d == "clef" else "ClaimBuster ") + pp(d, 'fuse_1.0', 'fuse_0.5')),
]
checks += [("best full-call acc clef", f"{acc('clef', best['clef']):.3f}"), ("best full-call acc cb", f"{acc('cb', best['cb']):.3f}"),
           ("svm vs nnppi: CLEF n_sig of 5", f"CLEF에서 5회 중 {ns('clef', 'svm_vs_gemma_nnppi_sel')}회 유의하게 높았고"),
           ("fuse50 vs sonnet+nnppi: 5/5, p bound", f"5회 모두 유의하게 높다(p≤{math.ceil(1000 * max(ps([(d, 'fuse_0.5_vs_sonnet_nnppi') for d in ('clef', 'cb')]))) / 1000:.3f})"),
           ("fuse50 vs best full: n.s. in all runs, p bound", "5회 모두 유의한 차이가 검출되지 않았으며(p≥" + f"{math.floor(100 * min(ps([(d, f'fuse_0.5_vs_best_full({best[d]})') for d in ('clef', 'cb')]))) / 100:.2f})"),
           ("reach rates (abstract)", f"교체형(50%)보다 적은 {round(100 * SR['clef']['first_rate_reaching_best_full']['fuse'])}~"
                                      f"{round(100 * SR['cb']['first_rate_reaching_best_full']['fuse'])}%의 호출률"),
           ("reach rates (results)", f"CLEF {round(100 * SR['clef']['first_rate_reaching_best_full']['fuse'])}%, "
                                     f"ClaimBuster {round(100 * SR['cb']['first_rate_reaching_best_full']['fuse'])}%의 호출률로 도달"),
           ("replacement gap at 50%", f"CLEF {100 * SR['clef']['gap_to_best_full']['0.5']['replace']:+.2f}%p, "
                                      f"ClaimBuster {100 * SR['cb']['gap_to_best_full']['0.5']['replace']:+.2f}%p"),
           ("cb raw - thr accuracy drop", pp('cb', 'sonnet_raw', 'sonnet_thr')),
           ("calib gap <=0.7pp", "0.7%p"),
           ("batch vs single AUC", "0.991 대 0.990"),
           ("example d", f"d={EX['example']['d']:.2f}".replace("-", "−")),
           ("example s", f"s={EX['example']['s']:.2f}"), ("example LLM threshold", f"t={EX['llm_threshold']:.2f}"),
           ("example bar", f"판정 기준이 {EX['example']['bar']:.2f}로 높아져"),
           ("example counts", f"결합형만 옳게 판정한 문장은 {EX['n_queried_fuse_right_replace_wrong']}개, 교체형만 옳게 판정한 문장은 "
                              f"{EX['n_queried_replace_right_fuse_wrong']}개"),
           ("calib gain after 50%", "50% 이후 증가 " + ", ".join(
               f"{n} {100 * max(cal(d, f'fuse_{b}') - cal(d, 'fuse_0.5') for b in (.6, .7, .8, .9, 1.0)):.2f}%p"
               for d, n in (("clef", "CLEF"), ("cb", "ClaimBuster")))),
           ("dagger definition", "†: 제안 대비 5회 중 3회 이상 p<0.05"),
           ("svm ms per sentence", f"문장당 약 {CT['svm_ms_per_sentence']:.0f} ms"),
           ("cost sample size", f"테스트 배치 {CT['n_calls']}개"),
           ("cost per 1,000", f"전량 호출 ${C['1.0']['usd']:.3f}, 결합형 50% ${C['0.5']['usd']:.3f}, 30% ${C['0.3']['usd']:.3f}"),
           ("time per 1,000", f"각각 {C['1.0']['seconds']:.0f}초, {C['0.5']['seconds']:.0f}초, {C['0.3']['seconds']:.0f}초"),
           ("abstract cost/time cut", f"LLM 비용은 {100 * (1 - C['0.5']['usd'] / C['1.0']['usd']):.0f}%, 처리 시간은 "
                                      f"{100 * (1 - C['0.5']['seconds'] / C['1.0']['seconds']):.0f}% 줄었다"),
           ("haiku sentence", f"전량 호출({hk('clef', 'sonnet_raw')}, {hk('cb', 'sonnet_raw')})과 5회 모두 유의한 차이가 검출되지 "
                              f"않았으나({hk('clef', 'fuse_0.5')}, {hk('cb', 'fuse_0.5')}), 교체형({hk('clef', 'replace_0.5')}, "
                              f"{hk('cb', 'replace_0.5')})은 CLEF에서 5회 중 {HK['clef']['per_seed']['replace_0.5_vs_best_full(sonnet_raw)']['n_sig']}회"),
           ("haiku nnppi", f"NN-PPI를 적용한 전량 호출({hk('clef', 'sonnet_nnppi')}, {hk('cb', 'sonnet_nnppi')})보다 유의하게 높지는 않았다"),
           ("conclusion time cut", f"처리 시간을 {100 * (1 - C['0.5']['seconds'] / C['1.0']['seconds']):.0f}% 줄이면서도")]
# the text names which full-call setting is the stronger one on each dataset
BF_NAME = {"sonnet_thr": "임계값 조정", "sonnet_raw": "조정 전"}
checks.append(("stronger full-call setting named", f"CLEF는 {BF_NAME[best['clef']]}, ClaimBuster는 {BF_NAME[best['cb']]}"))
# Table 1 daggers: a baseline cell carries † exactly when fused@50% is significantly higher on >= 3 of the 5 runs
DAG = {n: f"fuse_0.5_vs_{n}" for n in ("sonnet_raw", "sonnet_thr", "sonnet_nnppi", "replace_0.5")}
mark = lambda d, n: f"{acc(d, n):.3f}" + ("†" if n in DAG and ns(d, DAG[n]) >= 3 else "")
# fusion beats replacement on >= 3 of 5 runs exactly over the call-rate ranges the text names
RATES = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)
maj = {d: [b for b in RATES if ns(d, f"fuse_{b}_vs_replace_{b}") >= 3] for d in ("clef", "cb")}
rng = lambda r: f"{round(100 * r[0])}% 이상" if r[-1] == 1.0 else f"{round(100 * r[0])}~{round(100 * r[-1])}%"
checks.append(("fuse > replace majority ranges", f"CLEF {rng(maj['clef'])}, ClaimBuster {rng(maj['cb'])}의 호출률에서 5회 중 3회 이상 유의"))
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
    "fuse50 vs best full-call n.s. in all 5 runs": all(ns(d, f"fuse_0.5_vs_best_full({best[d]})") == 0 for d in ("clef", "cb")),
    "fuse50 > sonnet+nnppi significant in all 5 runs": all(ns(d, "fuse_0.5_vs_sonnet_nnppi") == 5 for d in ("clef", "cb")),
    "cb fuse100 > fuse50 significant in all 5 runs": ns("cb", "fuse_0.5_vs_fuse_1.0") == 5,
    "svm vs nnppi: CB n.s. in all 5 runs, CLEF higher": ns("cb", "svm_vs_gemma_nnppi_sel") == 0 and acc("clef", "svm") > acc("clef", "gemma_nnppi_sel"),
    "replacement first reaches all-call accuracy at 50%": all(SR[d]["first_rate_reaching_best_full"]["replace"] == 0.5 for d in ("clef", "cb")),
    "calib gap within 0.7pp": gap <= 0.007 + 1e-9,
    "haiku: stronger all-call = raw on both": all(HK[d]["best_full"] == "sonnet_raw" for d in ("clef", "cb")),
    "haiku: fuse50 vs all-call n.s. in all 5 runs": all(HK[d]["per_seed"]["fuse_0.5_vs_best_full(sonnet_raw)"]["n_sig"] == 0 for d in ("clef", "cb")),
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
