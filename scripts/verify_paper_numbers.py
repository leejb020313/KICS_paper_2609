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
RUNS = len(EQ["sonnet5"]["clef"]["rows"]["fuse_0.5"]["ci_lo"])
NI, NI_HK = 0.6, 1.6  # margins fixed before the 100 runs (results/analysis/runs100_preregistration.md)
nni = lambda llm, d, k, mg: sum(l >= -mg / 100 for l in EQ[llm][d]["rows"][k]["ci_lo"])
MAJ = RUNS // 2 + 1
gain = lambda d: f"{100 * sum(EQ['sonnet5'][d]['rows']['fuse_0.5']['diff']) / RUNS:.1f}"
hdrop = lambda d: f"{100 * (HK[d]['mean_acc'][HK[d]['best_full']] - HK[d]['mean_acc']['fuse_0.5']):.1f}"
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
FL = json.load(open(os.path.join(RESULTS, "flips.json"), encoding="utf-8"))  # mechanism + positive-class F1 (scripts/flips_paper.py)
wf = lambda d, n: f"{R[d]['metrics'][n]['wf1'][0]:.3f}"
up = lambda d: sum(1 for x, p in zip(SR[d]["per_seed"][f"fuse_0.5_vs_best_full({best[d]})"]["diff"],
                                     SR[d]["per_seed"][f"fuse_0.5_vs_best_full({best[d]})"]["p"]) if p < 0.05 and x > 0)


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
           ("fuse50 vs sonnet+nnppi: n_sig", f"NN-PPI를 적용한 전량 호출보다 {ns('clef', 'fuse_0.5_vs_sonnet_nnppi')}회, {ns('cb', 'fuse_0.5_vs_sonnet_nnppi')}회 유의하게 높았다"),
           ("abstract mean gain + accs", f"평균 정확도가 {gain('clef')}~{gain('cb')}%p 높았고(CLEF {acc('clef', 'fuse_0.5'):.3f}, ClaimBuster {acc('cb', 'fuse_0.5'):.3f}), {RUNS}회 반복 중 유의하게 낮은 경우는 없었다"),
           ("abstract mechanism", "맞힌 판정을 틀리게 바꾼 경우가 절반 안팎이었다"),
           ("results mean gain + sig higher", f"평균 {gain('clef')}%p, {gain('cb')}%p 높았고 {RUNS}회 중 {up('clef')}회, {up('cb')}회 유의하게 높았으며 유의하게 낮은 경우는 없었다"),
           ("results non-inferiority counts", f"−{NI:.1f}%p 이상인 반복은 {nni('sonnet5', 'clef', 'fuse_0.5', NI)}회, {nni('sonnet5', 'cb', 'fuse_0.5', NI)}회(교체형 {nni('sonnet5', 'clef', 'replace_0.5', NI)}회, {nni('sonnet5', 'cb', 'replace_0.5', NI)}회)였고"),
           ("weighted F1", f"결합형({wf('clef', 'fuse_0.5')}, {wf('cb', 'fuse_0.5')})이 교체형({wf('clef', 'replace_0.5')}, {wf('cb', 'replace_0.5')})과 전량 호출({wf('clef', 'sonnet_thr')}, {wf('cb', 'sonnet_thr')})보다 높았다"),
           ("mechanism broke", f"맞힌 판정을 CLEF {round(FL['clef']['replace']['broke'])}개, ClaimBuster {round(FL['cb']['replace']['broke'])}개 틀리게 바꿨으나 결합형은 {round(FL['clef']['fuse']['broke'])}개, {round(FL['cb']['fuse']['broke'])}개에 그쳤고"),
           ("mechanism fixed ratio", f"바로잡은 판정 수는 교체형의 {round(100 * FL['clef']['fuse']['fixed'] / FL['clef']['replace']['fixed'])}%, {round(100 * FL['cb']['fuse']['fixed'] / FL['cb']['replace']['fixed'])}%였다"),
           ("positive-class F1 (CB)", f"F1도 {FL['cb']['f1_pos']['fuse_0.5']:.3f}로 전량 호출({FL['cb']['f1_pos']['sonnet_thr']:.3f})보다 낮았다"),
           ("haiku mean drop", f"전량 호출보다 평균 {hdrop('clef')}%p, {hdrop('cb')}%p 낮았으나"),
           ("haiku non-inferiority counts", f"하한이 −{NI_HK:.1f}%p 이상인 반복은 {nni('haiku45', 'clef', 'fuse_0.5', NI_HK)}회, {nni('haiku45', 'cb', 'fuse_0.5', NI_HK)}회(교체형 {nni('haiku45', 'clef', 'replace_0.5', NI_HK)}회, {nni('haiku45', 'cb', 'replace_0.5', NI_HK)}회)였다"),
           ("haiku vs nnppi n_sig", f"유의하게 높은 경우는 {HK['clef']['per_seed']['fuse_0.5_vs_sonnet_nnppi']['n_sig']}회, {HK['cb']['per_seed']['fuse_0.5_vs_sonnet_nnppi']['n_sig']}회뿐이었다"),
           ("runs (method)", f"80%를 비복원 추출하여 {RUNS}회 반복한 평균"),
           ("calib rate rule", f"교차검증 정확도가 {100 * max(max(cal(d, f'fuse_{b}') for b in (.6, .7, .8, .9, 1.0)) - cal(d, 'fuse_0.5') for d in ('clef', 'cb')):.1f}%p 이하로만 오르는 50%"),
           ("fig 2 caption", f"({RUNS}회 평균, 띠는 표준편차)"),
           ("bootstrap", "부트스트랩(2,000회)"),

           ("example d", f"d={EX['example']['d']:.2f}".replace("-", "−")),
           ("example s", f"s={EX['example']['s']:.2f}"), ("example LLM threshold", f"t={EX['llm_threshold']:.2f}"),
           ("example bar", f"기준이 {EX['example']['bar']:.2f}로 높아져"),
           ("svm vs nnppi: CLEF n_sig", f"NN-PPI보다 CLEF에서 {RUNS}회 중 {ns('clef', 'svm_vs_gemma_nnppi_sel')}회 유의하게 높았고, ClaimBuster에서는 유의차가 없었다"),
           ("dagger definition", "†: 과반 반복에서 제안보다 유의하게 낮음"),
           ("cost sample size", f"테스트 배치 {CT['n_calls']}개"),
           ("cost per 1,000", f"전량 호출 ${C['1.0']['usd']:.3f}, 결합형 50% ${C['0.5']['usd']:.3f}, 순차"),
           ("time per 1,000", f"순차 처리 시간은 {C['1.0']['seconds']:.0f}초, {C['0.5']['seconds']:.0f}초였다"),
           ("abstract cost/time (expected)", f"이를 통해 LLM 비용은 약 {100 * (1 - C['0.5']['usd'] / C['1.0']['usd']):.0f}%, 처리 시간은 약 "
                                             f"{100 * (1 - C['0.5']['seconds'] / C['1.0']['seconds']):.0f}% 줄일 수 있을 것으로 기대된다"),
           ("conclusion cost/time (expected)", f"이에 따라 LLM 비용은 약 {100 * (1 - C['0.5']['usd'] / C['1.0']['usd']):.0f}%, 처리 시간은 약 "
                                               f"{100 * (1 - C['0.5']['seconds'] / C['1.0']['seconds']):.0f}% 줄일 수 있을 것으로 기대된다"),
           ("conclusion recall limitation", f"분류기의 재현율이 {m('cb', 'svm', 'rec1'):.2f}에 그쳐(CLEF {m('clef', 'svm', 'rec1'):.2f})"),
           ("drift warning", f"(학습 데이터 {100 * DR['warning']['cb_learning_oof']:.1f}% → 테스트 {100 * DR['warning']['cb_test']:.1f}%, CLEF는 증가 없음)"),
           ("drift remedy", f"300문장의 라벨을 학습 세트에 추가하면 나머지의 재현율이 {100 * (DR['remedy_cb']['300']['fused']['recall'] - DR['remedy_cb']['0']['fused']['recall']):.1f}%p 올랐다"),
           ("conclusion recall numbers", f"(재현율 {m('cb', 'fuse_0.5', 'rec1'):.2f} 대 {m('cb', 'sonnet_thr', 'rec1'):.2f})")]
# the text names which full-call setting is the stronger one on each dataset
BF_NAME = {"sonnet_thr": "임계값 조정", "sonnet_raw": "조정 전"}
assert best["clef"] == best["cb"] == "sonnet_thr"
checks.append(("stronger full-call setting named", "이하 전량 호출은 정확도가 더 높았던 임계값 조정 설정을 뜻한다"))
# Table 1 daggers: a baseline cell carries † exactly when fused@50% is significantly higher on more than half of the runs
DAG = {n: f"fuse_0.5_vs_{n}" for n in ("sonnet_raw", "sonnet_thr", "sonnet_nnppi", "replace_0.5")}
mark = lambda d, n: f"{acc(d, n):.3f}" + ("†" if n in DAG and ns(d, DAG[n]) >= MAJ else "")
# whole Table 1 rows (label, call rate, CLEF, ClaimBuster), so a † can only match in its own row
for label, rate, n in [("Gemma+NN-PPI [2]", "0%", "gemma_nnppi_sel"),
                       ("임베딩 SVM", "0%", "svm"), ("LLM 전량 호출", "100%", "sonnet_raw"), ("+ 임계값 조정", "100%", "sonnet_thr"),
                       ("+ NN-PPI [2]", "100%", "sonnet_nnppi"), ("교체형 캐스케이드", "50%", "replace_0.5"),
                       ("결합형 캐스케이드 (제안)", "50%", "fuse_0.5")]:
    checks.append((f"table row {n}", f"{label} {rate} {mark('clef', n)} {mark('cb', n)}"))
gap = max(cal(d, "sonnet_thr_oof") - cal(d, "fuse_0.5") for d in ("clef", "cb"))
# claims whose direction the text asserts: "fused > replacement at every rho > 0" and "significant" / "not significant"
directional = {
    "fuse >= replace at every rho>0": all(acc(d, f"fuse_{b}") >= acc(d, f"replace_{b}") for d in ("clef", "cb")
                                          for b in (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)),
    "fuse50 never significantly below the stronger all-call (any run, both datasets)": all(
        all(x >= 0 for x, p in zip(SR[d]["per_seed"][f"fuse_0.5_vs_best_full({best[d]})"]["diff"], SR[d]["per_seed"][f"fuse_0.5_vs_best_full({best[d]})"]["p"]) if p < 0.05)
        for d in ("clef", "cb")),
    "svm vs nnppi: CB n.s. in all runs, CLEF higher": ns("cb", "svm_vs_gemma_nnppi_sel") == 0 and acc("clef", "svm") > acc("clef", "gemma_nnppi_sel"),
    "drift: CLEF disagreement does not rise; CB rises; retraining keeps accuracy":
        DR["warning"]["clef_test"] <= DR["warning"]["clef_learning_oof"] and DR["warning"]["cb_test"] > DR["warning"]["cb_learning_oof"]
        and DR["remedy_cb"]["300"]["fused"]["acc"] >= DR["remedy_cb"]["0"]["fused"]["acc"],
    "CB recall: fused50 below threshold-tuned all-call; SVM recall lower on CB than CLEF":
        m("cb", "fuse_0.5", "rec1") < m("cb", "sonnet_thr", "rec1") and m("cb", "svm", "rec1") < m("clef", "svm", "rec1"),
    "haiku: stronger all-call = raw on both": all(HK[d]["best_full"] == "sonnet_raw" for d in ("clef", "cb")),
    "haiku: fuse50 never significantly below all-call": all(HK[d]["per_seed"]["fuse_0.5_vs_best_full(sonnet_raw)"]["n_sig"] == 0 for d in ("clef", "cb")),
    "haiku: fused50 mean below all-call (text: 낮았으나)": all(HK[d]["mean_acc"]["fuse_0.5"] < HK[d]["mean_acc"]["sonnet_raw"] for d in ("clef", "cb")),
    "haiku: fuse50 vs all-call+NN-PPI significant on fewer than half the runs": all(HK[d]["per_seed"]["fuse_0.5_vs_sonnet_nnppi"]["n_sig"] < MAJ for d in ("clef", "cb")),
    "LLM raw recall below tuned recall (text: 라벨보다 엄격)": all(m(d, "sonnet_raw", "rec1") < m(d, "sonnet_thr", "rec1") for d in ("clef", "cb")),
    "fused breaks fewer correct SVM decisions than replacement, both datasets": all(FL[d]["fuse"]["broke"] < FL[d]["replace"]["broke"] for d in ("clef", "cb")),
    "abstract '절반 안팎': fused/replacement broken ratio in [0.35, 0.6]": all(0.35 <= FL[d]["fuse"]["broke"] / FL[d]["replace"]["broke"] <= 0.6 for d in ("clef", "cb")),
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
