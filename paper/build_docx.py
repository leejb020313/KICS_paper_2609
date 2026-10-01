# -*- coding: utf-8 -*-
"""Build the KICS 2026 2-page draft (paper/cascade_kics_draft.docx).

Every number is read from the full held-out evaluation, results/paper/final_results_full.json
(scripts/final_eval.py --full), except
the CLEF NN-PPI reproduction F1, which is on the original paper's dev-test split (results/paper/final_results.json);
Figure 1 is figures/method_diagram.pdf (TikZ) and Figure 2 figures/cascade_budget_full.png (scripts/make_figure.py --full). Math is LaTeX rendered to PNG
by equations/render_equations.py.

    uv run --isolated --no-project --with python-docx --with pymupdf python paper/build_docx.py

Convert to PDF with Word (File > Save As > PDF), then check it:
    uv run --locked python scripts/verify_paper_numbers.py paper/cascade_kics_draft.pdf
    uv run --no-project --with pymupdf python paper/references/verify_references.py paper/cascade_kics_draft.pdf --online
"""
import json
import math
import os
import re

import pymupdf
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = json.load(open(os.path.join(ROOT, "results", "paper", "final_results_full.json"), encoding="utf-8"))
R_ORIG = json.load(open(os.path.join(ROOT, "results", "paper", "final_results.json"), encoding="utf-8"))
IMP = json.load(open(os.path.join(ROOT, "results", "paper", "streaming_results_full.json"), encoding="utf-8"))
# significance on each of the 5 runs, and the call rate at which each cascade reaches the all-call LLM (scripts/seed_robustness.py)
SR = json.load(open(os.path.join(ROOT, "results", "paper", "seed_robustness_full.json"), encoding="utf-8"))
# LLM cost (list price) and sequential processing time per 1,000 sentences: scripts/cost_time.py
COST_RAW = json.load(open(os.path.join(ROOT, "results", "paper", "cost_time.json"), encoding="utf-8"))
COST = COST_RAW["per_1000_sentences"]
# second LLM (Claude Haiku 4.5, same prompt and batches; "sonnet_*" keys mean that LLM): scripts/seed_robustness.py, CWC_LLM=haiku45
HK = json.load(open(os.path.join(ROOT, "results", "paper", "seed_robustness_full_haiku45.json"), encoding="utf-8"))
# non-inferiority: per-run 95% paired bootstrap CI of (cascade - stronger all-call setting): scripts/equivalence.py
EQ = json.load(open(os.path.join(ROOT, "results", "paper", "equivalence.json"), encoding="utf-8"))
import math  # noqa: E402
lo = lambda llm, d, k: min(EQ[llm][d]["rows"][k]["ci_lo"])  # worst lower bound over the 5 runs
NI = math.ceil(-100 * min(lo("sonnet5", d, "fuse_0.5") for d in ("clef", "cb")) * 10) / 10  # margin printed, %p
NI_HK = math.ceil(-100 * min(lo("haiku45", d, "fuse_0.5") for d in ("clef", "cb")) * 10) / 10
assert sum(l > -NI / 100 for l in EQ["sonnet5"]["cb"]["rows"]["replace_0.5"]["ci_lo"]) < 5  # replacement fails on CB
assert sum(l > -NI_HK / 100 for l in EQ["haiku45"]["clef"]["rows"]["replace_0.5"]["ci_lo"]) < 5
MAJ = 3  # "5회 중 3회 이상 유의" = significant on a majority of the runs


def nsig(ds, claim):
    return SR[ds]["per_seed"][claim]["n_sig"]


def p_all(pairs, lower):
    """Bound over every run of the given (dataset, claim) pairs: "p≥" (floored, 2 dp) or "p≤" (ceiled, 3 dp)."""
    ps = [p for ds, c in pairs for p in SR[ds]["per_seed"][c]["p"]]
    return f"p≥{math.floor(100 * min(ps)) / 100:.2f}" if lower else f"p≤{math.ceil(1000 * max(ps)) / 1000:.3f}"


def sig_range(ds):
    """Call rates at which fusion beats replacement on a majority of runs, as "20~30%" or "20% 이상"."""
    rates = [b for b in (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0) if nsig(ds, f"fuse_{b}_vs_replace_{b}") >= MAJ]
    lo, hi = round(100 * rates[0]), round(100 * rates[-1])
    all_rates = [b for b in (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0) if lo <= 100 * b <= hi]
    assert len(all_rates) == len(rates), "significant call rates are not contiguous"
    return f"{lo}% 이상" if hi == 100 else f"{lo}~{hi}%"
BODY_FONT = "HY신명조"  # the KICS template's 한양신명조 (installed as HY신명조, H2MJSM.TTF)
# layout of the official KICS Word template (paper/template/kics_sample_word.doc, conf.kics.or.kr/2026f):
# title block 1.5 cm margins; two-column body T/B/L 2 cm, R 1.5 cm, column gap 0.75 cm; body and references 9 pt


def acc(ds, name):
    return R[ds]["metrics"][name]["acc"][0]


def f3(x):
    return f"{x:.3f}"


def ro(num):
    """Korean instrumental particle after a number read digit-by-digit: 으로 after a final 0/3/6 (영/삼/육), else 로."""
    return f"{num}으로" if num[-1] in "036" else f"{num}로"


def eul(num):
    """Korean object particle after a number read digit-by-digit: 을 after a final 0/1/3/6/7/8 (batchim), else 를."""
    return f"{num}을" if num[-1] in "013678" else f"{num}를"


def pp(ds, a, b):
    """Accuracy difference a - b in percentage points, one decimal."""
    return f"{100 * (acc(ds, a) - acc(ds, b)):.1f}"


def cal(ds, name):
    return R[ds]["calib_curve"][name][0]


def pval(ds, key):
    return R[ds]["contrasts"][key]["mcnemar_p"]


def pfmt(p):
    """McNemar p as printed: p<0.001 below that, three decimals below 0.1, otherwise two."""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def best_full(ds):
    """The stronger of the two full-call settings on this dataset (raw 0.5 cut-off or calib-tuned threshold)."""
    return max(("sonnet_raw", "sonnet_thr"), key=lambda n: acc(ds, n))


LATIN_FONT = BODY_FONT  # template: Latin letters and digits are 신명조 too (was Times New Roman)


def set_font(run, name=BODY_FONT, size=9.5, bold=False, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = LATIN_FONT if name == BODY_FONT else name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name)


def centered(doc, text, size=10, bold=False, font=BODY_FONT, after=4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(after)
    set_font(p.add_run(text), font, size, bold)
    return p


def heading(doc, text, size=9):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(4.75)
    p.paragraph_format.keep_with_next = True  # never leave a heading alone at the bottom of a column
    set_font(p.add_run(text), BODY_FONT, size, bold=True)


def subheading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), BODY_FONT, 9, bold=True)


def figure(doc, png, width_cm, caption, detail=""):
    """Figure with a bold caption lead (number + the finding) and an optional plain-text detail."""
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.space_before = Pt(3)
    pic.paragraph_format.space_after = Pt(1)
    pic.paragraph_format.keep_with_next = True
    pic.add_run().add_picture(png, width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if detail else WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(5)
    set_font(cap.add_run(caption), BODY_FONT, 8, bold=True)
    if detail:
        set_font(cap.add_run(" " + detail), BODY_FONT, 8)


BODY_LINE_SPACING = 1.0  # KICS template default


def body(doc, text, size=9, indent=0.3175, after=3.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, lead=None):
    """Body paragraph; `lead` is a bold run-in sentence that states the paragraph's finding."""
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.line_spacing = BODY_LINE_SPACING
    if indent:
        pf.first_line_indent = Cm(indent)
    if lead:
        set_font(p.add_run(lead + " "), BODY_FONT, size, bold=True)
    set_font(p.add_run(text), BODY_FONT, size)
    return p


def add_math(run, name, scale=0.9):
    """Insert equations/<name>.png at the natural size of its LaTeX PDF (10pt math scaled to the 9pt body)."""
    rect = pymupdf.open(os.path.join(HERE, "equations", name + ".pdf"))[0].rect
    run.add_picture(os.path.join(HERE, "equations", name + ".png"), width=Pt(rect.width * scale))
    # w:position is in half-points: drop the image by the depth below the LaTeX baseline (~1/4 of its height)
    run._element.get_or_add_rPr().append(OxmlElement("w:position"))
    run._element.rPr[-1].set(qn("w:val"), str(-round(rect.height * scale * 0.25 * 2)))


def columns(doc, num):
    sectPr = doc.sections[-1]._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(0.75 * 567)))


doc = Document()
for s in doc.sections:
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Cm(1.5)

# ---------------- title block ----------------
centered(doc, "저비용 분류기와 대형 언어 모델의 선택적 결합을 통한", 13, True, after=0)
centered(doc, "팩트체크 필요성 탐지", 13, True, after=0)
# title block as in the peer KICS papers: authors with the corresponding author starred, affiliation, e-mails
AUTHORS_KO, AUTHORS_EN = "이정빈, 김은경*", "Jeongbin Lee, Eunkyung Kim*"
AFFIL_KO, AFFIL_EN = "국립한밭대학교", "Hanbat National Univ."
EMAILS = "20221065@edu.hanbat.ac.kr, *ekim@hanbat.ac.kr"
centered(doc, AUTHORS_KO, 11, after=0).paragraph_format.space_before = Pt(7.6)
centered(doc, AFFIL_KO, 11, after=0)
centered(doc, EMAILS, 11, after=7.6)
centered(doc, "Cost-Efficient Check-Worthy Claim Detection via Selective Fusion\nof a Low-Cost Classifier and a Large Language Model",
         13, True, after=0)
centered(doc, AUTHORS_EN, 11, after=0).paragraph_format.space_before = Pt(14.2)
centered(doc, AFFIL_EN, 11, after=14.2)

cb, cl = "cb", "clef"
BF = {d: best_full(d) for d in (cl, cb)}
BF_NAME = {"sonnet_thr": "임계값 조정", "sonnet_raw": "조정 전"}
# first call rate whose mean accuracy reaches the stronger all-call LLM setting
REACH = {d: round(100 * SR[d]["first_rate_reaching_best_full"]["fuse"]) for d in (cl, cb)}
assert all(SR[d]["first_rate_reaching_best_full"]["replace"] == 0.5 for d in (cl, cb))  # text: "교체형은 50%"
assert REACH[cl] <= REACH[cb]  # abstract prints the range as CLEF~ClaimBuster
centered(doc, "요 약", 11, False, after=3)
MEAN_GAIN = {d: 100 * sum(EQ["sonnet5"][d]["rows"]["fuse_0.5"]["diff"]) / 5 for d in (cl, cb)}
assert all(v > 0 for v in MEAN_GAIN.values())  # "평균적으로 오히려 … 높았고"
CUT_USD = 100 * (1 - COST["0.5"]["usd"] / COST["1.0"]["usd"])
CUT_SEC = 100 * (1 - COST["0.5"]["seconds"] / COST["1.0"]["seconds"])
rec = lambda d, k: R[d]["metrics"][k]["rec1"][0]
# drift analysis (results/analysis/drift_remedy.json, scripts/drift_remedy.py): label-free warning and retraining
_DR = json.load(open(os.path.join(ROOT, "results", "analysis", "drift_remedy.json"), encoding="utf-8"))
DR = {"warn_cb_learn": 100 * _DR["warning"]["cb_learning_oof"], "warn_cb_test": 100 * _DR["warning"]["cb_test"], "k": 300,
      "gain": 100 * (_DR["remedy_cb"]["300"]["fused"]["recall"] - _DR["remedy_cb"]["0"]["fused"]["recall"])}
assert _DR["warning"]["clef_test"] <= _DR["warning"]["clef_learning_oof"]  # "CLEF는 증가 없음"
assert _DR["remedy_cb"]["300"]["fused"]["acc"] >= _DR["remedy_cb"]["0"]["fused"]["acc"]  # retraining does not cost accuracy
REC = {"cb_svm": f"{rec(cb, 'svm'):.2f}", "cl_svm": f"{rec(cl, 'svm'):.2f}",
       "cb_fuse": f"{rec(cb, 'fuse_0.5'):.2f}", "cb_thr": f"{rec(cb, 'sonnet_thr'):.2f}"}
assert rec(cb, "fuse_0.5") < rec(cb, "sonnet_thr") and rec(cb, "svm") < rec(cl, "svm")  # "많이 놓쳤다", "그쳐"
assert BF[cb] == "sonnet_raw" and rec(cb, "sonnet_thr") > rec(cb, "sonnet_raw")  # compared with the tuned (higher-recall) all-call
body(doc, (
    "팩트체크 필요성 탐지(check-worthiness detection)는 수많은 문장 중 사실 확인이 필요한 문장을 골라내는 기술이다. "
    "대형 언어 모델(LLM)은 이 판정을 잘 수행하지만, 모든 문장에 LLM을 호출하면 비용과 처리 시간이 크게 늘어난다. "
    "본 논문에서는 가벼운 저비용 분류기가 모든 문장을 먼저 판정하고, 분류기가 확신하지 못하는 문장에만 LLM을 호출하되 "
    "분류기의 판정을 LLM의 판정으로 교체하지 않고 두 모델의 점수를 결합하여 최종 판정하는 결합형 캐스케이드를 제안한다. 두 공개 데이터셋(CLEF 2024, ClaimBuster)에서 평가한 결과, 제안 방법은 LLM 호출을 절반으로 줄이면서도 모든 "
    f"문장에 LLM을 호출한 경우와 같은 수준의 정확도(CLEF {f3(acc(cl,'fuse_0.5'))}, ClaimBuster {f3(acc(cb,'fuse_0.5'))})를 "
    f"유지하였다. 정확도는 평균적으로 오히려 {MEAN_GAIN[cl]:.1f}~{MEAN_GAIN[cb]:.1f}%p 높았고, 통계적으로도 {NI:.1f}%p 넘게 "
    f"낮지 않음을 확인하였다. 이를 통해 LLM 비용은 약 {CUT_USD:.0f}%, 처리 시간은 약 {CUT_SEC:.0f}% 줄일 수 있을 것으로 "
    "기대된다."
), size=9, indent=0.5, after=2.5)

doc.add_section(0)
_body = doc.sections[-1]
_body.top_margin = _body.bottom_margin = _body.left_margin = Cm(2.0)
_body.right_margin = Cm(1.5)
columns(doc, 2)

# ---------------- I. 서론 ----------------
heading(doc, "Ⅰ. 서 론")
body(doc, (
    "온라인 허위 정보가 급증하면서 팩트체크의 중요성이 커졌으며, 그 첫 단계는 사실 확인이 필요한 주장을 찾아내는 "
    "팩트체크 필요성 탐지이다[3]. 이 단계는 들어오는 모든 문장을 판정해야 하므로 처리할 문장이 매우 많다. 대형 언어 모델(LLM)은 이 판정을 "
    "잘 수행하지만, 실제 팩트체크 서비스 운영사는 모든 문장에 대형 LLM을 호출하는 것은 운영 규모에서 감당하기 "
    "어렵다고 보고하였다[1]. 같은 연구진은 소형 인코더가 확신하지 못하는 문장만 LLM에 넘기는 방식을 향후 과제로 "
    "제시하였다[2]."
))
body(doc, (
    "이처럼 가벼운 모델이 먼저 판정하고 확신하지 못하는 입력만 큰 모델에 넘기는 구조를 캐스케이드라 하며, 일반적으로 "
    "넘긴 입력의 판정은 큰 모델의 판정으로 교체한다[5]. 그러나 어떤 문장을 사실 확인이 필요하다고 볼지에 대한 LLM의 "
    "기준은 데이터셋의 라벨(사람이 붙인 정답)과 다를 수 있으며, 이 경우 교체는 분류기가 이미 맞힌 판정까지 틀리게 바꿀 수 "
    "있다[5]. 본 논문에서는 저비용 분류기가 확신하지 못하는 문장에만 LLM을 호출하되, LLM의 판정으로 교체하지 않고 두 모델의 "
    "점수를 결합하여 최종 판정하는 결합형 캐스케이드를 제안한다. 결합 가중치는 라벨로 학습하므로 LLM의 "
    "기준과 라벨의 차이를 보정한다."
))

# ---------------- II. 본론 ----------------
heading(doc, "Ⅱ. 본 론")
subheading(doc, "2.1 선행 연구")
body(doc, (
    "NN-PPI[1]는 소형 LLM이 출력한 점수를, 라벨이 있는 보정 세트에서 의미적으로 가까운 이웃 문장들의 잔차로 보정한다. "
    "넘긴 입력에서만 두 모델의 출력을 결합하는 캐스케이드는 일반 분류 과제에서 제안되었고[6], SRR[5]은 옳은 답을 뒤집는 손실까지 예측하여 넘길 입력을 고르지만 넘긴 입력은 큰 "
    "모델의 답으로 교체한다. 본 논문은 결합 가중치를 라벨로 학습해 LLM의 기준 차이를 반영한다."
))
subheading(doc, "2.2 전체 구조")
body(doc, (
    "제안 방법(그림 1)에서는 저비용 분류기가 모든 문장을 먼저 판정하고, 분류기가 가장 불확실한 문장, 즉 "
    "결정값의 절댓값 |d(x)|가 가장 작은 문장부터 전체의 ρ(호출률)만큼 골라 LLM을 호출한다. 호출하지 않은 문장은 분류기의 "
    "판정(d(x)>0)을 그대로 사용하고, 호출한 문장은 2.4절의 결합 판정으로 최종 판정한다."
))
# figures/method_diagram.tex, compiled at its printed size (Tectonic) and inserted at that size
_dg = pymupdf.open(os.path.join(ROOT, "figures", "method_diagram.pdf"))[0].rect
figure(doc, os.path.join(ROOT, "figures", "method_diagram.png"), _dg.width / 72 * 2.54, "그림 1. 결합형 캐스케이드의 전체 구조")
subheading(doc, "2.3 저비용 분류기")
body(doc, (
    "문장 x의 팩트체크 필요 여부 y∈{0,1}을 판정하며, 라벨이 있는 학습 세트 L(NN-PPI의 보정 세트)을 사용한다. 저비용 "
    "분류기(이하 임베딩 SVM)는 NN-PPI가 이웃 검색에 쓰는 문장 임베딩 all-MiniLM-L6-v2를 입력으로 하는 "
    "RBF-SVM이다. 클래스 불균형을 고려한 가중치로 L에서 학습하며, 결정값 d(x)가 양수이면 팩트체크가 필요하다는 판정이고 절댓값이 "
    "클수록 확신이 높다."
))
subheading(doc, "2.4 결합 판정")
body(doc, "호출한 문장에 대해 LLM이 출력한 점수 s(x)∈[0,1]와 분류기의 결정값 d(x)를 다음과 같이 결합한다.")
eq = doc.add_paragraph()
eq.paragraph_format.space_before = Pt(3)
eq.paragraph_format.space_after = Pt(3)
col_w = (21.0 - 2.0 - 1.5 - 0.75) / 2  # one text column (cm)
eq.paragraph_format.tab_stops.add_tab_stop(Cm(col_w / 2), WD_TAB_ALIGNMENT.CENTER)
eq.paragraph_format.tab_stops.add_tab_stop(Cm(col_w), WD_TAB_ALIGNMENT.RIGHT)
eq.add_run("\t")
add_math(eq.add_run(), "eq1")
set_font(eq.add_run("\t(1)"), BODY_FONT, 9)
p = body(doc, "여기서 (a, b, c)는 L에서 5겹 교차적합으로 얻은 d와 s로 학습한 로지스틱 회귀 계수, σ는 시그모이드 함수, ")
p.paragraph_format.widow_control = False  # may split across the column so its bottom is not left empty
add_math(p.add_run(), "indicator")
set_font(p.add_run(
    "는 조건이 참이면 1인 지시함수이다. 식 (1)을 s에 대해 정리하면 결합형은 s를 (−c−a·d)/b와 비교하므로, 분류기가 "
    "‘불필요’로 기울수록(d가 작을수록) LLM에 더 높은 점수를 요구한다. 반면 교체형 캐스케이드(이하 교체형)는 s를 "
    "L에서 정한 고정 임계값 t와 비교한다."), BODY_FONT, 9)

# ---------------- III. 실험 ----------------
heading(doc, "Ⅲ. 실 험")
subheading(doc, "3.1 실험 환경")
assert R[cl]["k_sel"] == R[cb]["k_sel"]  # the text says both datasets selected the same k
body(doc, (
    f"CLEF 2024 CheckThat! Task 1 영어 데이터[3]는 {R[cl]['n_calib_gemma']:,}문장을 학습 세트로, dev·dev-test·공식 test를 "
    f"합친 {R[cl]['n_test']:,}문장을 테스트로 사용하였다. ClaimBuster[4]는 2012년 토론 {R[cb]['n_calib_gemma']:,}문장과 "
    f"2016년 토론 {R[cb]['n_test']:,}문장을 각각 학습 세트와 테스트로 사용하였다. NN-PPI는 원 논문과 같이 Gemma 3 4B로 "
    "재현하였으며, 원 논문과 같은 분할에서 가중 F1은 "
    f"CLEF {R_ORIG[cl]['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}, "
    f"ClaimBuster {R[cb]['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}였다(원 논문 0.827, 0.760). LLM은 Claude Sonnet 5를 "
    "사용하였고, NN-PPI의 판정 기준 프롬프트로 40문장씩 묶어 채점하였다."
))
subheading(doc, "3.2 평가 방법")
CAL_GAIN = {d: max(cal(d, f'fuse_{b}') - cal(d, 'fuse_0.5') for b in (.6, .7, .8, .9, 1.0)) for d in (cl, cb)}
assert max(cal(d, 'sonnet_thr_oof') - cal(d, 'fuse_0.5') for d in (cl, cb)) <= 0.007 + 1e-9  # "0.7%p 이내"
body(doc, (
    "평가 지표는 테스트 정확도이며, 학습 세트의 80%를 비복원 추출하여 5회 반복한 평균을 보고한다. 방법 간 차이는 5회 "
    "각각의 예측에 McNemar 검정(α=0.05, 다중 비교 보정 없음)을 적용하고, 차이의 95% 신뢰구간은 테스트 문장 부트스트랩(2,000회)으로 구하였다. LLM 임계값, 결합 계수, 호출률, NN-PPI의 이웃 수 k는 모두 학습 세트 "
    f"안에서만 정하였고(두 데이터셋 모두 k={R[cl]['k_sel']}), 임베딩 SVM은 기본 하이퍼파라미터를 사용하였다. 호출률은 학습 "
    "세트의 교차검증 정확도가 두 데이터셋 모두 거의 오르지 않는 50%로 정하였다."
))

subheading(doc, "3.3 실험 결과")
DAG = {n: f"fuse_0.5_vs_{n}" for n in ("sonnet_raw", "sonnet_thr", "sonnet_nnppi", "replace_0.5")}


def cell(ds, name):
    """Accuracy as printed in Table 1; † when the fused cascade at 50% is significantly higher on a majority of the 5 runs."""
    dag = name in DAG and nsig(ds, DAG[name]) >= MAJ
    assert not dag or acc(ds, "fuse_0.5") > acc(ds, name)  # † means "lower than the proposed method"
    return f3(acc(ds, name)) + ("†" if dag else "")


def rule(tc, side, sz):
    """Set one border of a table cell (sz in eighths of a point; 0 removes it)."""
    tcPr = tc._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    el = OxmlElement(f"w:{side}")
    el.set(qn("w:val"), "single" if sz else "nil")
    if sz:
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), "000000")
    borders.append(el)


rows = [
    ("Gemma+NN-PPI [1]", "0%", cell(cl, "gemma_nnppi_sel"), cell(cb, "gemma_nnppi_sel")),
    ("임베딩 SVM", "0%", cell(cl, "svm"), cell(cb, "svm")),
    ("LLM 전량 호출", "100%", cell(cl, "sonnet_raw"), cell(cb, "sonnet_raw")),
    ("  + 임계값 조정", "100%", cell(cl, "sonnet_thr"), cell(cb, "sonnet_thr")),
    ("  + NN-PPI [1]", "100%", cell(cl, "sonnet_nnppi"), cell(cb, "sonnet_nnppi")),
    ("교체형 캐스케이드", "50%", cell(cl, "replace_0.5"), cell(cb, "replace_0.5")),
    ("결합형 캐스케이드 (제안)", "50%", cell(cl, "fuse_0.5"), cell(cb, "fuse_0.5")),
]
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.paragraph_format.space_before = Pt(4)
cap.paragraph_format.space_after = Pt(2)
set_font(cap.add_run("표 1. 테스트 정확도(†: 5회 중 3회 이상 제안보다 유의하게 낮음)"), BODY_FONT, 8, bold=True)
t = doc.add_table(rows=1 + len(rows), cols=4)
t.style = "Table Grid"  # compact cell paragraphs; its grid lines are switched off below
t.alignment = WD_TABLE_ALIGNMENT.CENTER
_tb = OxmlElement("w:tblBorders")
for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
    _el = OxmlElement(f"w:{side}")
    _el.set(qn("w:val"), "nil")
    _tb.append(_el)
t._tbl.tblPr.append(_tb)
for j, h in enumerate(["방법", "호출률", "CLEF", "ClaimBuster"]):
    c = t.rows[0].cells[j]
    c.text = ""
    set_font(c.paragraphs[0].add_run(h), BODY_FONT, 8, bold=True)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for i, (name, rate, a1, a2) in enumerate(rows, start=1):
    bold = "제안" in name
    for j, v in enumerate([name, rate, a1, a2]):
        c = t.rows[i].cells[j]
        c.text = ""
        set_font(c.paragraphs[0].add_run(v), BODY_FONT, 8, bold=bold)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
widths = [Cm(4.75), Cm(0.95), Cm(1.05), Cm(1.6)]  # 8.35 cm: one template column (신명조 digits are wide)
for row in t.rows:
    for c, w in zip(row.cells, widths):
        c.width = w
# horizontal rules only, as in the peer KICS papers: top and bottom 1 pt, below the header and above the proposed row 0.5 pt
for c in t.rows[0].cells:
    rule(c, "top", 8)
    rule(c, "bottom", 4)
for c in t.rows[-1].cells:
    rule(c, "top", 4)
    rule(c, "bottom", 8)
# keep caption and every row of Table 1 on one page/column
cap.paragraph_format.keep_with_next = True
for row in t.rows[:-1]:
    for c in row.cells:
        c.paragraphs[0].paragraph_format.keep_with_next = True

# replacement at 50% minus the stronger all-call LLM (mean over runs), signed
GAP = {d: f"{100 * SR[d]['gap_to_best_full']['0.5']['replace']:+.2f}" for d in (cl, cb)}
assert nsig(cb, "fuse_0.5_vs_fuse_1.0") == 5  # "ClaimBuster에서는 이 차이가 5회 모두 유의"
mneg = lambda x: f"{x:.1f}".replace("-", "−")
first = body(doc, (
    "표 1에서 LLM을 사용하지 않는 임베딩 SVM은 NN-PPI보다 CLEF에서 5회 중 "
    f"{nsig(cl,'svm_vs_gemma_nnppi_sel')}회 유의하게 높았고, ClaimBuster에서는 5회 모두 유의차가 없었다."
))
first.paragraph_format.space_before = Pt(5)  # air between Table 1 and the text below it
body(doc, (
    f"LLM 점수를 0.5 기준으로 쓰면 재현율이 CLEF {R[cl]['metrics']['sonnet_raw']['rec1'][0]:.2f}, ClaimBuster "
    f"{R[cb]['metrics']['sonnet_raw']['rec1'][0]:.2f}에 그쳤고, 임계값을 조정하면 재현율은 오르나 ClaimBuster 정확도는 "
    f"{pp(cb,'sonnet_raw','sonnet_thr')}%p 낮아졌다. 이는 LLM의 기준이 라벨과 다름을 시사한다."
))
body(doc, (
    "이하 전량 호출은 LLM 전량 호출의 두 설정(조정 전, 임계값 조정) 중 정확도가 높은 쪽"
    f"(CLEF는 {BF_NAME[BF[cl]]}, ClaimBuster는 {BF_NAME[BF[cb]]})을 뜻한다. 호출률 50%의 결합형은 전량 호출보다 정확도가 "
    f"평균 {MEAN_GAIN[cl]:.1f}%p, {MEAN_GAIN[cb]:.1f}%p 높았고, 5회 모두 차이의 95% 신뢰구간 하한이 −{NI:.1f}%p 이상이었다. "
    f"즉 결합형의 정확도는 전량 호출보다 {NI:.1f}%p 넘게 낮지 않다. 반면 같은 호출률의 교체형은 ClaimBuster에서 하한이 "
    f"{mneg(100*lo('sonnet5', cb, 'replace_0.5'))}%p까지 내려가 이를 보장하지 못했다. 또한 결합형은 NN-PPI를 적용한 전량 "
    f"호출보다 5회 모두 유의하게 높았다({p_all([(d, 'fuse_0.5_vs_sonnet_nnppi') for d in (cl, cb)], False)})."
))
body(doc, (
    f"테스트 배치 {COST_RAW['n_calls']}개를 다시 호출하여 추정한 1,000문장당 LLM 비용"
    f"(Claude Sonnet 5 정가)은 전량 호출 ${COST['1.0']['usd']:.3f}, 결합형 50% ${COST['0.5']['usd']:.3f}였고, 순차 처리 "
    f"시간은 임베딩 SVM을 포함하여 {COST['1.0']['seconds']:.0f}초, {COST['0.5']['seconds']:.0f}초로 추정되었다."
))
body(doc, (
    "그림 2와 같이 결합형은 LLM을 호출하는 모든 호출률에서 교체형보다 평균 정확도가 높았다. 전량 호출의 평균 정확도에 교체형은 "
    f"호출률 50%에서 이르지만, 결합형은 CLEF {REACH[cl]}%, ClaimBuster "
    f"{REACH[cb]}%에서 이르렀다."
))

# worked example: results/paper/example_case.json (scripts/example_case.py, seed-0 CLEF models, 50% calls)
EX = json.load(open(os.path.join(ROOT, "results", "paper", "example_case.json"), encoding="utf-8"))
E = EX["example"]
neg = lambda x, n=2: f"{x:.{n}f}".replace("-", "−")
assert E["d"] < 0 and EX["llm_threshold"] <= E["s"] < E["bar"]  # replacement says "needed", fused says "not needed"
body(doc, (
    "두 방법은 같은 문장에 LLM을 호출하므로 차이는 판정 방식에서만 생긴다. 예를 들어 CLEF의 "
    f"한 공약 문장(라벨: 불필요)은 LLM 점수가 s={E['s']:.2f}로, 임계값 t={eul(f'{EX['llm_threshold']:.2f}')} 사용하는 교체형은 "
    f"‘필요’로 잘못 판정하였으나, 결합형은 분류기의 결정값(d={neg(E['d'])})에 따라 판정 기준이 {ro(f'{E['bar']:.2f}')} 높아져 "
    "옳게 판정하였다."
))

# second LLM (Claude Haiku 4.5, same prompt and batches)
hk = lambda d, k: f3(HK[d]["mean_acc"][k])
HB = {d: HK[d]["best_full"] for d in (cl, cb)}
assert HB[cl] == HB[cb] == "sonnet_raw"  # text: "전량 호출" = Haiku at the 0.5 threshold on both datasets
assert all(HK[d]["per_seed"][f"fuse_0.5_vs_best_full({HB[d]})"]["n_sig"] == 0 for d in (cl, cb))
assert HK[cl]["per_seed"][f"replace_0.5_vs_best_full({HB[cl]})"]["n_sig"] >= 3
assert all(HK[d]["per_seed"]["fuse_0.5_vs_sonnet_nnppi"]["n_sig"] < 3 for d in (cl, cb))
body(doc, (
    f"LLM을 Claude Haiku 4.5로 바꾸어도 결합형(50%)의 정확도는 전량 호출보다 {NI_HK:.1f}%p 넘게 낮지 않았으나(5회 모두), "
    f"교체형은 CLEF에서 하한이 {mneg(100*lo('haiku45', cl, 'replace_0.5'))}%p까지 내려갔다. 단, 이때 결합형은 NN-PPI를 적용한 "
    f"전량 호출({hk(cl,'sonnet_nnppi')}, {hk(cb,'sonnet_nnppi')})보다 유의하게 높지 않았다."
))
body(doc, (
    "한편 학습 데이터(2012년 토론)와 시기가 다른 ClaimBuster 테스트(2016년 토론)에서는 분류기의 재현율이 "
    f"{REC['cb_svm']}에 그쳐(CLEF {REC['cl_svm']}), 결합형도 확인이 필요한 문장을 임계값을 조정한 전량 호출보다 많이 "
    f"놓쳤다(재현율 {REC['cb_fuse']} 대 {REC['cb_thr']}). 이러한 분류기의 노후화는 LLM을 호출한 문장 중 LLM만 ‘필요’로 "
    f"판정한 비율로 라벨 없이 감지할 수 있었고(학습 데이터 {DR['warn_cb_learn']:.1f}% → 테스트 {DR['warn_cb_test']:.1f}%, "
    f"CLEF는 증가 없음), 최근 문장 {DR['k']}개로 다시 학습하면 재현율이 {DR['gain']:.1f}%p 올랐다."
))

# built at the printed column width by scripts/make_figure.py --full, so inserted without rescaling
figure(doc, os.path.join(ROOT, "figures", "cascade_budget_full.png"), 8.2,
       "그림 2. 호출률에 따른 테스트 정확도 (5회 평균, 띠는 표준편차)")

# ---------------- IV. 결론 ----------------
heading(doc, "Ⅳ. 결 론")
body(doc, (
    "본 논문에서는 저비용 분류기가 확신하지 못하는 문장에만 LLM을 호출하고, 두 모델의 점수를 결합하여 최종 판정하는 "
    "결합형 캐스케이드를 제안하였다. 제안 방법은 LLM 호출을 절반으로 줄이면서도 모든 문장에 LLM을 호출한 경우와 같은 "
    f"수준의 정확도를 유지하였으며, 이에 따라 LLM 비용은 약 {CUT_USD:.0f}%, 처리 시간은 약 {CUT_SEC:.0f}% 줄일 수 있을 것으로 "
    "기대된다. 다만 학습 시기와 다른 데이터에서는 분류기가 노후화되어 확인이 필요한 문장을 더 놓칠 수 있으며, 이는 LLM과 "
    "분류기의 판정이 엇갈리는 비율로 감지하고 최근 데이터로 다시 학습하여 줄일 수 있었다. 또한 출처가 겹치는[3] 영어 두 "
    "데이터셋으로만 평가하였으므로, 향후에는 분류기의 주기적 재학습과 다국어 데이터로 평가를 확장할 계획이다."
))

# ACKNOWLEDGMENT omitted (no funding to acknowledge); re-add here if needed

heading(doc, "참 고 문 헌")
refs = [
    "[1] P. Amatya, Venktesh V, and V. Setty, \"Calibrating Small Language Models for Claim Check-Worthiness Detection,\" arXiv:2608.30731, 2026.",
    "[2] P. Amatya and V. Setty, \"Multilingual Fact-Checking at Scale: Fine-Tuned Compact Models vs LLMs,\" arXiv:2606.08605, 2026.",
    "[3] M. Hasanain et al., \"Overview of the CLEF-2024 CheckThat! Lab Task 1 on Check-Worthiness Estimation of Multigenre Content,\" CEUR-WS vol. 3740, pp. 276-286, 2024.",
    "[4] F. Arslan et al., \"A Benchmark Dataset of Check-Worthy Factual Claims,\" in Proc. ICWSM, vol. 14, pp. 821-829, 2020.",
    "[5] Z. Wang et al., \"Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference,\" arXiv:2609.07786, 2026.",
    "[6] Y. Zhang et al., \"Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous Model Collaboration,\" arXiv:2609.11446, 2026.",
]
for r in refs:
    ref = body(doc, r, size=9, indent=0, after=0, align=WD_ALIGN_PARAGRAPH.LEFT).paragraph_format
    ref.line_spacing = 1.0
    ref.left_indent, ref.first_line_indent = Pt(12.3), Pt(-12.3)  # hanging indent as in the template

# Word inserts a gap between Hangul and Latin/digits by default ("LLM을" rendered as "LLM 을");
# switch it off everywhere. pPr children must follow the OOXML schema order, so insert before the
# first element that the schema places after autoSpaceDN.
_AFTER_AUTOSPACE = {qn("w:" + n) for n in ("bidi", "adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing",
                                            "mirrorIndents", "suppressOverlap", "jc", "textDirection", "textAlignment",
                                            "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange")}


def no_autospace(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    anchor = next((c for c in pPr if c.tag in _AFTER_AUTOSPACE), None)
    for tag in ("w:autoSpaceDE", "w:autoSpaceDN"):
        el = OxmlElement(tag)
        el.set(qn("w:val"), "0")
        if anchor is None:
            pPr.append(el)
        else:
            anchor.addprevious(el)


_HANGUL_PAIR = re.compile(r"(?<=[가-힣])(?=[가-힣(])|(?<=[A-Za-z0-9%])(?=[가-힣])")  # also before "(" after Hangul, and between a Latin word or number and its particle (ClaimBuster|에서는), as in HWP


def hangul_breaks(paragraph):
    """Let justified lines break between Hangul syllables, as in HWP, by putting a zero-width space between adjacent
    syllables. Word otherwise breaks Korean only at spaces, so narrow justified columns stretch the gaps between 어절.
    (w:wordWrap=0 would do the same but also splits English words and numbers mid-token.)"""
    ts = [t for t in paragraph._p.iter(qn("w:t")) if t.text]
    for t in ts[:-1]:
        t.text = _HANGUL_PAIR.sub("​", t.text)
    if ts:  # keep the paragraph's last 어절 whole, so no single syllable ("다.") is left alone on the last line
        head, sep, tail = ts[-1].text.rpartition(" ")
        ts[-1].text = _HANGUL_PAIR.sub("​", head) + sep + tail


def no_break_hyphens(paragraph):
    """Replace '-' in text runs by Word's non-breaking hyphen so NN-PPI, RBF-SVM, dev-test never split across lines."""
    for t in list(paragraph._p.iter(qn("w:t"))):
        if "-" not in (t.text or ""):
            continue
        parts, parent, anchor = t.text.split("-"), t.getparent(), t
        t.text = parts[0]
        t.set(qn("xml:space"), "preserve")
        for part in parts[1:]:
            hyph = OxmlElement("w:noBreakHyphen")
            anchor.addnext(hyph)
            nt = OxmlElement("w:t")
            nt.set(qn("xml:space"), "preserve")
            nt.text = part
            hyph.addnext(nt)
            anchor = nt


for para in doc.paragraphs + [q for tbl in doc.tables for row in tbl.rows for cell in row.cells for q in cell.paragraphs]:
    no_autospace(para)
    if para.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY:
        hangul_breaks(para)
    if not para.text.startswith("["):  # reference entries may break at hyphens
        no_break_hyphens(para)

out = os.path.join(HERE, "cascade_kics_draft.docx")
doc.save(out)
print("saved", out)
