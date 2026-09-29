# -*- coding: utf-8 -*-
"""Build the KICS 2026 2-page draft (paper/cascade_kics_draft.docx).

Every number is read from the full held-out evaluation, results/final_results_full.json
(scripts/final_eval.py --full), except
the CLEF NN-PPI reproduction F1, which is on the original paper's dev-test split (results/final_results.json);
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
R = json.load(open(os.path.join(ROOT, "results", "final_results_full.json"), encoding="utf-8"))
R_ORIG = json.load(open(os.path.join(ROOT, "results", "final_results.json"), encoding="utf-8"))
IMP = json.load(open(os.path.join(ROOT, "results", "streaming_results_full.json"), encoding="utf-8"))
# significance on each of the 5 runs, and the call rate at which each cascade reaches the all-call LLM (scripts/seed_robustness.py)
SR = json.load(open(os.path.join(ROOT, "results", "seed_robustness_full.json"), encoding="utf-8"))
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


LATIN_FONT = "Times New Roman"  # Latin letters and digits; HY신명조's hyphen is drawn as a long dash


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


BODY_LINE_SPACING = 1.1  # 1.0 read as cramped at 9 pt in two columns


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
centered(doc, "이정빈", 11, after=0).paragraph_format.space_before = Pt(7.6)
centered(doc, "(소속 입력 필요)", 11, after=0)
centered(doc, "leejb020313@gmail.com", 11, after=7.6)
centered(doc, "Cost-Efficient Check-Worthy Claim Detection via Selective Fusion\nof a Low-Cost Classifier and a Large Language Model",
         13, True, after=0)
centered(doc, "Jeongbin Lee", 11, after=0).paragraph_format.space_before = Pt(14.2)
centered(doc, "(Affiliation)", 11, after=14.2)

cb, cl = "cb", "clef"
BF = {d: best_full(d) for d in (cl, cb)}
BF_NAME = {"sonnet_thr": "임계값 조정", "sonnet_raw": "조정 전"}
# first call rate whose mean accuracy reaches the stronger all-call LLM setting
REACH = {d: round(100 * SR[d]["first_rate_reaching_best_full"]["fuse"]) for d in (cl, cb)}
assert all(SR[d]["first_rate_reaching_best_full"]["replace"] == 0.5 for d in (cl, cb))  # text: "교체형은 50%"
assert REACH[cl] <= REACH[cb]  # abstract prints the range as CLEF~ClaimBuster
centered(doc, "요 약", 11, False, after=3)
body(doc, (
    "팩트체크 필요성 탐지(check-worthiness detection)는 유입되는 문장 중 사실 확인이 필요한 문장을 선별하는 기술로, "
    "모든 문장에 대형 언어 모델(LLM)을 호출하면 비용과 지연이 크다. 본 논문에서는 라벨로 "
    "학습한 저비용 분류기가 불확실하게 판정한 문장에만 LLM을 호출하고, LLM의 판정으로 교체하는 대신 두 모델의 점수를 "
    "결합하는 결합형 캐스케이드를 제안한다. 또한 선행 연구 NN-PPI를 재현하여, 같은 라벨로 학습한 분류기만으로도 NN-PPI와 "
    "같거나 높은 정확도를 얻음을 보인다. CLEF 2024와 ClaimBuster에서 평가한 결과, 제안 방법은 LLM 호출을 절반으로 줄이면서도 "
    f"LLM만 전량 호출한 경우와 유의한 차이가 없는 정확도(CLEF {f3(acc(cl,'fuse_0.5'))}, ClaimBuster {f3(acc(cb,'fuse_0.5'))})를 "
    f"달성하였으며, 교체형(50%)보다 적은 {REACH[cl]}~{REACH[cb]}%의 호출률에서 전량 호출 수준에 도달하였다."
), size=9, indent=0.5, after=2.5)

doc.add_section(0)
_body = doc.sections[-1]
_body.top_margin = _body.bottom_margin = _body.left_margin = Cm(2.0)
_body.right_margin = Cm(1.5)
columns(doc, 2)

# ---------------- I. 서론 ----------------
heading(doc, "Ⅰ. 서 론")
body(doc, (
    "팩트체크 필요성 탐지는 유입되는 모든 문장을 대상으로 사실 확인이 필요한지를 판정하므로 처리할 문장의 수가 많다. "
    "실제 팩트체크 서비스 운영사는 모든 문장에 LLM을 적용하는 것이 비현실적이라고 보고하였으며[1], 같은 연구진은 "
    "파인튜닝한 소형 인코더도 LLM과 경쟁할 수 있으나 관용적 표현에서는 LLM이 앞선다고 보고하고, 인코더가 확신하지 못할 "
    "때만 LLM에 넘기는 하이브리드 방식을 향후 과제로 제시하였다[2]."
))
body(doc, (
    "이러한 하이브리드는 일반적으로 캐스케이드[4] 형태로 구현되며, 분류기가 넘긴 문장의 판정을 LLM의 답으로 교체한다. "
    "그러나 LLM의 판정 기준이 데이터셋 라벨과 다를 경우, 교체는 분류기의 옳은 판정까지 틀리게 바꿀 수 있다[7]. 본 논문에서는 "
    "저비용 분류기가 불확실한 문장에서만 LLM을 호출하고, 두 모델의 점수를 라벨로 학습한 가중치로 결합하는 결합형 "
    "캐스케이드를 제안한다. 또한 선행 연구 NN-PPI[1]를 재현하여 저비용 분류기와 비교하고, LLM 단독 판정의 한계를 함께 "
    "분석한다."
))

# ---------------- II. 본론 ----------------
heading(doc, "Ⅱ. 본 론")
subheading(doc, "2.1 선행 연구")
body(doc, (
    "NN-PPI[1]는 소형 LLM이 출력한 점수를, 라벨이 있는 보정 세트에서 의미적으로 가까운 이웃 문장들의 잔차로 보정한다. "
    "LabelFusion[3]은 인코더와 LLM의 출력을 결합하지만 모든 입력에 LLM을 호출하며, 넘긴 입력에서만 두 모델의 출력을 "
    "결합하는 캐스케이드는 일반 분류 과제에서 제안되었다[8]. 반면 본 논문에서는 결합 가중치를 라벨로 학습하여 LLM과 "
    "라벨의 판정 기준 차이를 반영한다."
))
subheading(doc, "2.2 전체 구조")
body(doc, (
    "그림 1은 제안 방법의 전체 구조를 나타낸다. 저비용 분류기가 모든 문장을 먼저 판정하고, 분류기가 가장 불확실한 문장, 즉 "
    "결정값의 절댓값 |d(x)|가 가장 작은 문장부터 전체의 ρ(호출률)만큼 골라 LLM을 호출한다. 호출하지 않은 문장은 분류기의 "
    "판정(d(x)>0)을 그대로 사용하고, 호출한 문장은 2.4절의 결합 판정으로 최종 판정한다."
))
# figures/method_diagram.tex, compiled at its printed size (Tectonic) and inserted at that size
_dg = pymupdf.open(os.path.join(ROOT, "figures", "method_diagram.pdf"))[0].rect
figure(doc, os.path.join(ROOT, "figures", "method_diagram.png"), _dg.width / 72 * 2.54, "그림 1. 결합형 캐스케이드의 전체 구조")
subheading(doc, "2.3 저비용 분류기")
body(doc, (
    "문장 x의 팩트체크 필요 여부 y∈{0,1}을 판정하며, 라벨이 있는 학습 세트 L(NN-PPI의 보정 세트)을 사용한다. 저비용 "
    "분류기(이하 임베딩 SVM)는 NN-PPI가 이웃 검색에 사용하는 문장 임베딩(all-MiniLM-L6-v2)을 입력으로 하는 RBF-SVM이며, "
    "클래스 불균형을 고려한 가중치로 L에서 학습한다. 결정값 d(x)가 양수이면 팩트체크가 필요하다는 판정이며, 절댓값이 클수록 "
    "확신이 높다. 추론 시간은 CPU에서 문장당 약 6 ms이다."
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
    "는 조건이 참이면 1인 지시함수이다. 식 (1)을 s에 대해 정리하면 결합형은 s를 (−c−a·d)/b와 비교하게 되므로, LLM 점수의 "
    "판정 기준이 분류기의 결정값에 따라 달라진다. 반면 비교 대상인 교체형 캐스케이드(이하 교체형)는 같은 문장에 LLM을 "
    "호출하되 d를 사용하지 않고, s를 L에서 정한 고정 임계값 t와 비교한다."), BODY_FONT, 9)

# ---------------- III. 실험 ----------------
heading(doc, "Ⅲ. 실 험")
subheading(doc, "3.1 실험 환경")
assert R[cl]["k_sel"] == R[cb]["k_sel"]  # the text says both datasets selected the same k
body(doc, (
    f"CLEF 2024 CheckThat! Task 1 영어 데이터[5]는 {R[cl]['n_calib_gemma']:,}문장을 학습 세트로, dev·dev-test·공식 test를 "
    f"합친 {R[cl]['n_test']:,}문장을 테스트로 사용하였다. ClaimBuster[6]는 2012년 토론 {R[cb]['n_calib_gemma']:,}문장과 "
    f"2016년 토론 {R[cb]['n_test']:,}문장을 각각 학습 세트와 테스트로 사용하였다. NN-PPI는 원 논문과 같이 Gemma 3 4B로 "
    "재현하였으며, 원 논문과 같은 분할(CLEF dev-test, ClaimBuster 2016년)에서 가중 F1은 "
    f"CLEF {R_ORIG[cl]['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}, "
    f"ClaimBuster {R[cb]['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}였다(원 논문 0.827, 0.760). LLM은 Claude Sonnet 5를 "
    "사용하였고, NN-PPI의 판정 기준 프롬프트로 40문장씩 묶어 채점하였다(문장 단위 채점과 AUC 0.991 대 0.990)."
))
subheading(doc, "3.2 평가 방법")
CAL_GAIN = max(cal(d, f'fuse_{b}') - cal(d, 'fuse_0.5') for d in (cl, cb) for b in (.6, .7, .8, .9, 1.0))
assert max(cal(d, 'sonnet_thr_oof') - cal(d, 'fuse_0.5') for d in (cl, cb)) <= 0.007 + 1e-9  # "0.7%p 이내"
body(doc, (
    "평가 지표는 테스트 정확도이며, 학습 세트의 80%를 비복원 추출하여 5회 반복한 평균을 보고한다. 방법 간 차이는 5회 "
    "각각의 예측에 McNemar 검정(α=0.05)을 적용하여 평가하였다. LLM 임계값, 결합 계수, 호출률, NN-PPI의 이웃 수 k는 모두 학습 세트 "
    f"안에서만 정하였고(두 데이터셋 모두 k={R[cl]['k_sel']}), 임베딩 SVM은 기본 하이퍼파라미터를 사용하였다. 호출률은 학습 "
    f"세트의 교차검증 정확도가 더 오르지 않는 50%로 정하였다(50% 이후 증가 {100*CAL_GAIN:.1f}%p 이하, 전량 호출과 0.7%p "
    "이내)."
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
    ("Gemma 3 4B + NN-PPI [1]", "0%", cell(cl, "gemma_nnppi_sel"), cell(cb, "gemma_nnppi_sel")),
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
set_font(cap.add_run("표 1. 테스트 정확도 (†: 5회 중 3회 이상 제안보다 유의하게 낮음)"), BODY_FONT, 8.5, bold=True)
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
    set_font(c.paragraphs[0].add_run(h), BODY_FONT, 8.5, bold=True)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for i, (name, rate, a1, a2) in enumerate(rows, start=1):
    bold = "제안" in name
    for j, v in enumerate([name, rate, a1, a2]):
        c = t.rows[i].cells[j]
        c.text = ""
        set_font(c.paragraphs[0].add_run(v), BODY_FONT, 8.5, bold=bold)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
widths = [Cm(4.3), Cm(1.15), Cm(1.2), Cm(1.7)]  # 8.35 cm: one template column
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

first = body(doc, (
    "표 1은 각 방법의 테스트 정확도를 비교한 결과이다. LLM을 사용하지 않는 임베딩 SVM은 NN-PPI보다 CLEF에서 5회 중 "
    f"{nsig(cl,'svm_vs_gemma_nnppi_sel')}회 유의하게 높았고, ClaimBuster에서는 5회 모두 유의한 차이가 없었다. 즉 같은 라벨과 임베딩으로 학습한 저비용 분류기만으로도 NN-PPI와 같거나 "
    "높은 성능을 얻을 수 있다."
))
first.paragraph_format.space_before = Pt(5)  # air between Table 1 and the text below it
body(doc, (
    f"LLM 점수를 그대로(0.5 기준) 사용하면 정밀도는 CLEF {R[cl]['metrics']['sonnet_raw']['prec1'][0]:.2f}, "
    f"ClaimBuster {ro(f"{R[cb]['metrics']['sonnet_raw']['prec1'][0]:.2f}")} 높았으나 재현율은 "
    f"{R[cl]['metrics']['sonnet_raw']['rec1'][0]:.2f}, {R[cb]['metrics']['sonnet_raw']['rec1'][0]:.2f}에 그쳤다. 라벨로 임계값을 "
    f"조정하면 재현율은 {R[cl]['metrics']['sonnet_thr']['rec1'][0]:.2f}, {ro(f"{R[cb]['metrics']['sonnet_thr']['rec1'][0]:.2f}")} "
    "향상되었으나, ClaimBuster에서는 학습 세트가 클래스 균형인 반면 테스트는 팩트체크가 필요한 문장이 "
    f"{100*sum(v['n_pos'] for v in R[cb]['by_part'].values())/R[cb]['n_test']:.0f}%여서 정확도가 오히려 "
    f"{pp(cb,'sonnet_raw','sonnet_thr')}%p 낮아졌다. 이는 LLM의 판정 기준이 라벨과 다름을 시사한다."
))
body(doc, (
    f"제안한 결합형은 호출률 50%에서 CLEF {f3(acc(cl,'fuse_0.5'))}, ClaimBuster {f3(acc(cb,'fuse_0.5'))}의 정확도를 "
    "달성하였다. 이는 LLM 전량 호출 중 더 나은 설정"
    f"(CLEF는 {BF_NAME[BF[cl]]}, ClaimBuster는 {BF_NAME[BF[cb]]})과 5회 모두 유의한 차이가 없는 수준이며"
    f"({p_all([(d, f'fuse_0.5_vs_best_full({BF[d]})') for d in (cl, cb)], True)}), NN-PPI를 적용한 전량 호출보다는 5회 모두 "
    f"유의하게 높다({p_all([(d, 'fuse_0.5_vs_sonnet_nnppi') for d in (cl, cb)], False)})."
))


# replacement at 50% minus the stronger all-call LLM (mean over runs), signed
GAP = {d: f"{100 * SR[d]['gap_to_best_full']['0.5']['replace']:+.2f}" for d in (cl, cb)}
assert nsig(cb, "fuse_0.5_vs_fuse_1.0") == 5  # "ClaimBuster에서는 이 차이가 5회 모두 유의"


body(doc, (
    "그림 2는 호출률에 따른 테스트 정확도를 나타낸다. 점선은 LLM 전량 호출 중 더 나은 설정이고, 원은 선택한 호출률 50%이다. "
    "결합형은 LLM을 호출하는 모든 호출률에서 교체형보다 평균 정확도가 높았으며, 그 차이는 CLEF "
    f"{sig_range(cl)}, ClaimBuster {sig_range(cb)}의 호출률에서 5회 중 3회 이상 유의하였다. 한편 교체형도 호출률 50%에서 "
    f"LLM 전량 호출 수준에 이르므로(CLEF {GAP[cl]}%p, ClaimBuster {GAP[cb]}%p) 호출 절감 자체는 캐스케이드 구조에서 "
    f"비롯되며, 결합형은 이 수준에 CLEF {REACH[cl]}%, ClaimBuster {REACH[cb]}%의 호출률로 도달하였다. 호출률을 100%로 늘리면 정확도가 CLEF {pp(cl,'fuse_1.0','fuse_0.5')}%p, ClaimBuster "
    f"{pp(cb,'fuse_1.0','fuse_0.5')}%p 더 향상되었고, ClaimBuster에서는 이 차이가 5회 모두 유의하였다. 또한 테스트 문장의 "
    "|d| 순위 대신 학습 세트에서 정한 |d| 임계값으로 문장마다 호출 여부를 정해도 정확도는 "
    f"{IMP['clef']['acc']['stream_global_0.5'][0]:.3f}, {IMP['cb']['acc']['stream_global_0.5'][0]:.3f}였다"
    f"(실제 호출률 {100*IMP['clef']['test_call_rate']['0.5'][0]:.0f}%, {100*IMP['cb']['test_call_rate']['0.5'][0]:.0f}%)."
))

# worked example: results/example_case.json (scripts/example_case.py, seed-0 CLEF models, 50% calls)
EX = json.load(open(os.path.join(ROOT, "results", "example_case.json"), encoding="utf-8"))
E = EX["example"]
neg = lambda x, n=2: f"{x:.{n}f}".replace("-", "−")
assert E["d"] < 0 and EX["llm_threshold"] <= E["s"] < E["bar"]  # replacement says "needed", fused says "not needed"
body(doc, (
    "결합형과 교체형은 같은 문장에 LLM을 호출하므로, 두 방법의 차이는 호출 이후의 판정 방식에서 비롯된다. 예를 들어 CLEF의 "
    f"한 공약 문장(라벨: 불필요)은 LLM 점수가 s={E['s']:.2f}로, 임계값 t={eul(f'{EX['llm_threshold']:.2f}')} 사용하는 교체형은 "
    f"‘필요’로 잘못 판정하였으나, 결합형은 분류기의 결정값(d={neg(E['d'])})에 따라 판정 기준이 {ro(f'{E['bar']:.2f}')} 높아져 "
    "옳게 판정하였다. CLEF 1회차에서 LLM을 호출한 절반 중 결합형만 옳게 판정한 문장은 "
    f"{EX['n_queried_fuse_right_replace_wrong']}개, 교체형만 옳게 판정한 문장은 {EX['n_queried_replace_right_fuse_wrong']}개였다."
))

# built at the printed column width by scripts/make_figure.py --full, so inserted without rescaling
figure(doc, os.path.join(ROOT, "figures", "cascade_budget_full.png"), 8.2,
       "그림 2. 호출률에 따른 테스트 정확도 (5회 평균, 띠는 표준편차)")

# ---------------- IV. 결론 ----------------
heading(doc, "Ⅳ. 결 론")
body(doc, (
    "본 논문에서는 저비용 분류기가 불확실한 문장에서만 LLM을 호출하고 두 모델의 점수를 결합하는 결합형 캐스케이드를 "
    "제안하였다. 제안 방법은 LLM 호출을 절반으로 줄이면서도 LLM만 전량 "
    "호출한 경우와 유의한 차이가 없는 정확도를 달성하였다. 호출 절감은 캐스케이드 구조에서 비롯되며, 결합은 교체보다 "
    "적은 호출로 같은 정확도에 도달하였다. 다만 출처가 겹치는[5] "
    "영어 두 데이터셋과 단일 LLM으로 평가하였고, 비용을 호출률로만 측정하였다. 향후 다국어 데이터와 여러 LLM, 금액 기준 "
    "비용으로 평가를 확장할 계획이다."
))

# ACKNOWLEDGMENT omitted (no funding to acknowledge); re-add here if needed

heading(doc, "참 고 문 헌")
refs = [
    "[1] P. Amatya, Venktesh V, and V. Setty, \"Calibrating Small Language Models for Claim Check-Worthiness Detection,\" arXiv:2608.30731, 2026.",
    "[2] P. Amatya and V. Setty, \"Multilingual Fact-Checking at Scale: Fine-Tuned Compact Models vs LLMs,\" arXiv:2606.08605, 2026.",
    "[3] M. Schlee et al., \"LabelFusion: Fusing Large Language Models with Transformer Encoders for Robust Financial News Classification,\" arXiv:2512.10793, 2025.",
    "[4] T. Burleigh, \"Do Small Language Models Know When They're Wrong? Confidence-Based Cascade Scoring for Educational Assessment,\" arXiv:2604.19781, 2026.",
    "[5] M. Hasanain et al., \"Overview of the CLEF-2024 CheckThat! Lab Task 1 on Check-Worthiness Estimation of Multigenre Content,\" CEUR-WS vol. 3740, pp. 276–286, 2024.",
    "[6] F. Arslan, N. Hassan, C. Li, and M. Tremayne, \"A Benchmark Dataset of Check-Worthy Factual Claims,\" in Proc. ICWSM, vol. 14, pp. 821–829, 2020.",
    "[7] Z. Wang et al., \"Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference,\" arXiv:2609.07786, 2026.",
    "[8] Y. Zhang et al., \"Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous Model Collaboration,\" arXiv:2609.11446, 2026.",
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
