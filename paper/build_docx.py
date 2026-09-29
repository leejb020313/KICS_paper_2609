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


def figure(doc, png, width_cm, caption):
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.space_before = Pt(3)
    pic.paragraph_format.space_after = Pt(1)
    pic.paragraph_format.keep_with_next = True
    pic.add_run().add_picture(png, width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(4)
    set_font(cap.add_run(caption), BODY_FONT, 8, bold=True)


def body(doc, text, size=9, indent=0.3175, after=3.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, lead=None):
    """Body paragraph; `lead` is a bold run-in sentence that states the paragraph's finding."""
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
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
FL = R["cb"]["flips_confident_half_seed0"]
centered(doc, "요 약", 11, False, after=3)
body(doc, (
    "팩트체크 필요성 탐지(check-worthiness detection)는 유입되는 모든 문장에 적용되므로, 모든 문장에 대형 언어 모델(LLM)을 "
    "호출하면 비용과 지연이 크다. 본 논문은 소형 LLM 점수를 이웃 문장으로 보정하는 선행 연구 NN-PPI를 "
    "CLEF 2024와 ClaimBuster에서 재현하고, 같은 라벨과 문장 임베딩으로 학습한 저비용 분류기만으로도 NN-PPI와 같거나 높은 "
    "정확도를 얻음을 보인다. 이어 분류기가 불확실한 문장에만 LLM을 호출하고, 그 판정을 LLM의 답으로 교체하는 대신 두 점수를 "
    "결합하는 결합형 캐스케이드를 제안한다. LLM 호출을 절반으로 줄여도 테스트 정확도"
    f"(CLEF {f3(acc(cl,'fuse_0.5'))}, ClaimBuster {f3(acc(cb,'fuse_0.5'))})는 LLM 단독 전량 호출의 최고 정확도"
    f"({f3(acc(cl,best_full(cl)))}, {f3(acc(cb,best_full(cb)))})와 유의한 차이가 없었고, NN-PPI를 적용한 전량 호출"
    f"({f3(acc(cl,'sonnet_nnppi'))}, {f3(acc(cb,'sonnet_nnppi'))})보다는 유의하게 높았다. 결합형은 모든 호출률에서 "
    "교체형 캐스케이드보다 평균 정확도가 높아, 라벨에 담긴 판정 기준과 LLM의 일반 지식이 서로 보완적임을 시사한다."
), size=9, indent=0.5, after=2.5)

doc.add_section(0)
_body = doc.sections[-1]
_body.top_margin = _body.bottom_margin = _body.left_margin = Cm(2.0)
_body.right_margin = Cm(1.5)
columns(doc, 2)

# ---------------- I. 서론 ----------------
heading(doc, "Ⅰ. 서 론")
body(doc, (
    "팩트체크 필요성 탐지는 유입되는 모든 문장에 적용되므로, 실제 팩트체크 서비스 운영사는 모든 문장에 LLM을 적용하는 것이 "
    "비현실적이라고 보고하였다[1]. 이에 NN-PPI[1]는 소형 LLM의 점수를, 라벨이 있는 보정 세트에서 의미적으로 가까운 이웃 "
    "문장의 잔차로 보정한다. 같은 연구진은 파인튜닝한 소형 인코더도 LLM과 경쟁할 수 있으나 관용적 표현에서는 LLM이 앞선다고 "
    "보고하며, 인코더가 확신하지 못할 때만 LLM에 넘기는 하이브리드를 향후 과제로 남겼다[2]."
))
body(doc, (
    "이러한 하이브리드의 일반적 형태인 캐스케이드[4]는 넘긴 입력의 판정을 LLM의 답으로 교체한다. 그러나 LLM의 판정 기준은 "
    "데이터셋 라벨과 다를 수 있어, 교체는 분류기의 옳은 판정까지 바꿀 수 있다. LabelFusion[3]은 인코더와 LLM의 출력을 "
    "결합하지만 모든 입력에 LLM을 호출하며, 넘긴 입력에서만 두 출력을 결합하는 캐스케이드는 일반 분류 과제에서 "
    "제안되었다[8]. 본 논문은 [2]의 하이브리드를 팩트체크 필요성 탐지에 구현하되, 결합 가중치를 라벨로 학습하여 LLM과 "
    "라벨의 판정 기준 차이까지 반영한다."
))
body(doc, (
    "본 논문의 기여는 세 가지이다. (1) NN-PPI를 재현하고, 같은 라벨과 임베딩으로 학습한 분류기가 LLM 없이도 NN-PPI와 "
    "같거나 높은 정확도를 냄을 보인다. (2) 상용 LLM을 모든 문장에 적용해도 팩트체크가 필요한 문장을 상당수 놓침을 보인다. "
    "(3) 분류기가 불확실한 문장에서만 LLM 점수를 결합하는 결합형 캐스케이드를 제안하고, 호출을 절반으로 줄여도 전량 "
    "호출과 유의한 차이가 없는 정확도를 얻음을 보인다."
))

# ---------------- II. 제안 기법 ----------------
heading(doc, "Ⅱ. 제안 기법")
subheading(doc, "2.1 저비용 분류기")
body(doc, (
    "문장 x에 대해 팩트체크 필요 여부 y∈{0,1}을 판정하며, 학습에는 NN-PPI와 같은 라벨 있는 보정 세트 L을 쓴다. 저비용 "
    "분류기(이하 임베딩 SVM)는 NN-PPI가 이웃 검색에 쓰는 문장 임베딩(all-MiniLM-L6-v2)을 입력으로 하는 RBF-SVM으로, "
    "클래스 불균형을 고려한 가중치로 L에서 학습한다. 결정값 d(x)가 양수이면 팩트체크가 필요하다는 판정이고, 절댓값이 클수록 "
    "분류기가 확신한다는 뜻이다. 추론은 CPU에서 문장당 약 6 ms가 걸린다."
))
subheading(doc, "2.2 결합형 캐스케이드")
body(doc, (
    "결합형 캐스케이드(이하 결합형)는 그림 1과 같이 동작한다. 분류기가 가장 불확실한 문장, 즉 |d(x)|가 가장 작은 문장부터 "
    "전체의 ρ(호출률)만큼 골라 LLM을 호출하고 점수 s(x)∈[0,1]을 얻는다. 호출한 문장은 식 (1)로 d와 s를 결합해 판정하고, "
    "나머지 문장은 분류기의 판정(d(x)>0)을 그대로 쓴다."
))
# figures/method_diagram.tex, compiled at its printed size (Tectonic) and inserted at that size
_dg = pymupdf.open(os.path.join(ROOT, "figures", "method_diagram.pdf"))[0].rect
figure(doc, os.path.join(ROOT, "figures", "method_diagram.png"), _dg.width / 72 * 2.54,
       "그림 1. 결합형 캐스케이드의 구조")
eq = doc.add_paragraph()
eq.paragraph_format.space_before = Pt(4)
eq.paragraph_format.space_after = Pt(4)
col_w = (21.0 - 2.0 - 1.5 - 0.75) / 2  # one text column (cm)
eq.paragraph_format.tab_stops.add_tab_stop(Cm(col_w / 2), WD_TAB_ALIGNMENT.CENTER)
eq.paragraph_format.tab_stops.add_tab_stop(Cm(col_w), WD_TAB_ALIGNMENT.RIGHT)
eq.add_run("\t")
add_math(eq.add_run(), "eq1")
set_font(eq.add_run("\t(1)"), BODY_FONT, 9)
p = body(doc, "(a, b, c)는 L에서 5겹 교차적합으로 얻은 d와 s로 학습한 로지스틱 회귀 계수, "
             "σ는 시그모이드 함수, ")
p.paragraph_format.widow_control = False  # may split across the page so the column bottom is not left empty
add_math(p.add_run(), "indicator")
set_font(p.add_run(
    "는 조건이 참이면 1인 지시함수이다. 비교 대상인 교체형 캐스케이드(이하 교체형)는 같은 문장에 LLM을 호출하되, d를 버리고 "
    "s에 L에서 고른 임계값만 적용한다."), BODY_FONT, 9)

# ---------------- III. 실험 ----------------
heading(doc, "Ⅲ. 실험 및 결과")
subheading(doc, "3.1 실험 설정")
body(doc, (
    f"CLEF 2024 CheckThat! Task 1 영어 데이터[5]는 {R[cl]['n_calib_gemma']:,}문장을 보정 세트로, dev·dev-test·공식 test를 "
    f"합친 {R[cl]['n_test']:,}문장을 테스트로 썼다. ClaimBuster[6]는 2012년 토론 {R[cb]['n_calib_gemma']:,}문장을 보정 세트로, "
    f"2016년 토론 {R[cb]['n_test']:,}문장을 테스트로 썼다."
), lead="데이터.")
assert R[cl]["k_sel"] == R[cb]["k_sel"]  # the text says both datasets selected the same k
body(doc, (
    "NN-PPI는 원 논문과 같이 Gemma 3 4B로 재현하였다. 원 논문과 같은 분할(CLEF dev-test, ClaimBuster 2016년)에서 가중 F1은 "
    f"CLEF {R_ORIG[cl]['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}, "
    f"ClaimBuster {R[cb]['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}였다(원 논문 0.827, 0.760). 이웃 수 k는 "
    f"보정 세트 내부 검증으로 정하였다(두 데이터셋 모두 k={R[cl]['k_sel']}). LLM은 Claude Sonnet 5이며, NN-PPI의 판정 기준 "
    "프롬프트로 40문장씩 묶어 채점하였다. 문장 단위로 채점한 표본에서 AUC는 0.991 대 0.990으로 묶음 채점과 거의 같았다."
), lead="모델.")
body(doc, (
    "LLM 임계값, 결합 계수, k, 호출률은 모두 보정 세트 안에서만 정하였고, 임베딩 SVM은 기본 하이퍼파라미터를 썼다. 보정 "
    "세트의 80%를 비복원 추출하여 5회 반복한 평균을 보고하며, 방법 간 차이는 1회차 예측에 McNemar 검정(α=0.05)을 적용해 "
    "평가하였다."
), lead="평가.")

subheading(doc, "3.2 결과")
DAG = {"sonnet_raw": "fuse_0.5_vs_sonnet_raw", "sonnet_thr": "fuse_0.5_vs_sonnet_thr",
       "sonnet_nnppi": "fuse_0.5_vs_sonnet_nnppi", "replace_0.5": "fuse_0.5_vs_replace_0.5",
       "nnppi_0.5": "fuse_0.5_vs_nnppi_0.5"}


def cell(ds, name):
    """Accuracy as printed in Table 1; † when the fused cascade at 50% is significantly higher (McNemar p<0.05)."""
    return f3(acc(ds, name)) + ("†" if name in DAG and pval(ds, DAG[name]) < 0.05 else "")


rows = [
    ("Gemma 3 4B", "0%", cell(cl, "gemma_raw"), cell(cb, "gemma_raw")),
    ("  + NN-PPI [1]", "0%", cell(cl, "gemma_nnppi_sel"), cell(cb, "gemma_nnppi_sel")),
    ("임베딩 SVM", "0%", cell(cl, "svm"), cell(cb, "svm")),
    ("LLM 전량 호출", "100%", cell(cl, "sonnet_raw"), cell(cb, "sonnet_raw")),
    ("  + 임계값 조정", "100%", cell(cl, "sonnet_thr"), cell(cb, "sonnet_thr")),
    ("  + NN-PPI [1]", "100%", cell(cl, "sonnet_nnppi"), cell(cb, "sonnet_nnppi")),
    ("교체형 캐스케이드", "50%", cell(cl, "replace_0.5"), cell(cb, "replace_0.5")),
    ("NN-PPI 판정 캐스케이드", "50%", cell(cl, "nnppi_0.5"), cell(cb, "nnppi_0.5")),
    ("결합형 캐스케이드 (제안)", "50%", cell(cl, "fuse_0.5"), cell(cb, "fuse_0.5")),
    ("  (호출률 100%, 참고)", "100%", cell(cl, "fuse_1.0"), cell(cb, "fuse_1.0")),
]
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.paragraph_format.space_before = Pt(4)
cap.paragraph_format.space_after = Pt(2)
set_font(cap.add_run("표 1. 테스트 정확도 (5회 평균, †: 제안 방법(호출률 50%)보다 유의하게 낮음, McNemar p<0.05)"), BODY_FONT, 8.5, bold=True)
t = doc.add_table(rows=1 + len(rows), cols=4)
t.style = "Table Grid"
t.alignment = WD_TABLE_ALIGNMENT.CENTER
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
# keep caption and every row of Table 1 on one page/column
cap.paragraph_format.keep_with_next = True
for row in t.rows[:-1]:
    for c in row.cells:
        c.paragraphs[0].paragraph_format.keep_with_next = True

table_gap = body(doc, (
    f"CLEF에서는 임베딩 SVM이 NN-PPI보다 유의하게 높았고({pfmt(pval(cl,'svm_vs_gemma_nnppi_sel'))}), ClaimBuster에서는 "
    f"차이가 유의하지 않았다({pfmt(pval(cb,'svm_vs_gemma_nnppi_sel'))})."
), lead="LLM 없이도 NN-PPI와 같거나 높다.")
table_gap.paragraph_format.space_before = Pt(5)  # air between Table 1 and the text below it
body(doc, (
    f"LLM 점수를 그대로(0.5 기준) 쓰면 정밀도는 CLEF {R[cl]['metrics']['sonnet_raw']['prec1'][0]:.2f}, ClaimBuster "
    f"{ro(f"{R[cb]['metrics']['sonnet_raw']['prec1'][0]:.2f}")} 높지만 재현율은 {R[cl]['metrics']['sonnet_raw']['rec1'][0]:.2f}, "
    f"{R[cb]['metrics']['sonnet_raw']['rec1'][0]:.2f}에 그쳤다. 라벨로 임계값을 조정하면 재현율이 "
    f"{R[cl]['metrics']['sonnet_thr']['rec1'][0]:.2f}, {ro(f"{R[cb]['metrics']['sonnet_thr']['rec1'][0]:.2f}")} 올라, "
    "LLM의 판정 기준이 데이터셋 라벨과 다름을 시사한다. 다만 ClaimBuster는 보정 세트가 클래스 균형인 반면 테스트는 "
    f"팩트체크가 필요한 문장이 {100*sum(v['n_pos'] for v in R[cb]['by_part'].values())/R[cb]['n_test']:.0f}%뿐이어서, "
    f"조정 후 정확도가 오히려 {pp(cb,'sonnet_raw','sonnet_thr')}%p 낮아졌다."
), lead="LLM 단독은 팩트체크가 필요한 문장을 놓친다.")


def p_upper_multi(pairs):
    """Largest p over (dataset, contrast) pairs, rounded up to 3 decimals so that "p≤" never understates it."""
    return f"p≤{math.ceil(1000 * max(pval(ds, k) for ds, k in pairs)) / 1000:.3f}"


BF = {d: best_full(d) for d in (cl, cb)}
CAL_GAIN = max(cal(d, f'fuse_{b}') - cal(d, 'fuse_0.5') for d in (cl, cb) for b in (.6, .7, .8, .9, 1.0))
assert max(cal(d, 'sonnet_thr_oof') - cal(d, 'fuse_0.5') for d in (cl, cb)) <= 0.007 + 1e-9  # "0.7%p 이내"
body(doc, (
    f"호출률 50%는 보정 세트에서 정하였다. 교차적합 정확도가 50% 이후 {100*CAL_GAIN:.1f}%p 이하로만 올랐고, 전량 호출과의 "
    "차이도 0.7%p 이내였기 때문이다. 이 호출률에서 결합형은 LLM 단독 전량 호출의 최고 정확도(표 1의 두 설정 중 높은 쪽)와 "
    f"유의한 차이가 없었고(CLEF {pfmt(pval(cl,f'fuse_0.5_vs_{BF[cl]}'))}, ClaimBuster {pfmt(pval(cb,f'fuse_0.5_vs_{BF[cb]}'))}), "
    "NN-PPI를 적용한 전량 호출보다는 유의하게 높았다"
    f"({pfmt(max(pval(d,'fuse_0.5_vs_sonnet_nnppi') for d in (cl,cb)))}). 호출률을 100%로 올리면(그림 2) 정확도가 "
    f"CLEF {pp(cl,'fuse_1.0','fuse_0.5')}%p, ClaimBuster {pp(cb,'fuse_1.0','fuse_0.5')}%p 더 올랐고, ClaimBuster에서는 이 "
    f"차이가 유의하였다({pfmt(pval(cb,'fuse_0.5_vs_fuse_1.0_seed0'))}). 테스트 문장을 모아 순위를 매기지 않고 보정 세트에서 "
    f"정한 |d| 임계값으로 문장마다 바로 라우팅해도 정확도는 {IMP['clef']['acc']['stream_global_0.5'][0]:.3f}, "
    f"{IMP['cb']['acc']['stream_global_0.5'][0]:.3f}(실제 호출률 {100*IMP['clef']['test_call_rate']['0.5'][0]:.0f}%, "
    f"{100*IMP['cb']['test_call_rate']['0.5'][0]:.0f}%)였다."
), lead="호출을 절반으로 줄여도 정확도가 유지된다.")
body(doc, (
    "결합형은 LLM을 호출하는 모든 호출률에서 교체형보다 평균 정확도가 높았다(ClaimBuster 30~50%, CLEF 20~30%에서 유의, "
    f"{p_upper_multi([(cb, f'fuse_{b}_vs_replace_{b}') for b in (0.3,0.4,0.5)] + [(cl, f'fuse_{b}_vs_replace_{b}') for b in (0.2,0.3)])}). "
    "같은 문장에 LLM을 호출하고 NN-PPI로 판정하는 캐스케이드보다도 유의하게 높아"
    f"({pfmt(max(pval(d,'fuse_0.5_vs_nnppi_0.5') for d in (cl,cb)))}), 차이는 어떤 문장을 호출하느냐가 아니라 호출한 뒤 "
    "어떻게 판정하느냐에서 온다. 호출률을 100%로 올려 임베딩 SVM이 확신하는 절반까지 LLM을 호출하면, 교체형은 그중 옳은 "
    f"판정 {FL['replace']['broke']}건을 틀리게, 틀린 판정 {FL['replace']['fixed']}건을 옳게 바꾸었지만 결합형은 각각 "
    f"{FL['fuse']['broke']}건, {FL['fuse']['fixed']}건만 바꾸었다(ClaimBuster 1회차). 교체가 옳은 답을 해칠 수 있다는 "
    "보고[7]와 일치한다."
), lead="교체보다 결합이 낫다.")

# worked example: results/example_case.json (scripts/example_case.py, seed-0 CLEF models)
EX = json.load(open(os.path.join(ROOT, "results", "example_case.json"), encoding="utf-8"))
E = EX["example"]
neg = lambda x, n=2: f"{x:.{n}f}".replace("-", "−")
assert E["d"] < 0 and EX["llm_threshold"] <= E["s"] < E["bar"]  # replacement says "needed", fused says "not needed"
body(doc, (
    "식 (1)은 d에 따라 LLM 점수의 판정 기준을 옮긴다. 분류기가 ‘불필요’ 쪽으로 기울수록(d<0) 결합형은 LLM에 더 높은 "
    f"점수를 요구하며, CLEF 1회차에서 이 기준은 d=0일 때 {EX['bar_at_0']:.2f}이다. 예를 들어 공약 문장 “{E['text']}”"
    f"(라벨: 불필요)은 d={neg(E['d'])}여서 기준이 {ro(f"{E['bar']:.2f}")} 올라가고, s={E['s']:.2f}인 이 문장을 결합형은 옳게 "
    f"‘불필요’로 판정한다. 반면 LLM 임계값 {eul(f'{EX['llm_threshold']:.2f}')} 쓰는 교체형은 ‘필요’로 잘못 판정한다. "
    f"LLM을 호출한 절반 전체에서도 결합형만 옳은 문장({EX['n_queried_fuse_right_replace_wrong']}개)이 교체형만 옳은 "
    f"문장({EX['n_queried_replace_right_fuse_wrong']}개)보다 많았다."
), lead="작동 예.")

# built at the printed column width by scripts/make_figure.py --full, so inserted without rescaling
figure(doc, os.path.join(ROOT, "figures", "cascade_budget_full.png"), 8.2,
       "그림 2. 호출률에 따른 테스트 정확도 (5회 평균, 띠는 표준편차)")

# ---------------- IV. 결론 ----------------
heading(doc, "Ⅳ. 결 론")
body(doc, (
    "라벨로 학습한 저비용 분류기와 LLM은 서로 다른 정보를 가지며, 분류기가 불확실한 문장에서만 LLM을 호출해 두 점수를 "
    "결합하면 호출을 절반으로 줄여도 전량 호출과 유의한 차이가 없는 정확도를 얻는다. 또한 결합은 같은 문장을 LLM의 판정으로 "
    "교체하는 방식보다 모든 호출률에서 평균 정확도가 높았다. 다만 출처가 겹치는[5] 영어 두 데이터셋과 단일 LLM으로 "
    "평가하였고 비용을 호출률로만 측정하였다. 향후 다국어 데이터와 여러 LLM, 금액 기준 비용으로 평가를 확장할 계획이다."
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


_HANGUL_PAIR = re.compile(r"(?<=[가-힣])(?=[가-힣])")


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
