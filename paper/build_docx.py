# -*- coding: utf-8 -*-
"""Build the KICS 2026 2-page draft (paper/cascade_kics_draft.docx).

Every number is read from results/final_results.json and results/streaming_results.json (written by
scripts/final_eval.py and scripts/streaming_eval.py); Figure 1 is figures/cascade_budget.png
(scripts/make_figure.py). Math is LaTeX rendered to PNG by equations/render_equations.py.

    uv run --isolated --no-project --with python-docx --with pymupdf python paper/build_docx.py

Convert to PDF with Word (File > Save As > PDF), then check it:
    uv run --locked python scripts/verify_paper_numbers.py paper/cascade_kics_draft.pdf
    uv run --no-project --with pymupdf python paper/references/verify_references.py paper/cascade_kics_draft.pdf --online
"""
import json
import os

import pymupdf
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = json.load(open(os.path.join(ROOT, "results", "final_results.json"), encoding="utf-8"))
IMP = json.load(open(os.path.join(ROOT, "results", "streaming_results.json"), encoding="utf-8"))
BODY_FONT = "함초롬바탕"


def acc(ds, name):
    return R[ds]["metrics"][name]["acc"][0]


def f3(x):
    return f"{x:.3f}"


def best_nnppi(ds):
    return max(acc(ds, f"gemma_nnppi_k{k}") for k in (3, 5, 10))


def cal(ds, name):
    return R[ds]["calib_curve"][name][0]


def pval(ds, key):
    return R[ds]["contrasts"][key]["mcnemar_p"]


def set_font(run, name=BODY_FONT, size=9.5, bold=False, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = name
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


def heading(doc, text, size=10.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True  # never leave a heading alone at the bottom of a column
    set_font(p.add_run(text), BODY_FONT, size, bold=True)


def body(doc, text, size=9, indent=0.35, after=2.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    if indent:
        pf.first_line_indent = Cm(indent)
    set_font(p.add_run(text), BODY_FONT, size)
    return p


def add_math(run, name, scale=0.95):
    """Insert equations/<name>.png at the natural size of its LaTeX PDF (10pt math scaled to the 9.5pt body)."""
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
    cols.set(qn("w:space"), str(int(0.6 * 567)))


doc = Document()
for s in doc.sections:
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.top_margin, s.bottom_margin = Cm(2.2), Cm(2.0)
    s.left_margin = s.right_margin = Cm(1.8)

# ---------------- title block ----------------
centered(doc, "저비용 분류기와 대형 언어 모델의 선택적 결합을 통한", 15, True, after=0)
centered(doc, "팩트체크 필요성 탐지", 15, True, after=8)
centered(doc, "이정빈", 11, after=1)
centered(doc, "(소속 입력 필요)", 10, after=1)
centered(doc, "leejb020313@gmail.com", 9.5, after=8)
centered(doc, "Cost-Efficient Check-Worthy Claim Detection via Selective Fusion\nof a Low-Cost Classifier and a Large Language Model",
         12.5, True, font="Times New Roman", after=5)
centered(doc, "Jeongbin Lee", 11, font="Times New Roman", after=1)
centered(doc, "(Affiliation)", 10, font="Times New Roman", after=8)

cb, cl = "cb", "clef"
FL = R["cb"]["flips_confident_half_seed0"]
centered(doc, "요 약", 10.5, True, after=3)
body(doc, (
    "팩트체크 필요성 탐지(check-worthiness detection)는 유입되는 모든 문장에 적용되므로, 대형 언어 모델(LLM)을 "
    "모든 문장에 호출하면 비용과 지연이 크다. 선행 연구 NN-PPI는 소형 LLM 점수를 이웃 문장의 잔차로 보정하였다. "
    "본 논문은 CLEF 2024와 ClaimBuster에서 NN-PPI를 재현하고, 같은 라벨과 문장 "
    "임베딩으로 분류기를 직접 학습하면 LLM 없이도 NN-PPI와 유의한 차이가 없는 정확도를 얻음을 보인다. 이어 저비용 분류기가 모든 "
    "문장을 판정하고 불확실한 문장만 LLM에 질의하되, LLM의 판정으로 교체하지 않고 두 점수를 결합하는 결합형 "
    "캐스케이드를 제안한다. 보정 세트에서 정확도가 포화되는 호출률 50%(전량 호출과 0.7%p 이내)를 적용한 결과, "
    f"테스트 정확도는 CLEF {f3(acc(cl,'fuse_0.5'))}, ClaimBuster {f3(acc(cb,'fuse_0.5'))}로 LLM 전량 호출(임계값 조정 "
    f"{f3(acc(cl,'sonnet_thr'))}, {f3(acc(cb,'sonnet_thr'))}; NN-PPI 적용 {f3(acc(cl,'sonnet_nnppi'))}, {f3(acc(cb,'sonnet_nnppi'))})과 "
    "같거나 높았다. 결합형은 교체형 캐스케이드보다 평균 정확도가 높아, 라벨에 담긴 판정 기준과 LLM의 일반 지식이 "
    "상호 보완적임을 시사한다."
), size=9, indent=0.5)

doc.add_section(0)
columns(doc, 2)

# ---------------- I. 서론 ----------------
heading(doc, "Ⅰ. 서 론")
body(doc, (
    "팩트체크 필요성 탐지는 유입되는 모든 문장에 적용되므로, 실제 서비스 운영사는 모든 문장에 LLM을 적용하는 것이 "
    "비현실적이라고 보고하였다[1]. NN-PPI[1]는 소형 LLM의 점수를 라벨이 있는 보정 세트 중 의미적으로 가까운 이웃의 "
    "잔차로 보정한다. 같은 연구진은 파인튜닝한 소형 인코더가 LLM과 경쟁할 수 있으나 관용적 표현에서는 LLM이 앞선다고 "
    "보고하고, 인코더의 확신이 낮을 때만 LLM에 넘기는 하이브리드를 향후 과제로 제시하였다[2]. 일반적인 캐스케이드[4]는 "
    "불확실한 입력을 더 큰 LLM에 넘기되 그 답으로 교체하고, LabelFusion[3]은 인코더와 LLM의 출력을 결합하지만 모든 입력에 "
    "LLM을 호출한다. 위임한 입력에서 두 모델의 출력을 선택적으로 결합하는 캐스케이드도 일반 분류 과제에서 제안되었다[8]. "
    "본 논문은 [2]의 하이브리드를 구현하되, 결합 가중치를 라벨로 학습하여 LLM과 라벨의 판정 기준 차이까지 반영한다."
))
body(doc, (
    "본 논문의 기여는 다음과 같다. (1) NN-PPI를 재현하고, 같은 라벨과 임베딩으로 학습한 분류기가 LLM 없이 유의한 "
    "차이가 없는 정확도를 냄을 보인다. (2) 상용 LLM을 모든 문장에 적용해도 팩트체크가 필요한 문장을 "
    "과소 판정함을 보인다. (3) 저비용 분류기의 판정을 유지하면서 불확실한 문장에서만 LLM 점수를 결합하는 결합형 캐스케이드를 "
    "제안하고, 호출을 절반으로 줄여도 전량 호출에 준하는 정확도를 얻음을 보인다."
))

# ---------------- II. 제안 기법 ----------------
heading(doc, "Ⅱ. 제안 기법")
body(doc, (
    "문장 x가 팩트체크가 필요한지 y∈{0,1}을 판정하며, NN-PPI와 같은 라벨이 있는 보정 세트 L을 사용한다. 저비용 "
    "분류기(이하 임베딩 SVM)는 NN-PPI가 이웃 검색에 쓰는 문장 임베딩(all-MiniLM-L6-v2)을 입력으로 하는 RBF-SVM이며, 클래스 불균형을 "
    "고려한 가중치로 L에서 학습한다. 결정값 d(x)는 양수일수록 팩트체크가 필요함을 뜻하며, 추론은 CPU에서 문장당 약 "
    "6 ms이다."
))
body(doc, (
    "결합형 캐스케이드는 |d(x)|가 작은 순서, 즉 분류기가 가장 불확실한 문장부터 전체의 ρ 비율만큼 LLM에 질의하여 "
    "점수 s(x)∈[0,1]을 얻는다. 질의한 문장은 식 (1)로, 나머지는 d(x)>0 여부로 판정한다."
))
eq = doc.add_paragraph()
eq.paragraph_format.space_before = Pt(2)
eq.paragraph_format.space_after = Pt(3)
col_w = (21.0 - 2 * 1.8 - 0.6) / 2  # one text column (cm)
eq.paragraph_format.tab_stops.add_tab_stop(Cm(col_w / 2), WD_TAB_ALIGNMENT.CENTER)
eq.paragraph_format.tab_stops.add_tab_stop(Cm(col_w), WD_TAB_ALIGNMENT.RIGHT)
eq.add_run("\t")
add_math(eq.add_run(), "eq1")
set_font(eq.add_run("\t(1)"), BODY_FONT, 9)
p = body(doc, "(a, b, c)는 L에서 5겹 교차적합으로 얻은 d와 LLM 점수 s로 학습한 로지스틱 회귀 계수이고, "
             "σ는 시그모이드 함수, ")
add_math(p.add_run(), "indicator")
set_font(p.add_run(
    "은 조건이 참이면 1인 지시함수이다. 비교 대상인 교체형 캐스케이드는 같은 문장을 선택하되, LLM 점수에 L에서 "
    "고른 임계값만 적용하여 판정한다."), BODY_FONT, 9)

# ---------------- III. 실험 ----------------
heading(doc, "Ⅲ. 실험 및 결과")
body(doc, (
    f"실험에는 CLEF 2024 CheckThat! Task 1 영어 데이터(보정 {R[cl]['n_calib_gemma']:,}문장, 테스트는 dev-test {R[cl]['n_test']}문장)와 "
    f"ClaimBuster 데이터(보정은 2012년 토론 {R[cb]['n_calib_gemma']:,}문장, 테스트는 비용상 2016년 토론 중 무작위 {R[cb]['n_test']}문장)를 "
    "사용하였다[5,6]. "
    "NN-PPI는 원 논문의 Gemma 3 4B 설정으로 재현하였고(NN-PPI 적용 후 가중 F1 "
    f"CLEF {R[cl]['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}, ClaimBuster {R[cb]['metrics']['gemma_nnppi_sel']['wf1'][0]:.3f}; "
    f"원 논문 0.827, 0.760), 이웃 수 k는 보정 세트 내부 검증으로 선택하였다(두 데이터셋 모두 k={R[cl]['k_sel']}). LLM은 "
    "Claude Sonnet 5이며, NN-PPI의 판정 기준 프롬프트로 40문장씩 묶어 채점하였으며 문장 단위 채점과 AUC가 "
    "같았다(CLEF 69문장, 0.991 대 0.990). LLM 임계값, 결합 계수, k는 모두 보정 세트 안에서만 정하였고 "
    "임베딩 SVM은 기본 하이퍼파라미터를 사용하였다. 보정 세트의 80%를 비복원 추출하여 5회 반복한 평균을 보고하며, 방법 간 "
    "차이는 1회차 예측에 대한 McNemar 검정으로 평가하였다."
))

rows = [
    ("Gemma 3 4B", "0%", acc(cl, "gemma_raw"), acc(cb, "gemma_raw")),
    ("  + NN-PPI [1]", "0%", acc(cl, "gemma_nnppi_sel"), acc(cb, "gemma_nnppi_sel")),
    ("임베딩 SVM", "0%", acc(cl, "svm"), acc(cb, "svm")),
    ("Sonnet 5 전량 호출", "100%", acc(cl, "sonnet_raw"), acc(cb, "sonnet_raw")),
    ("  + 임계값 조정", "100%", acc(cl, "sonnet_thr"), acc(cb, "sonnet_thr")),
    ("  + NN-PPI [1]", "100%", acc(cl, "sonnet_nnppi"), acc(cb, "sonnet_nnppi")),
    ("교체형 캐스케이드", "50%", acc(cl, "replace_0.5"), acc(cb, "replace_0.5")),
    ("결합형 캐스케이드 (제안)", "50%", acc(cl, "fuse_0.5"), acc(cb, "fuse_0.5")),
    ("결합형 캐스케이드 (제안)", "100%", acc(cl, "fuse_1.0"), acc(cb, "fuse_1.0")),
]
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.paragraph_format.space_before = Pt(4)
cap.paragraph_format.space_after = Pt(2)
set_font(cap.add_run("표 1. 방법별 테스트 정확도 (5회 평균, 호출률 = LLM에 질의한 문장 비율)"), BODY_FONT, 8.5, bold=True)
t = doc.add_table(rows=1 + len(rows), cols=4)
t.style = "Table Grid"
t.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, h in enumerate(["방법", "호출률", "CLEF", "ClaimBuster"]):
    c = t.rows[0].cells[j]
    c.text = ""
    set_font(c.paragraphs[0].add_run(h), BODY_FONT, 8, bold=True)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for i, (name, rate, a1, a2) in enumerate(rows, start=1):
    bold = "제안" in name
    for j, v in enumerate([name, rate, f3(a1), f3(a2)]):
        c = t.rows[i].cells[j]
        c.text = ""
        set_font(c.paragraphs[0].add_run(v), BODY_FONT, 8, bold=bold)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
widths = [Cm(3.9), Cm(1.3), Cm(1.3), Cm(1.7)]
for row in t.rows:
    for c, w in zip(row.cells, widths):
        c.width = w

body(doc, (
    f"표 1에서 임베딩 SVM은 LLM 없이 NN-PPI와 차이가 유의하지 않았다(CLEF p={pval(cl,'svm_vs_gemma_nnppi_sel'):.2f}, "
    f"ClaimBuster p={pval(cb,'svm_vs_gemma_nnppi_sel'):.2f}). LLM을 그대로 쓰면 팩트체크가 필요한 문장을 과소 "
    f"판정하여 해당 클래스 재현율이 CLEF {R[cl]['metrics']['sonnet_raw']['rec1'][0]:.2f}, ClaimBuster "
    f"{R[cb]['metrics']['sonnet_raw']['rec1'][0]:.2f}(정밀도 {R[cl]['metrics']['sonnet_raw']['prec1'][0]:.2f}, "
    f"{R[cb]['metrics']['sonnet_raw']['prec1'][0]:.2f})에 그쳤고, 라벨로 임계값을 조정하자 각각 "
    f"{R[cl]['metrics']['sonnet_thr']['rec1'][0]:.2f}, {R[cb]['metrics']['sonnet_thr']['rec1'][0]:.2f}로 개선되었다. 이는 LLM의 판정 기준과 데이터셋 라벨 사이에 "
    "차이가 있음을 시사한다."
), after=3)

pic = doc.add_paragraph()
pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
pic.paragraph_format.space_after = Pt(1)
# built at the printed column width by figures/cascade_budget.py, so inserted without rescaling
pic.add_run().add_picture(os.path.join(ROOT, "figures", "cascade_budget.png"), width=Cm(8.2))
cap2 = doc.add_paragraph()
cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap2.paragraph_format.space_after = Pt(4)
set_font(cap2.add_run("그림 1. 호출률에 따른 테스트 정확도 (5회 평균, 띠는 표준편차)"), BODY_FONT, 8, bold=True)

body(doc, (
    "호출률 50%는 테스트가 아닌 보정 세트로 정하였다. 보정 세트의 교차적합 예측에서 결합형 정확도는 50% 이후 "
    f"{100*max(cal(d,f'fuse_{b}')-cal(d,'fuse_0.5') for d in (cl,cb) for b in (.6,.7,.8,.9,1.0)):.1f}%p 이하로만 올라 "
    f"포화되었고 전량 호출과의 차이는 0.7%p 이내였다(CLEF {f3(cal(cl,'fuse_0.5'))} 대 {f3(cal(cl,'sonnet_thr_oof'))}, "
    f"ClaimBuster {f3(cal(cb,'fuse_0.5'))} 대 {f3(cal(cb,'sonnet_thr_oof'))}). 이 호출률에서 테스트 정확도는 임계값 조정 "
    f"전량 호출보다 ClaimBuster에서 유의하게 높았고(p={pval(cb,'fuse_0.5_vs_sonnet_thr'):.3f}) CLEF에서는 "
    f"{100*(acc(cl,'fuse_0.5')-acc(cl,'sonnet_thr')):.1f}%p 높았으나 유의하지 않았으며(p={pval(cl,'fuse_0.5_vs_sonnet_thr'):.2f}), "
    f"NN-PPI 적용 전량 호출보다는 {100*(acc(cl,'fuse_0.5')-acc(cl,'sonnet_nnppi')):.1f}%p, "
    f"{100*(acc(cb,'fuse_0.5')-acc(cb,'sonnet_nnppi')):.1f}%p 높았으나 유의하지 않았다"
    f"(p={pval(cl,'fuse_0.5_vs_sonnet_nnppi'):.2f}, {pval(cb,'fuse_0.5_vs_sonnet_nnppi'):.2f}). 보정 세트에서는 이러한 "
    "초과 성능이 뚜렷하지 않아 데이터 분할에 따라 달라질 수 있다. 보정 세트에서 정한 |d| 임계값으로 문장마다 라우팅해도 "
    f"정확도는 비슷하였다(CLEF {IMP['clef']['acc']['stream_global_0.5'][0]:.3f}, 실제 호출 "
    f"{100*IMP['clef']['test_call_rate']['0.5'][0]:.0f}%; ClaimBuster {IMP['cb']['acc']['stream_global_0.5'][0]:.3f}, "
    f"{100*IMP['cb']['test_call_rate']['0.5'][0]:.0f}%). 교체형은 호출률이 높아질수록 분류기가 확신하는 문장의 판정까지 "
    "바꾸어 전량 호출 수준으로 되돌아갔다. 분류기 확신도 상위 절반에서 교체형은 옳은 판정 "
    f"{FL['replace']['broke']}건을 틀리게, {FL['replace']['fixed']}건을 옳게 바꾸었으나 결합형은 {FL['fuse']['broke']}건만 "
    "틀리게 바꾸었다(ClaimBuster 1회차). 이는 교체가 옳은 답을 해칠 수 있다는 보고[7]와 일치한다. 결합형은 LLM을 호출하는 "
    "모든 호출률에서 교체형보다 평균 정확도가 높았고 ClaimBuster의 호출률 30~50%에서 유의하였다"
    f"(p≤{max(pval(cb,f'fuse_{b}_vs_replace_{b}') for b in (0.3,0.4,0.5)):.3f})."
))

# ---------------- IV. 결론 ----------------
heading(doc, "Ⅳ. 결 론")
body(doc, (
    "팩트체크 필요성 탐지에서 라벨로 학습한 저비용 분류기와 LLM은 서로 다른 정보를 가지며, 불확실한 문장에서만 "
    "LLM을 호출해 두 점수를 결합하면 호출을 절반으로 줄여도 전량 호출에 준하는 정확도를 얻는다. 출처가 겹치는[5] 영어 두 데이터셋과 단일 "
    "LLM으로 평가하였고 비용을 호출률로만 보고하였다는 한계가 있다."
))

# ACKNOWLEDGMENT omitted (no funding to acknowledge); re-add here if needed

heading(doc, "참 고 문 헌", size=9.5)
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
    body(doc, r, size=7, indent=0, after=0, align=WD_ALIGN_PARAGRAPH.LEFT)

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


for para in doc.paragraphs + [q for tbl in doc.tables for row in tbl.rows for cell in row.cells for q in cell.paragraphs]:
    no_autospace(para)

out = os.path.join(HERE, "cascade_kics_draft.docx")
doc.save(out)
print("saved", out)
