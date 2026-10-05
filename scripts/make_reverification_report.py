"""Advisor-facing report of the 2026-10-05/06 re-verification (prior shift, CAUC gate, JEV), as an editable docx.

All numbers are read from results/analysis/*.json. Usage:
    uv run --locked --with python-docx python scripts/make_reverification_report.py <out.docx>
"""
import json
import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

A = os.path.join(os.path.dirname(__file__), "..", "results", "analysis")
J = lambda f: json.load(open(os.path.join(A, f), encoding="utf-8"))
WH = J("why_haiku.json")
PS = {(l, d): J(f"prior_shift_{l}_{d}.json") for l in ("sonnet5", "haiku45") for d in ("clef", "cb")}
CR = J("cauc_cr.json")
JP = {d: J(f"jev_prior_sonnet5_{d}.json") for d in ("clef", "cb")}
JR = {d: J(f"jev_replace_sonnet5_{d}.json") for d in ("clef", "cb")}
f3 = lambda x: f"{x:.3f}"
pp = lambda x: f"{100 * x:+.1f}%p".replace("-", "−")

doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = Cm(2)
    s.left_margin = s.right_margin = Cm(2.2)
st = doc.styles["Normal"]
st.font.name, st.font.size = "맑은 고딕", Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")


def para(text, bold=False, size=None, align=None, after=4):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)
    p.paragraph_format.space_after = Pt(2)


def table(rows, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = str(v)
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
                    r.bold = i == 0
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def h(text):
    para(text, bold=True, size=12, after=3)


para("결합형 캐스케이드 재검증 결과 보고", bold=True, size=15, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("이정빈 · 2026-10-06 · 저장소 브랜치 prof-feedback-1005", size=9, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)

h("1. 요약")
s, sc = PS[("sonnet5", "clef")]["mean_acc"], PS[("sonnet5", "cb")]["mean_acc"]
bullet("학습 세트를 정답 50:50으로 맞춰 뽑았는데, 원래 데이터와 테스트는 '필요' 문장이 24~27%입니다. "
       "이 때문에 학습 세트에서 정한 모든 기준선(LLM 임계값, 교체형 임계값, 결합형 절편)이 치우쳐 있었습니다.")
bullet(f"원래 데이터의 정답 비율(테스트 라벨 없이 알 수 있는 값)로 보정하면 Sonnet 전량 호출이 "
       f"{f3(s['allcall_prior'])} / {f3(sc['allcall_prior'])}로 올라, 논문의 결합형 50%보다 높습니다.")
bullet("같은 보정을 모든 방법에 적용하면 결합형은 교체형보다 낫지 않습니다. CAUC 논문의 결합 판정 기준을 적용해도 "
       "'결합하지 말라'는 결론이 나옵니다.")
bullet("따라서 현재 원고의 두 핵심 주장(호출 절반으로도 전량 호출보다 낮지 않음, 결합형 > 교체형)은 Sonnet에서 공정한 비교로 성립하지 않습니다.")

h("2. 어떻게 발견했나: Haiku가 Sonnet보다 정확해 보인 이유")
w = {d: WH[d] for d in ("clef", "cb")}
table([["", "Sonnet 5 (CLEF / CB)", "Haiku 4.5 (CLEF / CB)"],
       ["순위 판별력 AUC", f"{f3(w['clef']['sonnet5']['auc_test'])} / {f3(w['cb']['sonnet5']['auc_test'])}",
        f"{f3(w['clef']['haiku45']['auc_test'])} / {f3(w['cb']['haiku45']['auc_test'])}"],
       ["0.5 기준 정확도", f"{f3(w['clef']['sonnet5']['acc_05'])} / {f3(w['cb']['sonnet5']['acc_05'])}",
        f"{f3(w['clef']['haiku45']['acc_05'])} / {f3(w['cb']['haiku45']['acc_05'])}"],
       ["50:50 학습 세트로 고른 임계값", f"{w['clef']['sonnet5']['t_learn']:.2f} / {w['cb']['sonnet5']['t_learn']:.2f}",
        f"{w['clef']['haiku45']['t_learn']:.2f} / {w['cb']['haiku45']['t_learn']:.2f}"],
       ["비율 보정 임계값 → 정확도 (학습 세트 전체 1회)", f"{w['clef']['sonnet5']['t_prior']:.2f}→{f3(w['clef']['sonnet5']['acc_t_prior'])} / "
                                 f"{w['cb']['sonnet5']['t_prior']:.2f}→{f3(w['cb']['sonnet5']['acc_t_prior'])}",
        f"{w['clef']['haiku45']['t_prior']:.2f}→{f3(w['clef']['haiku45']['acc_t_prior'])} / "
        f"{w['cb']['haiku45']['t_prior']:.2f}→{f3(w['cb']['haiku45']['acc_t_prior'])}"]])
b = {x["bin"]: x for x in w["clef"]["sonnet5"]["bins"]}
para(f"Sonnet은 문장을 줄 세우는 능력이 Haiku와 같거나 더 좋지만, 확률을 낮게 부릅니다. 예를 들어 CLEF에서 Sonnet이 "
     f"0.3~0.4를 준 문장의 {100 * b['[0.3,0.4)']['label_rate']:.0f}%가 실제 '필요' 문장입니다. Haiku는 0.5가 우연히 최적 위치 근처에 "
     "있었을 뿐이며, 50:50 학습 세트는 Sonnet의 임계값을 지나치게 낮췄습니다.")

h("3. 같은 보정을 모든 방법에 적용한 비교 (100회 반복)")
rows = [["", "전량 호출", "교체형 50%", "결합형 50%", "결합형이 전량 호출보다 유의하게 낮은 반복", "결합형이 교체형보다 유의하게 낮은 반복"]]
for l, ln in (("sonnet5", "Sonnet 5"), ("haiku45", "Haiku 4.5")):
    for d, dn in (("clef", "CLEF"), ("cb", "ClaimBuster")):
        r = PS[(l, d)]
        m = r["mean_acc"]
        rows.append([f"{ln} · {dn}", f3(m["allcall_prior"]), f3(m["replace_prior_0.5"]), f3(m["fuse_prior_0.5"]),
                     f"{r['per_run']['fuse_prior_0.5_vs_allcall_prior']['sig_lower']}회",
                     f"{r['per_run']['fuse_prior_0.5_vs_replace_prior_0.5']['sig_lower']}회"])
table(rows)
para("사전 등록한 판정 규칙(results/analysis/prior_shift_preregistration.md)의 A에 해당합니다. Sonnet에서는 결합형이 전량 호출과 "
     "교체형 모두보다 낮습니다. Haiku에서는 세 방법이 거의 같습니다.")

h("4. CAUC[5]의 결합 판정 기준으로 본 결과")
para("CAUC는 결합 전에 상보성 비율 CR(작은 모델이 더 확신할 때 맞힌 수 − 틀린 수, 4쪽 식 5)을 계산해 양수일 때만 결합하고, "
     "아니면 큰 모델 답을 씁니다(=교체형). CAUC 자신도 MMLU 6개 모델 쌍 중 3개에서는 교체가 더 나았습니다(표 3).")
table([["", "CLEF", "ClaimBuster"]] + [[f"{n} — LLM을 부르는 절반의 CR",
        f"{pp(CR[f'{l}_clef']['called_half']['cr_mean'])} (양수 {CR[f'{l}_clef']['called_half']['runs_cr_pos']}/20)",
        f"{pp(CR[f'{l}_cb']['called_half']['cr_mean'])} (양수 {CR[f'{l}_cb']['called_half']['runs_cr_pos']}/20)"]
       for l, n in (("sonnet5", "SVM·Sonnet"), ("haiku45", "SVM·Haiku"))])
para("모두 음수입니다. LLM을 부르는 문장은 SVM이 가장 헷갈려하는 절반이라 SVM이 덧붙일 정보가 거의 없습니다. "
     "CAUC 규칙대로라면 이 과제에서는 결합하지 않는 것이 맞습니다.")

h("5. JEV를 첫 단계로 바꾼 경우")
table([["", "CLEF", "ClaimBuster"],
       ["Sonnet 전량 호출 (보정)", f3(JR["clef"]["mean_acc"]["sonnet_all"]), f3(JR["cb"]["mean_acc"]["sonnet_all"])],
       ["JEV 단독 (보정)", f3(JP["clef"]["mean_acc"]["jev_alone"]), f3(JP["cb"]["mean_acc"]["jev_alone"])],
       ["JEV→Sonnet 결합형 50%", f3(JP["clef"]["mean_acc"]["jevfuse_0.5"]), f3(JP["cb"]["mean_acc"]["jevfuse_0.5"])],
       ["JEV→Sonnet 교체형 30% / 50%",
        f"{f3(JR['clef']['mean_acc']['jevrep_0.3'])} / {f3(JR['clef']['mean_acc']['jevrep_0.5'])}",
        f"{f3(JR['cb']['mean_acc']['jevrep_0.3'])} / {f3(JR['cb']['mean_acc']['jevrep_0.5'])}"],
       ["  └ 전량 호출보다 유의하게 낮은 반복 (30%)", f"{JR['clef']['tests']['jevrep_0.3_vs_sonnet_all']['sig_lower']}회",
        f"{JR['cb']['tests']['jevrep_0.3_vs_sonnet_all']['sig_lower']}회"],
       ["학습 세트로 고른 호출률 (평균) → 유의하게 낮은 반복",
        f"{100 * JR['clef']['rate_mean']:.0f}% → {JR['clef']['tests']['jevrep_sel_vs_sonnet_all']['sig_lower']}회",
        f"{100 * JR['cb']['rate_mean']:.0f}% → {JR['cb']['tests']['jevrep_sel_vs_sonnet_all']['sig_lower']}회"]])
para("JEV는 CAUC 기준(CR > 0)은 통과하지만 결합형은 여전히 교체형보다 낮습니다. JEV→Sonnet 교체형은 30~50% 호출로 전량 호출과 같은 "
     "정확도를 내지만, 이 호출률은 테스트 곡선에서 읽은 값입니다. 학습 세트로 고르면 4~9%만 골라 실패합니다. 학습 세트에서는 "
     "Sonnet과 JEV의 정확도가 모두 약 0.87로 비슷한데 테스트에서는 Sonnet이 0.935로 높아, 학습 세트가 테스트를 대변하지 못합니다.")

h("6. 여쭙고 싶은 방향")
bullet(f"A. 비용-정확도 절충으로 재작성: 보정된 SVM→Sonnet 교체형 50%는 전량 호출보다 평균 "
       f"{100 * (s['allcall_prior'] - s['replace_prior_0.5']):.1f} / {100 * (sc['allcall_prior'] - sc['replace_prior_0.5']):.1f}%p 낮은 수준"
       "(반복별 유의성은 아직 미확인). 결합형은 제안하지 않음.")
bullet("B. 발견 자체를 주제로: 'LLM 점수의 눈금 차이와 학습 세트 비율 편향이 캐스케이드 평가를 왜곡한다'. 오늘 결과를 그대로 사용.")
bullet("C. 학습 세트와 테스트의 라벨 성격 차이를 먼저 해결(예: 테스트와 같은 분포의 소량 라벨로 호출률 결정)한 뒤 JEV→LLM 교체형으로 재작성.")

h("7. 재현 정보")
para("모든 실험은 결과를 보기 전에 판정 규칙을 커밋했습니다. 사전 등록: results/analysis/prior_shift_preregistration.md, "
     "jev_prior_preregistration.md, jev_replace_preregistration.md. 스크립트: scripts/why_haiku.py, prior_shift.py, cauc_cr.py, "
     "jev_prior.py, jev_replace.py. LLM 점수는 기존 채점 결과를 그대로 사용했고 추가 API 호출은 없습니다.", size=9)

doc.save(sys.argv[1])
print("saved", sys.argv[1])
