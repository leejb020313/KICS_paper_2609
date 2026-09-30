# 참고문헌 검증 자료 (jev_kics_draft)

`verify_references.py`가 생성합니다. 모든 인용문은 `pdfs/`의 원문 PDF에서 그대로 찾은 문장이며, 쪽수는 PDF 쪽 번호입니다. `support: inference`는 원문에 그대로 쓰인 문장이 아니라 원문을 근거로 한 **우리의 해석**이라는 뜻입니다.

## 요약

- 참고문헌 6편, 근거 17건 (원문 직접 확인 16건, 해석 1건)
- 원문 PDF에서 찾은 인용문 17/17건 (유료 원문 0건은 공개 초록으로 확인)
- 해석에 해당하는 근거 (심사에서 질문받으면 note의 논리로 답해야 함): 4c (본 논문은 ... 결합 가중치를 라벨로 학습하여 LLM과 라벨의 판정 기…)

## [1] P. Amatya, Venktesh V, and V. Setty, "Calibrating Small Language Models for Claim Check-Worthiness Detection," arXiv:2608.30731, 2026.

- 식별자: arXiv:2608.30731 (v2 확인) · 공개일 2026-08-31 · arXiv preprint (anonymised ARR code link: likely under ACL Rolling Review)
- 공식 저자: Pratuat Amatya, V Venktesh, Vinay Setty
- 원문 PDF: `pdfs/2608.30731.pdf`

**1a** (서론 1문단) — 논문: 실제 서비스 운영사는 모든 문장에 LLM을 적용하는 것이 비현실적이라고 보고하였다[1].

> "invoking large LLMs on every incoming claim is prohibitive at production volumes" (p. 1)

- 근거 유형: direct

**1b** (서론 1문단) — 논문: (운영사) — 저자들이 실제 서비스 운영 맥락에서 쓴 논문이라는 근거

> "This work is motivated by the practical demands of operating a commercial fact-checking service at an early-stage startup." (p. 1)

- 근거 유형: direct

**1c** (Ⅱ. 본론 2.1 선행 연구) — 논문: NN-PPI[1]는 소형 LLM의 점수를 라벨이 있는 보정 세트 중 의미적으로 가까운 이웃의 잔차로 보정한다.

> "For each test claim, we retrieve the k nearest neighbors to form a localized calibration set. The baseline LLM prediction is then adjusted using this set" (p. 5)

- 근거 유형: direct

## [2] Y. Li, Y. Miao, R. Krishnan, and R. Padman, "JEV-as-a-Judge: Accept When Confident, Escalate When Unsure," arXiv:2609.26550, 2026.

- 식별자: arXiv:2609.26550 (v3 확인) · 공개일 2026-09-22 · arXiv preprint
- 공식 저자: Yubo Li, Yidi Miao, Ramayya Krishnan, Rema Padman
- 원문 PDF: `pdfs/2609.26550.pdf`

**2a** (서론 1문단) — 논문: JEV는 텍스트를 생성하지 않고 라벨 확률만 반환하는 결정 전용 모델이다[2].

> "a hosted decision-only service that takes natural-language instructions and structured inputs and returns a verdict over a specified output type, with a probability for every label and no generated text" (p. 2)

- 근거 유형: direct

**2b** (서론 1문단) — 논문: 생성형 LLM보다 훨씬 저렴하다[2] (1,000건당 0.044달러).

> "it is 277 times cheaper ($0.044 per 1,000 judgments)" (p. 2)

- 근거 유형: direct — Fee comparison is against GPT-6 at low reasoning effort on their timing panel, not against Claude Sonnet 5.

**2c** (서론 2문단) — 논문: JEV가 확신할 때 판정을 받아들이고 나머지는 LLM의 판정으로 교체하는 방식이었다[2].

> "whose verdict replaces JEV’s" (p. 10)

- 근거 유형: direct

**2d** (서론 2문단) — 논문: 두 모델의 출력을 결합하는 방식은 향후 과제로 남아 있다[2].

> "Fusion, conditioning, and rules that model the correlation are left to future work." (p. 14)

- 근거 유형: direct

## [3] D. Rao and C. Callison-Burch, "JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places," arXiv:2609.29769, 2026.

- 식별자: arXiv:2609.29769 (v2 확인) · 공개일 2026-09-24 · arXiv preprint
- 공식 저자: Delip Rao, Chris Callison-Burch
- 원문 PDF: `pdfs/2609.29769.pdf`

**3a** (서론 2문단) — 논문: 이러한 캐스케이드는 비용을 줄일 뿐 정확도는 거의 높이지 못한다고 보고되었다[3].

> "such cascades only lower cost while adding little accuracy" (p. 1)

- 근거 유형: direct — Their cascades defer to an LLM judge whose verdict is used (replacement); rubric-judging panels, not check-worthiness.

**3b** (서론 2문단) — 논문: JEV와 LLM은 같은 곳에서 틀리는 경향이 있어[3]

> "these judges are wrong in the same places" (p. 1)

- 근거 유형: direct

## [4] Y. Zhang et al., "Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous Model Collaboration," arXiv:2609.11446, 2026.

- 식별자: arXiv:2609.11446 (v1 확인) · 공개일 2026-09-10 · arXiv preprint; comment: 'Under review'
- 공식 저자: Yilin Zhang, Han Jiang, Cai Xu, Ying Liu, Wei Zhao
- 원문 PDF: `pdfs/2609.11446.pdf`

**4a** (Ⅱ. 본론 2.1 선행 연구) — 논문: 위임한 입력에서 두 모델의 출력을 선택적으로 결합하는 캐스케이드도 일반 분류 과제에서 제안되었다[8].

> "For deferred inputs, CAUC selectively combines model outputs when their predictions are complementary." (p. 1)

- 근거 유형: direct

**4b** (Ⅱ. 본론 2.1 선행 연구) — 논문: (CAUC 결과: 호출 약 절반 절감, 강한 모델 단독보다 정확도 향상 — 우리 결과와 같은 방향)

> "CAUC achieves an average relative accuracy improvement of 1.9% over strong-model-only inference while avoiding approximately 47% of strong-model calls" (p. 1)

- 근거 유형: direct

**4c** (Ⅱ. 본론 2.1 선행 연구) — 논문: 본 논문은 ... 결합 가중치를 라벨로 학습하여 LLM과 라벨의 판정 기준 차이까지 보정한다.

> "This sign test is performed once on the calibration set and introduces no learned decision network." (p. 4)

- 근거 유형: inference — CAUC fuses standardized, temperature-calibrated logits weighted by each model's calibrated confidence (Eq. 6: z_fuse = (p_s q_s + p_l q_l)/(p_s + p_l)) and enables fusion only if a calibration-set sign test is positive. No fusion coefficient is fitted to labels. Our Eq. (1) fits (a, b, c) by logistic regression on labelled calibration data, so the fused score can re-weight an LLM whose decision criterion differs from the labels. The 'difference' sentence is our characterization based on these passages.

## [5] M. Hasanain et al., "Overview of the CLEF-2024 CheckThat! Lab Task 1 on Check-Worthiness Estimation of Multigenre Content," CEUR-WS vol. 3740, pp. 276–286, 2024.

- 식별자: https://ceur-ws.org/Vol-3740/paper-24.pdf · 공개일 2024 · CEUR Workshop Proceedings vol. 3740 (CLEF 2024 Working Notes)
- 공식 저자: Maram Hasanain, Reem Suwaileh, Sanne Weering, Chengkai Li, Tommaso Caselli, Wajdi Zaghouani, Alberto Barrón-Cedeño, Preslav Nakov, Firoj Alam
- 원문 PDF: `pdfs/clef2024_task1_overview.pdf`

**5a** (III. 실험 (데이터)) — 논문: CLEF 2024 CheckThat! Task 1 영어 (테스트: dev·dev-test·공식 test)

> "Task 1 involves determining whether a text item is check-worthy, with a special emphasis on COVID-19, political news, and political debates and speeches." (p. 1)

- 근거 유형: direct

**5b** (결론 (한계)) — 논문: 출처가 겹치는[5] 영어 두 데이터셋 — CLEF 영어 데이터는 ClaimBuster에서 구축

> "As for the English subset, it was sourced from the annotated dataset described by Arslan et al. [4], and consists of transcribed sentences from candidates during the US Presidential election debates." (p. 2)

- 근거 유형: direct — Consequence: the two benchmarks share a source corpus. Our measurement (2026-09-27, text match against the Zenodo ClaimBuster release): 2,405/2,406 CLEF calibration sentences are ClaimBuster sentences; only 3/318 CLEF dev-test sentences match ClaimBuster text (0 ids); 313 of the 2,745 ClaimBuster 2016 test sentences also occur in the CLEF calibration set. No evaluation leakage, because each dataset is calibrated only on its own calibration set.

**5c** (III. 실험 (데이터)) — 논문: 테스트 = dev 1,032 + dev-test 318 + 공식 test 341 = 1,691문장

> "Dev 411 682 102 150 238 794 704 4,296 Dev-test 377 123 316 350 108 210 509 4,491 Test 218 392 397 603 88 253" (p. 3)

- 근거 유형: direct — Table 1, columns Arabic/Dutch/English/Spanish (Yes, No). English: Dev 238+794=1,032; Dev-test 108+210=318; Test 88+253=341; total 1,691 = our test set. Calibration comes from Train only.

**5d** (결론 (한계)) — 논문: 출처가 겹치는[5] 영어 두 데이터셋 — CLEF 테스트 문장은 ClaimBuster에 없던 새 문장

> "The English test set was constructed by manually annotating transcribed sentences that did not appear in Arslan et al. [4]." (p. 2)

- 근거 유형: direct

## [6] F. Arslan, N. Hassan, C. Li, and M. Tremayne, "A Benchmark Dataset of Check-Worthy Factual Claims," in Proc. ICWSM, vol. 14, pp. 821–829, 2020.

- 식별자: DOI 10.1609/icwsm.v14i1.7346 · 공개일 2020-05-26 · Proc. Int. AAAI Conf. Web and Social Media (ICWSM), vol. 14, pp. 821-829 (Crossref)
- 공식 저자: Fatma Arslan, Naeemul Hassan, Chengkai Li, Mark Tremayne
- 원문 PDF: `pdfs/icwsm2020_claimbuster.pdf`

**6a** (III. 실험 (데이터)) — 논문: ClaimBuster (미국 대선 토론 문장)

> "we present the ClaimBuster dataset of 23,533 statements extracted from all U.S. general election presidential debates and annotated by human coders" (p. 1)

- 근거 유형: direct — We use the Zenodo release record 3836810 (the paper links an earlier version, 10.5281/zenodo.3609356); 23,533 = groundtruth 1,032 + crowdsourced 22,501.

## 검토했으나 인용하지 않은 논문

| arXiv | 제목 | 확인 깊이 | 판단 |
|---|---|---|---|
| 2609.37647 | Evaluating and Benchmarking the System One Model Jev | title/abstract (orx discover) | general JEV benchmark; not needed for our claims |
| 2609.24574 | Evaluating Decision Models for Text Annotation in Computational Social Science | title/abstract | annotation tasks, no cascade/fusion; not cited |
| 2609.28940, 2609.30216, 2609.24965, 2609.34024, 2609.23986, 2609.30186, 2609.36965 | other 2026 JEV application papers (pentest, ecosystem, science, medicine, memory, mobile, Chinese) | title (keyword search) | different domains; none on check-worthiness or fusion |

## 검색 기록

| 날짜 | 소스 | 질의 | 결과 |
|---|---|---|---|
| 2026-09-30 | alphaXiv keyword | JEV TypeSafe / JEV | 2609.26550, 2609.29769, 2609.37647 and JEV application papers |
| 2026-09-30 | alphaXiv embedding | zero-shot decision model as cheap first stage in a cascade with a large language model for classification | 2609.24574, cascade papers (SRR, conformal cascade ...) |
| 2026-09-30 | alphaXiv keyword | Jev check-worthiness / Jev fact-checking | no JEV paper on check-worthiness or fact-checking |
| 2026-09-30 | full text | 2609.26550 §7-8, 2609.29769 abstract/RQ3 | both cascades are serial replacement; 2609.26550 leaves fusion to future work |
