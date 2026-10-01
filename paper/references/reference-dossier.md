# 참고문헌 검증 자료 (cascade_kics_draft)

`verify_references.py`가 생성합니다. 모든 인용문은 `pdfs/`의 원문 PDF에서 그대로 찾은 문장이며, 쪽수는 PDF 쪽 번호입니다. `support: inference`는 원문에 그대로 쓰인 문장이 아니라 원문을 근거로 한 **우리의 해석**이라는 뜻입니다.

## 요약

- 참고문헌 8편, 근거 30건 (원문 직접 확인 27건, 해석 3건)
- 원문 PDF에서 찾은 인용문 30/30건 (유료 원문 0건은 공개 초록으로 확인)
- 해석에 해당하는 근거 (심사에서 질문받으면 note의 논리로 답해야 함): 1j (프런티어 LLM + NN-PPI가 원 논문에서 가장 강한 설정…), 3b (모든 입력에 LLM을 호출한다.…), 8c (본 논문은 ... 결합 가중치를 라벨로 학습하여 LLM과 라벨의 판정 기…)

## [1] P. Amatya, Venktesh V, and V. Setty, "Calibrating Small Language Models for Claim Check-Worthiness Detection," arXiv:2608.30731, 2026.

- 식별자: arXiv:2608.30731 (v2 확인) · 공개일 2026-08-31 · arXiv preprint (anonymised ARR code link: likely under ACL Rolling Review)
- 공식 저자: Pratuat Amatya, V Venktesh, Vinay Setty
- 원문 PDF: `pdfs/2608.30731.pdf`

**1a** (서론 1문단) — 논문: 실제 팩트체크 서비스 운영사는 모든 문장에 대형 LLM을 호출하는 것은 실제 서비스 규모에서 감당하기 어렵다고 보고하였다[1].

> "invoking large LLMs on every incoming claim is prohibitive at production volumes" (p. 1)

- 근거 유형: direct

**1b** (서론 1문단) — 논문: (운영사) — 저자들이 실제 서비스 운영 맥락에서 쓴 논문이라는 근거

> "This work is motivated by the practical demands of operating a commercial fact-checking service at an early-stage startup." (p. 1)

- 근거 유형: direct

**1c** (Ⅱ. 본론 2.1 선행 연구) — 논문: NN-PPI[1]는 소형 LLM의 점수를 라벨이 있는 보정 세트 중 의미적으로 가까운 이웃의 잔차로 보정한다.

> "For each test claim, we retrieve the k nearest neighbors to form a localized calibration set. The baseline LLM prediction is then adjusted using this set" (p. 5)

- 근거 유형: direct

**1d** (II. 제안 기법) — 논문: NN-PPI가 이웃 검색에 쓰는 문장 임베딩(all-MiniLM-L6-v2)

> "indexed in a ChromaDB vector store using all-MiniLM-L6-v2 sentence embeddings (Reimers and Gurevych, 2019) with cosine similarity" (p. 5)

- 근거 유형: direct

**1e** (III. 실험 (설정)) — 논문: NN-PPI는 원 논문의 Gemma 3 4B 설정으로 재현 — 샘플링 T=1.0, top-k=64, top-p=0.95 (scripts/score_gemma.py 기본값)

> "Gemma 3 4B 1.0 64 0.95" (p. 13)

- 근거 유형: direct — Appendix C sampling table (model, temperature, top-k, top-p).

**1f** (III. 실험 (재현 수준)) — 논문: 원 논문 0.827, 0.760 (Gemma 3 4B + NN-PPI, k=10의 가중 F1; CLEF, ClaimBuster)

> "10 0.568 0.707 0.760" (p. 6)

- 근거 유형: direct — Table 2, ClaimBuster block, Gemma 3 4B row k=10: Baseline 0.568, KNN 0.707, NN-PPI 0.760. CLEF block: see 1g.

**1g** (III. 실험 (재현 수준)) — 논문: 원 논문 0.827 (CLEF, Gemma 3 4B + NN-PPI, k=10)

> "10 0.688 0.807 0.827" (p. 6)

- 근거 유형: direct — Table 2, CLEF 2024 block, Gemma 3 4B row k=10.

**1h** (III. 실험 (데이터)) — 논문: ClaimBuster(2012년 토론 보정 1,314문장, 2016년 토론 테스트)

> "We use 2012 election data for calibration and 2016 data for testing." (p. 5)

- 근거 유형: direct

**1i** (III. 실험 (데이터)) — 논문: NN-PPI 재현 F1은 원 논문과 같은 CLEF dev-test 분할(318문장)에서 측정

> "CLEF 2024 2,406 (of 22,501) 317 107 210" (p. 5)

- 근거 유형: direct — NN-PPI's CLEF test = 317 (107 CW / 210 NCW); this is the CLEF dev-test partition (108/210, see 5c), not the official 341-sentence test. Our 318 = the same partition.

**1j** (표 1 'Sonnet 5 + NN-PPI [1]' 행) — 논문: 프런티어 LLM + NN-PPI가 원 논문에서 가장 강한 설정

> "5 0.855 0.801 0.899" (p. 6)

- 근거 유형: inference — Table 2, CLEF block, Claude Opus 4.6 k=5: NN-PPI weighted F1 0.899 (0.925 class-0 F1) - the highest CLEF rows are frontier LLM + NN-PPI. That this is the 'strongest setting' is our reading of Table 2, not the authors' wording.

**1k** (III. 실험 (프롬프트)) — 논문: NN-PPI의 판정 기준 프롬프트로 채점; few-shot 예시 6개는 원 논문에 공개되지 않아 직접 작성

> "The {{examples}} placeholder is filled with 6 fixed few-shot examples, one per check-worthiness tier" (p. 4)

- 근거 유형: direct — Figure 2 caption. The six examples themselves are not printed anywhere in the paper.

**1l** (III. 실험 (재현)) — 논문: 원 논문과 같은 ClaimBuster 2016년 분할 (원 논문 2,740문장, 본 논문 2,745문장)

> "ClaimBuster 1,314 (of 2,487) 2,740 725 2,015" (p. 5)

- 근거 유형: direct — NN-PPI Table 1: ClaimBuster test 2,740 (725 CW / 2,015 NCW). Our 2016 set built from the Zenodo release has 2,745 (728 CW); the 5-sentence difference is unexplained (likely de-duplication or parsing).

## [2] P. Amatya and V. Setty, "Multilingual Fact-Checking at Scale: Fine-Tuned Compact Models vs LLMs," arXiv:2606.08605, 2026.

- 식별자: arXiv:2606.08605 (v1 확인) · 공개일 2026-06-07 · arXiv preprint
- 공식 저자: Pratuat Amatya, Vinay Setty
- 원문 PDF: `pdfs/2606.08605.pdf`

**2c** (서론) — 논문: 같은 연구진은 소형 인코더가 확신하지 못하는 문장만 LLM에 넘기는 방식을 향후 과제로 제시하였다[2].

> "hybrid encoder-LLM systems that escalate to a generative model only when the encoder's confidence is low" (p. 9)

- 근거 유형: direct — Stated in the conclusion as 'Future work will focus on ...'.

## [3] M. Schlee et al., "LabelFusion: Fusing Large Language Models with Transformer Encoders for Robust Financial News Classification," arXiv:2512.10793, 2025.

- 식별자: arXiv:2512.10793 (v2 확인) · 공개일 2025-12-11 · arXiv preprint
- 공식 저자: Michael Schlee, Christoph Weisser, Timo Kivimäki, Melchizedek Mashiku, Benjamin Saefken
- 원문 PDF: `pdfs/2512.10793.pdf`

**3a** (Ⅱ. 본론 2.1 선행 연구) — 논문: LabelFusion[3]은 인코더와 LLM의 출력을 결합하지만

> "combines the output of a prompt-engineered LLM with contextual embeddings produced by a fine-tuned RoBERTa encoder through a lightweight Multilayer Perceptron" (p. 1)

- 근거 유형: direct

**3b** (Ⅱ. 본론 2.1 선행 연구) — 논문: 모든 입력에 LLM을 호출한다.

> "LabelFusion combines two complementary components" (p. 2)

- 근거 유형: inference — The method feeds every input text x to both components and fuses them (Sec. 3); no selective/gated LLM invocation is described anywhere in the paper. 'Calls the LLM for every input' is our reading of the architecture, not a sentence the authors wrote.

## [4] T. Burleigh, "Do Small Language Models Know When They're Wrong? Confidence-Based Cascade Scoring for Educational Assessment," arXiv:2604.19781, 2026.

- 식별자: arXiv:2604.19781 (v1 확인) · 공개일 2026-03-29 · arXiv preprint; comment: 'Accepted at NCME 2026'
- 공식 저자: Tyler Burleigh
- 원문 PDF: `pdfs/2604.19781.pdf`

**4a** (서론) — 논문: 가벼운 모델이 먼저 판정하고 확신하지 못하는 입력만 큰 모델에 넘기는 구조를 캐스케이드라 하며

> "small language models (LMs) handle easier scoring tasks while escalating harder ones to larger LMs" (p. 1)

- 근거 유형: direct

**4b** (서론) — 논문: 일반적으로 넘긴 입력의 판정은 큰 모델의 판정으로 교체한다[4].

> "the small LM's score is replaced with the large LM's score, and all other decisions keep the small LM's score" (p. 4)

- 근거 유형: direct

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

**5e** (Ⅰ. 서론 1문단) — 논문: 온라인 허위 정보가 급증하면서 팩트체크의 중요성이 커졌으며

> "Due to the significant surge of disinformative content online the importance of improving the capabilities of fact-checking pipeline is paramount" (p. 6)

- 근거 유형: direct

**5f** (Ⅰ. 서론 1문단) — 논문: 그 첫 단계는 사실 확인이 필요한 주장을 찾아내는 팩트체크 필요성 탐지이다[5].

> "the first part of the pipeline is finding claims that important to fact check" (p. 6)

- 근거 유형: direct

## [6] F. Arslan, N. Hassan, C. Li, and M. Tremayne, "A Benchmark Dataset of Check-Worthy Factual Claims," in Proc. ICWSM, vol. 14, pp. 821–829, 2020.

- 식별자: DOI 10.1609/icwsm.v14i1.7346 · 공개일 2020-05-26 · Proc. Int. AAAI Conf. Web and Social Media (ICWSM), vol. 14, pp. 821-829 (Crossref)
- 공식 저자: Fatma Arslan, Naeemul Hassan, Chengkai Li, Mark Tremayne
- 원문 PDF: `pdfs/icwsm2020_claimbuster.pdf`

**6a** (III. 실험 (데이터)) — 논문: ClaimBuster (미국 대선 토론 문장)

> "we present the ClaimBuster dataset of 23,533 statements extracted from all U.S. general election presidential debates and annotated by human coders" (p. 1)

- 근거 유형: direct — We use the Zenodo release record 3836810 (the paper links an earlier version, 10.5281/zenodo.3609356); 23,533 = groundtruth 1,032 + crowdsourced 22,501.

## [7] Z. Wang et al., "Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference," arXiv:2609.07786, 2026.

- 식별자: arXiv:2609.07786 (v1 확인) · 공개일 2026-09-07 · arXiv preprint
- 공식 저자: Zheyuan Wang, Siyu Li, Peiqiao Song, Sijia Chen, Qianqian Song, Qian Liu
- 원문 PDF: `pdfs/2609.07786.pdf`

**7a** (서론 2문단) — 논문: 이 경우 교체는 분류기가 이미 맞힌 판정까지 틀리게 바꿀 수 있다[7].

> "it is harmful when the large model replaces a correct answer with an incorrect one" (p. 1)

- 근거 유형: direct — SRR studies LLM-to-LLM cascades on MMLU/HellaSwag/ARC; our setting is classifier-to-LLM, so [7] is cited as consistent evidence, not as the same setting.

**7b** (Ⅱ. 본론 2.1 선행 연구) — 논문: SRR[7]은 옳은 답을 뒤집는 손실까지 예측하여 넘길 입력을 고르지만

> "a budgeted routing method that predicts these two events separately and ranks requests by their difference" (p. 1)

- 근거 유형: direct — 'these two events' = rescue (large model corrects the small one) and harm (large model replaces a correct answer), abstract.

**7c** (Ⅱ. 본론 2.1 선행 연구) — 논문: 넘긴 입력은 큰 모델의 답으로 교체한다.

> "if g ≥τq then" (p. 4)

- 근거 유형: direct — Algorithm 1, online routing: 'if g ≥ τq then return Mℓ(x) else return as' — an escalated request gets the large model's answer (replacement); Fig. 1 labels the same branch 'Final answer from Mℓ'.

## [8] Y. Zhang et al., "Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous Model Collaboration," arXiv:2609.11446, 2026.

- 식별자: arXiv:2609.11446 (v1 확인) · 공개일 2026-09-10 · arXiv preprint; comment: 'Under review'
- 공식 저자: Yilin Zhang, Han Jiang, Cai Xu, Ying Liu, Wei Zhao
- 원문 PDF: `pdfs/2609.11446.pdf`

**8a** (Ⅱ. 본론 2.1 선행 연구) — 논문: 위임한 입력에서 두 모델의 출력을 선택적으로 결합하는 캐스케이드도 일반 분류 과제에서 제안되었다[8].

> "For deferred inputs, CAUC selectively combines model outputs when their predictions are complementary." (p. 1)

- 근거 유형: direct

**8b** (Ⅱ. 본론 2.1 선행 연구) — 논문: (CAUC 결과: 호출 약 절반 절감, 강한 모델 단독보다 정확도 향상 — 우리 결과와 같은 방향)

> "CAUC achieves an average relative accuracy improvement of 1.9% over strong-model-only inference while avoiding approximately 47% of strong-model calls" (p. 1)

- 근거 유형: direct

**8c** (Ⅱ. 본론 2.1 선행 연구) — 논문: 본 논문은 ... 결합 가중치를 라벨로 학습하여 LLM과 라벨의 판정 기준 차이까지 보정한다.

> "This sign test is performed once on the calibration set and introduces no learned decision network." (p. 4)

- 근거 유형: inference — CAUC fuses standardized, temperature-calibrated logits weighted by each model's calibrated confidence (Eq. 6: z_fuse = (p_s q_s + p_l q_l)/(p_s + p_l)) and enables fusion only if a calibration-set sign test is positive. No fusion coefficient is fitted to labels. Our Eq. (1) fits (a, b, c) by logistic regression on labelled calibration data, so the fused score can re-weight an LLM whose decision criterion differs from the labels. The 'difference' sentence is our characterization based on these passages.

## 검토했으나 인용하지 않은 논문

| arXiv | 제목 | 확인 깊이 | 판단 |
|---|---|---|---|
| DOI 10.1007/978-3-031-71908-0_2 | Overview of the CLEF-2024 CheckThat! Lab (Barrón-Cedeño et al., LNCS, pp. 28-52) | public Springer abstract (paywalled) | Was [5] until 2026-09-27; replaced by the Task 1 overview, which the dataset card of iai-group/clef2024_checkthat_task1_en asks users to cite and which documents the English data provenance and split sizes. |
| 2609.26913 | COMED: The Missing Middle Between Routing and Collaboration in Multi-LLM Inference | full text read | Was [7] in an earlier draft; removed. Concerns LLM-LLM collaboration ('a correct model may also be swayed by noisy or misleading peer outputs'), only an analogy for classifier-to-LLM replacement. Replaced by SRR [7] plus our own flip counts. |
| 2603.25269 | When Hate Meets Facts: LLMs-in-the-Loop for Check-worthiness Detection in Hate Speech | keyword scan of full text | Check-worthiness + LLM annotation in hate speech; no cascade or selective LLM invocation found. Not overlapping. |
| 2603.03752 | Confidence-Calibrated Small-Large Language Model Collaboration for Cost-Efficient Reasoning | keyword scan of full text | SLM-to-LLM escalation for reasoning; the LLM response 'serves as the final answer' (replacement). Same family as [4]; not overlapping with fusion. |
| 2608.22879 | Large-Small Model Collaboration for Zero-Shot Surgical Phase Recognition | keyword scan of full text | Ensembling for pseudo-label training in surgical video; unrelated. |
| 2605.18796 | UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing | title only - NOT verified | Routing criterion paper by title; method not read. Re-check if a reviewer raises routing criteria. |
| 2607.25018 | Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference | title only - NOT verified | Deferral guarantees by title; method not read. |
| 2606.27457 | Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving | title only - NOT verified | Serving-side cascade by title; method not read. |
| 2608.17711 | Accuracy and Robustness of Model Cascades Under Data Perturbations | title only - NOT verified | Cascade robustness by title; method not read. |
| 2603.14828 | The CLEF-2025 CheckThat! Lab | title only | Later CheckThat! edition; not the data we use. |

## 검색 기록

| 날짜 | 소스 | 질의 | 결과 |
|---|---|---|---|
| 2026-09-27 | alphaXiv embedding | cascade that sends only uncertain inputs from a small classifier to a large language model and combines both scores instead of replacing |  |
| 2026-09-27 | alphaXiv keyword | check-worthiness cascade LLM |  |
| 2026-09-27 | alphaXiv keyword | deferral ensemble cascade combine small model large model predictions |  |
| 2026-09-27 | OpenAlex | model cascade deferral combining small and large model predictions | failed (HTTP 429), not retried |
| 2026-09-27 | alphaXiv keyword | CheckThat! 2024 overview check-worthiness subjectivity persuasion |  |
