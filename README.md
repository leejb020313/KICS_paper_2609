# 저비용 분류기와 대형 언어모델의 선택적 결합을 통한 팩트체크 필요성 탐지

**Cost-Efficient Check-Worthy Claim Detection via Selective Fusion of a Low-Cost Classifier and a Large Language Model**

이정빈, 김은경(교신저자) · 국립한밭대학교 · 2026 한국통신학회(KICS) 추계종합학술발표회 투고 논문의 코드·데이터·결과

---

## 요약

팩트체크 필요성 탐지(check-worthiness detection)는 유입되는 모든 문장에 적용되므로, 모든 문장에 대형 언어모델(LLM)을 호출하면 비용과 지연이 커진다. 본 저장소는 다음을 재현한다.

1. **NN-PPI 재현**: 선행연구 NN-PPI([arXiv:2608.30731](https://arxiv.org/abs/2608.30731))를 원 논문과 같은 Gemma 3 4B 설정으로 재현하였다. 같은 라벨과 문장 임베딩으로 SVM을 직접 학습하면 LLM 없이도 NN-PPI와 같거나 높은 정확도를 얻는다.
2. **프런티어 LLM의 과소 판정**: Claude Sonnet 5를 모든 문장에 그대로 적용하면 팩트체크가 필요한 문장을 과소 판정한다. 라벨로 임계값을 조정하면 재현율이 크게 오른다.
3. **결합형 캐스케이드(제안)**: 분류기가 가장 불확실한 문장에만 LLM을 호출하고, LLM의 판정으로 **교체하지 않고** 분류기 결정값과 LLM 점수를 로지스틱 회귀로 **결합**한다. 호출을 절반으로 줄여도 LLM 전량 호출과 유의한 차이가 없는 정확도를 얻는다. 다만 교체형도 호출률 50%에서 전량 호출 수준에 이르므로 호출 절감 자체는 캐스케이드 구조의 효과이고, 결합형은 같은 수준에 더 적은 호출률(CLEF 30%, ClaimBuster 40%)로 도달한다.

## 주요 결과

테스트 정확도이며, 보정 세트의 80%를 비복원 추출해 5회 반복한 평균이다. 호출률은 프런티어 LLM을 호출한 문장의 비율이다. 테스트는 CLEF 2024 영어의 dev·dev-test·공식 test를 합친 1,691문장, ClaimBuster 2016년 토론 전체 2,745문장이다.

| 방법 | 호출률 | CLEF 2024 (n=1,691) | ClaimBuster (n=2,745) |
|---|---:|---:|---:|
| Gemma 3 4B | 0% | 0.778 | 0.745 |
| + NN-PPI (k=10, 보정 세트에서 선택) | 0% | 0.834 | 0.785 |
| 임베딩 SVM | 0% | 0.859 | 0.790 |
| Claude Sonnet 5 전량 호출 | 100% | 0.896 | 0.835 |
| + 임계값 조정 | 100% | 0.915 | 0.830 |
| + NN-PPI (k=10, 보정 세트에서 선택) | 100% | 0.897 | 0.826 |
| 교체형 캐스케이드 | 50% | 0.916 | 0.835 |
| NN-PPI형 캐스케이드 (같은 문장 호출, Sonnet + NN-PPI로 판정) | 50% | 0.898 | 0.824 |
| **결합형 캐스케이드 (제안)** | **50%** | **0.923** | **0.846** |
| 결합형 캐스케이드 (제안) | 100% | 0.924 | 0.852 |

유의성은 5회 각각의 예측에 McNemar 검정(α=0.05)을 적용해 판단한다(`scripts/seed_robustness.py` → `results/seed_robustness_full.json`). 논문 표 1은 이 중 Gemma 원점수, NN-PPI형 캐스케이드, 결합형 100% 행을 빼고 7행만 싣는다.

- 결합형(50%)은 LLM 전량 호출 중 더 높은 설정(CLEF는 임계값 조정, ClaimBuster는 조정 없음)과 5회 모두 유의한 차이가 없다(평균 +0.8%p, +1.1%p, p≥0.10). NN-PPI를 적용한 전량 호출(선행연구의 가장 강한 설정)보다는 두 데이터셋 모두 5회 모두 유의하게 높다(p≤0.016).
- 교체형도 호출률 50%에서 전량 호출 수준에 이른다(CLEF +0.01%p, ClaimBuster +0.05%p). 즉 호출 절감은 캐스케이드 구조에서 오고, 결합은 같은 수준에 더 적은 호출(CLEF 30%, ClaimBuster 40%)로 도달하게 한다.
- 임베딩 SVM은 NN-PPI보다 CLEF에서 5회 중 4회 유의하게 높고 ClaimBuster에서는 5회 모두 차이가 없다.
- ClaimBuster에서는 클래스 균형인 보정 세트와 달리 테스트의 팩트체크 필요 문장이 27%라서, 임계값 조정이 재현율(0.42 → 0.86)을 올리는 대신 정확도를 0.5%p 낮춘다.
- 호출률 50%는 테스트가 아니라 보정 세트에서 정했다. 보정 세트의 교차적합 정확도는 50% 이후 0.1%p 이하로만 올라 포화되었고, 전량 호출과의 차이는 0.7%p 이내였다. (`final_eval.py`의 `fuse_sel`은 별도로 미리 정해 둔 규칙 "보정 정확도가 전량 호출에 도달하는 최소 호출률"의 결과로, 포화 수준이 전량 호출보다 조금 낮아 대부분 100%를 고른다. 논문 수치에는 쓰지 않았다.) 테스트에서 호출률을 100%로 올리면 0.1%p, 0.6%p 더 오르며, ClaimBuster에서는 이 차이가 5회 모두 유의하다.
- 결합형은 LLM을 호출하는 모든 호출률에서 교체형보다 평균 정확도가 높고, 그 차이는 CLEF 20~30%, ClaimBuster 20% 이상에서 5회 중 3회 이상 유의하다(ClaimBuster에서 회차에 따라 흔들리는 쪽은 임계값이 학습 표본에 따라 바뀌는 교체형이다: 표준편차 0.67%p 대 결합형 0.14%p). 같은 문장을 호출해 NN-PPI로 판정하는 캐스케이드보다도 유의하게 높다.
- 분류기가 확신하는 절반의 문장에서 교체형은 SVM의 옳은 판정 94건을 틀리게, 틀린 판정 66건을 옳게 바꾸었고 결합형은 각각 9건, 26건이었다(ClaimBuster, 1회차).
- 문장 단위 스트리밍 라우팅(보정 세트에서 |d| 문턱을 고정)에서는 정확도가 CLEF 0.920(실제 호출 38%), ClaimBuster 0.845(실제 호출 49%)였다.

![호출률에 따른 테스트 정확도](figures/cascade_budget_full.png)

## 방법

![결합형 캐스케이드의 구조](figures/method_diagram.png)

- **저비용 분류기**: `all-MiniLM-L6-v2` 문장 임베딩(NN-PPI가 이웃 검색에 쓰는 것과 같은 임베딩)을 입력으로 받는 RBF-SVM이다. 클래스 균형 가중치를 쓰며, 하이퍼파라미터는 기본값이다. CPU에서 문장당 약 6 ms가 걸린다.
- **라우팅**: 결정값 $|d(x)|$가 작은 순서로 비율 $\rho$만큼의 문장에 LLM을 호출한다.
- **결합**: $\hat y(x) = \mathbb 1[\sigma(a\,d(x) + b\,s(x) + c) \ge 0.5]$. $(a,b,c)$는 보정 세트에서 5겹 교차적합으로 얻은 $d$와 LLM 점수 $s$로 학습한 로지스틱 회귀 계수다.
- **교체형(비교군)**: 같은 문장을 선택하되, 보정 세트에서 고른 임계값을 LLM 점수에 적용한 판정으로 교체한다.
- **누수 방지**: LLM 임계값, 결합 계수, NN-PPI의 k, 호출률은 모두 보정 세트 안에서만 정했다.

## 저장소 구조

```
cwcascade/                 공통 라이브러리
  data.py                  데이터·LLM 점수 로딩(전체 테스트셋 포함), 임베딩, 임계값 탐색
  nnppi.py                 NN-PPI 베이스라인
  prompt.py                NN-PPI 판정 기준 프롬프트 (Gemma few-shot)
scripts/
  final_eval.py            본 실험 (--full: 논문의 전체 테스트셋)   -> results/final_results_full.json
  streaming_eval.py        문장 단위 라우팅 점검 (--full)           -> results/streaming_results_full.json
  seed_robustness.py       5회 각각의 McNemar 검정, 호출률별 교체형/결합형 대 전량 호출  -> results/seed_robustness_full.json
  sanity_checks.py         배치/단건 채점 AUC, 재채점 상관, SVM 지연시간
  example_case.py          본문 3.2절의 판정 기준 예시 (CLEF 1회차 계수, 예시 문장)  -> results/example_case.json
  verify_paper_numbers.py  논문 PDF의 모든 수치와 유의성 주장을 results/*.json과 대조
  make_figure.py           그림 2 (--full)                         -> figures/cascade_budget_full.{pdf,png}
  figstyle.py              그림 공통 스타일 (크기, 글꼴, 색)
  score_gemma.py           Gemma 3 4B 채점 (llama.cpp 서버)
  build_frontier_prompts.py / run_frontier_batch.sh   Claude Sonnet 5 배치 프롬프트 생성·채점
data/
  build_datasets.py        원본 → data/processed 분할 생성
  build_full_sets.py       전체 테스트셋 중 800/318문장 표본 밖의 문장 (data/frontier/full_set.json)
  processed/               CLEF 2024, ClaimBuster 보정/테스트 분할 (CSV)
  frontier/                프런티어 LLM 평가 세트 (eval_set.json, calib_set.json, full_set.json)
results/
  gemma/                   Gemma 3 4B 원시 응답과 점수
  frontier/batches/        Claude Sonnet 5 배치 프롬프트(.txt)와 응답(.out)
  frontier/single/         CLEF 80문장 단건 채점 (배치 채점 타당성 점검용; JSON이 없는 11개는 CLI 사용량 한도로 실패한 응답 그대로)
  frontier/rerun/          CLEF 테스트 2차 배치 채점 (안정성 점검용)
  *.json, *.log            평가 결과와 실행 로그 (*_full: 논문 수치, 나머지: 이전 표본 318/800문장)
figures/                   논문 그림 (method_diagram.tex: 그림 1, TikZ; cascade_budget_full: 그림 2;
                           cascade_budget: --full 없이 만든 이전 표본의 같은 그림)
paper/
  cascade_kics_draft.{docx,pdf}  투고 원고 (교신저자 이메일 입력 전). 여백·글꼴·크기는 KICS 공식 워드 양식
                           (conf.kics.or.kr/2026f의 "양식-논문샘플(워드).doc")을 따른다
  build_docx.py            results/*.json과 figures/에서 원고 docx 생성
  equations/               식을 LaTeX로 렌더링한 이미지와 스크립트
  references/              참고문헌 검증 자료: 인용 문장별 원문 발췌·쪽수(references.json,
                           reference-dossier.md), 검증 스크립트, 원문 PDF 다운로드 스크립트
```

## 재현 방법

[uv](https://docs.astral.sh/uv/)가 필요하다. 모든 명령은 저장소 루트에서 실행한다.

```bash
uv sync --locked --all-extras

# 1) 평가 (저장된 LLM 점수를 사용하며 CPU에서 수십 분 소요)
uv run --locked python scripts/final_eval.py --full
uv run --locked python scripts/streaming_eval.py --full
uv run --locked python scripts/seed_robustness.py      # 5회 각각의 유의성 검정
uv run --locked python scripts/sanity_checks.py
(cd scripts && uv run --locked --with matplotlib python make_figure.py --full)
(cd figures && tectonic method_diagram.tex)     # 그림 1 (XeTeX, Arial)

# 2) 원고 생성 (docx; Word에서 PDF로 저장)
uv run --isolated --no-project --with python-docx --with pymupdf python paper/build_docx.py

# 3) 원고 검증: 본문 수치, 참고문헌 원문 대조
uv run --locked --with pymupdf python scripts/verify_paper_numbers.py paper/cascade_kics_draft.pdf   # -> ALL MATCH
bash paper/references/fetch_sources.sh     # 인용 논문 PDF 다운로드 (저작권 때문에 레포에는 없음)
uv run --no-project --with pymupdf python paper/references/verify_references.py paper/cascade_kics_draft.pdf --online   # -> ALL REFERENCES VERIFIED
```

평가는 커밋된 LLM 점수(`results/gemma`, `results/frontier`)만 사용하므로 GPU나 API 키가 필요 없다. `--full` 없이 실행하면 이전 표본(CLEF dev-test 318, ClaimBuster 800문장) 결과인 `results/final_results.json`이 동일하게 재생성된다.

LLM 점수를 처음부터 다시 만들려면 다음을 실행한다. 샘플링이 있으므로 점수는 조금씩 달라질 수 있다.

```bash
# 원본 데이터를 data/raw/에 받은 뒤 분할 생성 (출처는 아래 '데이터' 참고)
uv run --locked python data/build_datasets.py
uv run --locked python data/build_full_sets.py

# Gemma 3 4B: llama.cpp 서버에서 NN-PPI 부록 C 설정(T=1.0, top-k=64, top-p=0.95)
llama-server -m google_gemma-3-4b-it-Q4_K_M.gguf --port 8080 -c 8192 -ngl 99
uv run --locked python scripts/score_gemma.py --dataset data/processed/clef_test.csv --out results/gemma/clef_test_scores.jsonl
#   (clef_calib, claimbuster_calib, claimbuster_test도 같은 방식; CLEF dev와 공식 test는 --out results/gemma/clef_test_scores_full.jsonl)

# Claude Sonnet 5: Claude Code CLI(`claude -p`)로 40문장씩 배치 채점
uv run --locked python scripts/build_frontier_prompts.py
uv run --locked python scripts/build_frontier_prompts.py full_set.json
cd results/frontier && ls batches/*.txt | xargs -P 6 -n 1 bash ../../scripts/run_frontier_batch.sh
```

## 데이터

| 데이터셋 | 출처 | 라이선스 | 사용 분할 |
|---|---|---|---|
| CLEF 2024 CheckThat! Task 1 (영어) | [HF `iai-group/clef2024_checkthat_task1_en`](https://huggingface.co/datasets/iai-group/clef2024_checkthat_task1_en), 공식 test 정답은 [CheckThat! GitLab](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab) | CC BY-SA 4.0 | 보정: train에서 클래스 균형 2,406문장(Gemma 파싱 성공 2,405) / 테스트: dev 1,032 + dev-test 318 + 공식 test 341 = 1,691문장 (NN-PPI 재현 수치는 원 논문과 같은 dev-test 분할) |
| ClaimBuster | [Zenodo 3836810](https://zenodo.org/records/3836810) | CC BY 4.0 | 보정: 2012년 토론 1,314문장 / 테스트: 2016년 토론 2,745문장 (NN-PPI 원 논문은 2,740문장) |

`data/processed/`의 CSV는 위 원본에서 `data/build_datasets.py`, `data/build_full_sets.py`로 만든 파생물이며 원본 라이선스를 따른다.

CLEF 2024 영어 데이터는 ClaimBuster 말뭉치에서 가져왔다(Hasanain et al., CEUR-WS Vol. 3740, pp. 276–286). 실제로 확인해 보면 CLEF 보정 문장 2,406개 중 2,405개가 ClaimBuster에 있다. 즉 두 데이터셋의 출처는 독립적이지 않다. ClaimBuster 테스트 2,745문장 중 313문장은 CLEF 보정 세트에도 있지만, 각 데이터셋은 자기 보정 세트만으로 학습·보정하므로 평가 누수는 없다. CLEF 안에서는 dev의 1문장이 보정 세트와 텍스트가 같다(1,691문장 중 1문장).

## 유의 사항

- **이전 표본**: 초기 실험은 프런티어 LLM 비용 때문에 ClaimBuster 2016년 테스트 중 800문장(추출 시드 미기록, `data/frontier/eval_set.json`으로 고정)과 CLEF dev-test 318문장만 사용했다. 논문은 나머지 문장을 추가로 채점한 전체 테스트셋 결과(`*_full`)를 보고한다.
- **Few-shot 예시**: NN-PPI는 few-shot 예시 6개를 공개하지 않았다. `cwcascade/prompt.py`의 예시는 원 논문의 6단계 기준에 맞춰 직접 작성한 것이다.
- **LLM 모델**: `claude -p --model sonnet`의 모든 채점 호출(2026-09-25~26, 09-28)은 Claude Code 세션 기록상 `claude-sonnet-5`였다.
- **배치 채점**: 프런티어 LLM은 한 번에 40문장씩 채점하였다. CLEF 80문장을 단건으로도 채점했는데, 그중 11건은 CLI 사용량 한도 초과로 응답이 없었다. 나머지 69문장에서 ROC-AUC는 단건 0.991, 배치 0.990으로 같았다. CLEF 테스트를 두 번 배치 채점한 점수의 상관계수는 0.973이다.
- **선행 연구와의 관계**: 위임한 입력에서 두 모델의 출력을 선택적으로 결합하는 캐스케이드는 일반 분류 과제에서 CAUC([arXiv:2609.11446](https://arxiv.org/abs/2609.11446))가 먼저 제안하였다. 본 연구는 팩트체크 필요성 탐지에 이를 적용하면서 결합 가중치를 라벨로 학습한다.
- **한계**: 영어 데이터셋 두 개와 프런티어 모델 하나로만 평가하였고, 비용은 호출률로만 보고하였다.

## 다른 브랜치

이 브랜치(`release`)에는 논문에 쓰인 코드만 정리해 두었다. 연구 과정의 탐색 기록은 다음 브랜치에 그대로 남아 있다.

| 브랜치 | 내용 |
|---|---|
| `full-benchmark` | 전체 테스트셋 확장과 원고 개정 작업 이력 (이 브랜치에 병합됨) |
| `fusion-cascade` | 결합형 캐스케이드 실험의 원본 커밋 이력 (탐색용 스크립트 포함) |
| `nnppi-reproduction` | NN-PPI 재현 및 신뢰구간 커버리지 보정 연구 (`REPORT.md`) |
| `prompt-reorder-ablation` | 프롬프트·채점 방식 변형 실험과 임베딩 분류기 발견 과정 |
| `safety-flag-extension` | 다른 도메인(Safety-Flag)으로의 전이 실험 |
| `topic-venue-research` | 주제 선정과 학회 적합성 검토 기록 |
| `ct22-tweets` | 세 번째 데이터셋 시험: CheckThat! 2022 Task 1A 영어 트윗([GitLab](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab), "free for general research use"). 미리 정한 기준(`results/ct22_preregistration.md`) 중 결합형 > LLM 전량 호출(0.746 대 0.596)과 결합형 > 교체형은 성립했으나, 결합형 > NN-PPI 적용 전량 호출(0.756)은 성립하지 않아 논문에는 넣지 않았다. 이 브랜치의 채점은 `--restricted` 모드라 9월 채점과 조건이 조금 다르다(`results/frontier/modelcheck`). |

## 참고 문헌

- P. Amatya, Venktesh V, V. Setty, "Calibrating Small Language Models for Claim Check-Worthiness Detection," arXiv:2608.30731, 2026.
- M. Hasanain et al., "Overview of the CLEF-2024 CheckThat! Lab Task 1 on Check-Worthiness Estimation of Multigenre Content," CEUR-WS vol. 3740, pp. 276–286, 2024.
- F. Arslan, N. Hassan, C. Li, M. Tremayne, "A Benchmark Dataset of Check-Worthy Factual Claims," ICWSM, 2020.
