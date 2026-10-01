# 저비용 분류기와 대형 언어 모델의 선택적 결합을 통한 팩트체크 필요성 탐지

**Cost-Efficient Check-Worthy Claim Detection via Selective Fusion of a Low-Cost Classifier and a Large Language Model**

이정빈, 김은경(교신저자) · 국립한밭대학교 · 2026 한국통신학회(KICS) 추계종합학술발표회 학부논문 투고작의 원고, 코드, 데이터, 결과

- 원고: [`paper/cascade_kics_draft.pdf`](paper/cascade_kics_draft.pdf) (A4 2쪽), [`.docx`](paper/cascade_kics_draft.docx)
- 교수님 검토용 설명 자료: [`docs/professor-briefing.md`](docs/professor-briefing.md)

---

## 한눈에 보기

팩트체크 필요성 탐지(check-worthiness detection)는 수많은 문장 중 사실 확인이 필요한 문장을 골라내는 작업이다. 모든 문장에 대형 언어 모델(LLM)을 호출하면 비용과 시간이 크게 든다.

이 저장소의 방법(**결합형 캐스케이드**)은 이렇게 동작한다.
1. 가벼운 분류기(임베딩 SVM)가 모든 문장을 먼저 판정한다.
2. 분류기가 가장 확신하지 못하는 **절반**에만 LLM을 호출한다.
3. 호출한 문장은 LLM의 판정으로 **교체하지 않고**, 분류기의 결정값과 LLM 점수를 라벨로 학습한 가중치로 **결합**해 판정한다.

![결합형 캐스케이드의 구조](figures/method_diagram.png)

| Claude Sonnet 5, 테스트 정확도 (5회 평균) | LLM을 모든 문장에 호출 | 결합형 캐스케이드 (절반만 호출) |
|---|---:|---:|
| 정확도 CLEF 2024 / ClaimBuster | 0.915 / 0.835 | **0.923 / 0.846** |
| LLM 비용, 1,000문장당 (추정) | \$0.168 | \$0.084 |
| 순차 처리 시간, 1,000문장당 (추정) | 107초 | 60초 |

- 정확도는 평균적으로 오히려 0.8~1.1%p 높고, 5회 모두 "전량 호출보다 0.6%p 넘게 낮지 않다"를 95% 신뢰구간으로 확인했다.
- 같은 호출률에서 LLM 답으로 교체하는 방식(교체형)은 ClaimBuster에서 이를 보장하지 못했다.
- 비용과 시간은 테스트 배치 일부를 다시 호출해 추정한 값이라, 논문에는 "약 50%, 약 44% 줄일 수 있을 것으로 기대된다"로 적었다.

## 주요 결과

### 표 1: 테스트 정확도

보정(학습) 세트의 80%를 비복원 추출해 5회 반복한 평균이다. 테스트는 CLEF 2024 영어의 dev·dev-test·공식 test를 합친 1,691문장, ClaimBuster 2016년 토론 2,745문장이다. †는 5회 중 3회 이상 제안 방법보다 유의하게 낮음(McNemar, α=0.05)을 뜻한다.

| 방법 | 호출률 | CLEF 2024 | ClaimBuster |
|---|---:|---:|---:|
| Gemma 3 4B + NN-PPI (재현, k=10) | 0% | 0.834 | 0.785 |
| 임베딩 SVM | 0% | 0.859 | 0.790 |
| Claude Sonnet 5 전량 호출 | 100% | 0.896† | 0.835 |
| + 임계값 조정 | 100% | 0.915 | 0.830† |
| + NN-PPI | 100% | 0.897† | 0.826† |
| 교체형 캐스케이드 | 50% | 0.916 | 0.835† |
| **결합형 캐스케이드 (제안)** | **50%** | **0.923** | **0.846** |
| *(논문 표 외)* Gemma 3 4B 원점수 | 0% | 0.778 | 0.745 |
| *(논문 표 외)* NN-PPI형 캐스케이드 (같은 문장, Sonnet + NN-PPI로 판정) | 50% | 0.898 | 0.824 |
| *(논문 표 외)* 결합형 캐스케이드 | 100% | 0.924 | 0.852 |

![호출률에 따른 테스트 정확도](figures/cascade_budget_full.png)

### 핵심 주장: 절반만 호출해도 정확도는 그대로 (비열등성)

"전량 호출"은 두 설정(조정 전 / 임계값 조정) 중 정확도가 높은 쪽이다(CLEF는 임계값 조정, ClaimBuster는 조정 전). 아래는 정확도 차이(캐스케이드 − 전량 호출)의 95% 부트스트랩 신뢰구간 하한 중, 5회 가운데 가장 나빴던 값이다.

| 호출률 50% | CLEF | ClaimBuster |
|---|---:|---:|
| **결합형 (Sonnet 5)** | **−0.35%p** | **−0.58%p** |
| 교체형 (Sonnet 5) | −0.59%p | −2.26%p |
| 결합형 (Haiku 4.5) | −1.60%p | −1.38%p |
| 교체형 (Haiku 4.5) | −3.08%p | −2.66%p |

- 결합형은 두 데이터셋 모두 5회 모두 0.6%p 이내다(Sonnet 5). 교체형은 ClaimBuster에서 하한이 −2.3%p까지 내려간다.
- 0.6%p는 미리 정한 기준이 아니라 관측된 최악 하한(−0.58%p)을 올림한 값이다. 논문에는 사실 진술로만 적었다.
- 평균으로는 결합형이 CLEF 30%, ClaimBuster 40% 호출률에서 이미 전량 호출 정확도에 이른다(교체형은 50%). 다만 30~40%에서는 위의 0.6%p 보장이 성립하지 않는다(40%의 최악 하한: CLEF −0.65%p, ClaimBuster −1.39%p).
- 결합형(50%)은 NN-PPI를 적용한 전량 호출보다 두 데이터셋 모두 5회 모두 유의하게 높다(p≤0.016).

### 두 번째 LLM: Claude Haiku 4.5

같은 프롬프트와 배치로 다시 채점했다. 결과를 보기 전에 정한 기준([`results/analysis/haiku_preregistration.md`](results/analysis/haiku_preregistration.md))으로 판정하면 이렇다.

| 기준 | 결과 |
|---|---|
| 1. 결합형(50%)이 전량 호출보다 유의하게 낮지 않다 | **통과** (0.929 / 0.850 대 0.934 / 0.849, 5회 중 0회) |
| 2. 결합형(50%)이 NN-PPI 적용 전량 호출보다 유의하게 높다 | **실패** (0.930 / 0.839, 5회 중 1회씩). 논문에 그대로 적음 |
| 3. 결합형 평균 ≥ 교체형 평균 | **통과** (교체형 0.920 / 0.844) |

### 한계: 놓치는 문장(재현율)과 대응

| 재현율 | CLEF | ClaimBuster |
|---|---:|---:|
| 임베딩 SVM | 0.848 | **0.587** |
| 전량 호출 + 임계값 조정 | 0.942 | 0.859 |
| 결합형 50% | 0.923 | **0.735** |

- ClaimBuster는 2012년 토론으로 학습하고 2016년 토론으로 시험한다. 분류기는 시기가 바뀐 데이터에서 재현율이 크게 떨어지고(테스트 재현율 CLEF 0.85 대 ClaimBuster 0.59), 결합형도 이를 물려받아 '필요' 문장을 더 놓친다. 정확도가 오히려 높은 것은 잘못된 경보가 적기 때문이다(테스트의 '필요' 문장 비율 27%).
- 학습 세트에서 재현율 목표로 판정 기준을 다시 잡아도 0.737 → 0.764로만 회복된다([`scripts/recall_target.py`](scripts/recall_target.py)). 변화가 테스트에서만 드러나기 때문이다.
- **감지**: LLM을 호출한 문장 중 "LLM은 '필요', 분류기는 '불필요'"인 비율이 ClaimBuster에서만 오른다(학습 15.9% → 테스트 23.6%; CLEF는 14.8% → 11.7%). 라벨 없이 분류기 노후화를 알아챌 수 있다.
- **대응**: 2016년 문장 300개를 학습에 추가하면 결합형 재현율이 3.7%p 오른다(0.738 → 0.775, 정확도 유지; [`scripts/drift_remedy.py`](scripts/drift_remedy.py)).
- 이 두 분석은 결과를 본 뒤에 설계한 사후 분석이다. 논문에는 관찰로만 적었다.

### 설계 선택의 근거 (보조 분석)

같은 Sonnet 점수로 첫 단계와 결합 공식을 바꿔 비교했다([`results/analysis/ablation_stage_fusion.json`](results/analysis/ablation_stage_fusion.json)). 이 비교는 표본 추출 방식이 본 실험과 조금 달라 결합형 50%가 0.920 / 0.847로 나온다. 모든 행을 같은 방식으로 비교했다.

| 첫 단계 (결합 공식은 동일) | 결합형 50% | 문장당 CPU 시간 |
|---|---:|---:|
| 임베딩 SVM (논문) | 0.920 / 0.847 | 6.6 ms |
| 같은 임베딩 + 로지스틱 회귀 | 0.919 / 0.848 | 비슷 |
| 임베딩 SVM, C 튜닝 | 0.921 / 0.845 | 비슷 |
| 더 큰 임베딩(all-mpnet-base-v2) + SVM | 0.925 / 0.856 | 43 ms |

| 결합 공식 (첫 단계는 동일) | 결합형 50% |
|---|---:|
| LR(d, s) (논문 식 1) | 0.920 / 0.847 |
| LR(d, s, d·s) | 0.920 / 0.848 |
| LR(d, logit s) | 0.917 / 0.840 |
| 그래디언트 부스팅(d, s) | 0.916 / 0.843 |
| 학습 없이 평균 (Platt 보정한 d와 s) | 0.915 / 0.836 |

- 더 큰 임베딩은 더 정확하지만 첫 단계가 6.5배 느려져 시간 절감 효과가 크게 줄어든다. 논문의 임베딩은 NN-PPI와 같아 "같은 정보로 비교"가 성립한다.
- 결합 가중치를 학습하지 않으면 확실히 떨어진다. 라벨로 학습한 결합이 핵심이라는 근거다.
- 호출되는 구간만으로 결합 모델을 학습해도 나아지지 않았다([`scripts/ablation_fusion_region.py`](scripts/ablation_fusion_region.py)).
- 문장마다 학습 세트에서 정한 \|d\| 문턱으로 호출을 정하는 스트리밍 방식에서도 정확도는 0.920 / 0.845였다(실제 호출률 38% / 49%).

## 방법

- **저비용 분류기**: `all-MiniLM-L6-v2` 문장 임베딩(NN-PPI가 이웃 검색에 쓰는 것과 같음)을 입력으로 하는 RBF-SVM. 클래스 균형 가중치, 기본 하이퍼파라미터. CPU에서 임베딩 포함 문장당 약 6 ms.
- **호출**: 결정값 $|d(x)|$가 작은 순서로 비율 $\rho$(50%)만큼의 문장에 LLM을 호출한다.
- **결합**: $\hat y(x) = \mathbb 1[\sigma(a\,d(x) + b\,s(x) + c) \ge 0.5]$. $(a,b,c)$는 보정 세트에서 5겹 교차적합으로 얻은 $d$와 LLM 점수 $s$로 학습한 로지스틱 회귀 계수다. $s$에 대해 정리하면 $s \ge (-c - a\,d)/b$이므로, 분류기가 '불필요'로 기울수록($d$가 작을수록) LLM에 더 높은 점수를 요구한다.
- **교체형(비교군)**: 같은 문장을 호출하되, 보정 세트에서 정한 고정 임계값 $t$로 LLM 점수만 보고 판정한다.
- **누수 방지**: LLM 임계값, 결합 계수, NN-PPI의 k, 호출률은 모두 보정 세트 안에서만 정했다.

## 논문의 주장과 근거 파일

| 원고 위치 | 주장 | 결과 파일 | 만든 스크립트 |
|---|---|---|---|
| 표 1, 3.3절 | 각 방법의 정확도, 재현율, 정밀도 | `results/paper/final_results_full.json` | `scripts/final_eval.py --full` |
| 3.1절 | NN-PPI 재현 가중 F1 (원 논문 분할) | `results/paper/final_results.json` | `scripts/final_eval.py` |
| 3.3절, 표 1의 † | 5회 각각의 McNemar 검정, 호출률별 도달 지점 | `results/paper/seed_robustness_full.json` | `scripts/seed_robustness.py` |
| 요약, 3.3절 | 0.6%p 비열등성, 교체형과의 차이 | `results/paper/equivalence.json` | `scripts/equivalence.py` |
| 요약, 3.3절 | 비용·시간 추정 | `results/paper/cost_time.json` | `scripts/cost_time.py` |
| 3.3절 | Haiku 4.5 결과 | `results/paper/seed_robustness_full_haiku45.json` | `CWC_LLM=haiku45 scripts/seed_robustness.py` |
| 3.3절 | 사례 문장 (s=0.35, 기준 0.41) | `results/paper/example_case.json` | `scripts/example_case.py` |
| 3.3절 | 재현율 한계의 감지와 대응 | `results/analysis/drift_remedy.json` | `scripts/drift_remedy.py` |
| 그림 2 | 호출률-정확도 곡선 | `figures/cascade_budget_full.{pdf,png}` | `scripts/make_figure.py --full` |

원고의 모든 숫자는 `paper/build_docx.py`가 위 결과 파일에서 직접 읽어 넣는다. `scripts/verify_paper_numbers.py`가 PDF의 숫자와 주장 방향을 결과 파일과 대조하고, `paper/references/verify_references.py`가 인용 문장마다 원문 PDF의 해당 쪽에서 문구를 찾아 확인한다.

## 저장소 구조

```
paper/
  cascade_kics_draft.{pdf,docx}   투고 원고 (KICS 공식 워드 양식의 여백·글꼴·크기)
  build_docx.py                   results/paper/*.json과 figures/에서 원고 docx 생성
  equations/                      식 이미지와 렌더링 스크립트
  references/                     인용 문장별 원문 발췌·쪽수(references.json, reference-dossier.md),
                                  검증 스크립트, 원문 PDF 다운로드 스크립트
docs/
  professor-briefing.{md,docx,pdf}  교수님 검토용 설명 자료
figures/
  method_diagram.tex              그림 1 (TikZ)
  cascade_budget_full.{pdf,png}   그림 2 (cascade_budget.*: 이전 표본으로 그린 같은 그림)
results/
  paper/                          논문 숫자가 나오는 결과 파일 8개
  analysis/                       논문 밖 보조 분석 (대안 비교, 재현율 기준점, 노후화 감지·재학습,
                                  Haiku 사전 기준, 이전 표본의 스트리밍 결과)
  logs/                           실행 로그
  llm_scores/
    gemma/                        Gemma 3 4B 원시 응답과 점수
    sonnet5/                      Claude Sonnet 5 배치 프롬프트(.txt)와 응답(.out)
    haiku45/                      Claude Haiku 4.5 응답 (.out: 텍스트, .json: 모델 ID 포함 원본)
    checks/single/                CLEF 80문장 단건 채점 (배치 채점 타당성 점검)
    checks/rerun/                 CLEF 테스트 2차 배치 채점 (안정성 점검)
    checks/sonnet5_cost/          비용·시간 측정용 재호출 23배치와 CLI 오버헤드 측정
cwcascade/                        공통 라이브러리 (데이터·LLM 점수 로딩, 임베딩, NN-PPI, 프롬프트)
scripts/                          평가, 분석, 그림, LLM 채점 스크립트
data/
  processed/                      CLEF 2024, ClaimBuster 보정/테스트 분할 (CSV)
  frontier/                       LLM 평가 세트 (eval_set.json, calib_set.json, full_set.json)
  build_datasets.py, build_full_sets.py   원본에서 위 분할을 만드는 스크립트
```

### scripts/ 안내

| 역할 | 스크립트 |
|---|---|
| 본 평가 | `final_eval.py` (표 1), `seed_robustness.py` (회차별 검정), `equivalence.py` (비열등성), `streaming_eval.py` (문장 단위 호출) |
| 원고 숫자의 보조 결과 | `cost_time.py` (비용·시간), `example_case.py` (사례 문장), `drift_remedy.py` (노후화 감지·재학습) |
| 보조 분석 | `ablation_stage_fusion.py`, `ablation_fusion_region.py`, `recall_target.py`, `sanity_checks.py` |
| 그림 | `make_figure.py`, `figstyle.py` |
| 원고 검증 | `verify_paper_numbers.py` |
| LLM 채점 | `score_gemma.py`, `build_frontier_prompts.py`, `run_frontier_batch.sh` (Sonnet 5), `run_haiku_batch.sh`, `run_sonnet_costcheck.sh` |

## 재현 방법

[uv](https://docs.astral.sh/uv/)가 필요하다. 모든 명령은 저장소 루트에서 실행한다. 평가는 커밋된 LLM 점수(`results/llm_scores/`)만 쓰므로 GPU나 API 키가 필요 없다.

```bash
uv sync --locked --all-extras

# 1) 평가 (CPU에서 수십 분)
uv run --locked python scripts/final_eval.py --full
uv run --locked python scripts/final_eval.py                 # NN-PPI 재현 F1 (원 논문 분할)
uv run --locked python scripts/streaming_eval.py --full
uv run --locked python scripts/seed_robustness.py
CWC_LLM=haiku45 uv run --locked python scripts/seed_robustness.py
uv run --locked python scripts/equivalence.py
uv run --locked python scripts/example_case.py
uv run --locked python scripts/drift_remedy.py
uv run --locked python scripts/cost_time.py                  # SVM 시간은 다시 재므로 기기에 따라 조금 달라짐
(cd scripts && uv run --locked --with matplotlib python make_figure.py --full)
(cd figures && tectonic method_diagram.tex)                   # 그림 1 (XeTeX, Arial)

# 2) 원고 생성 (docx를 Word에서 PDF로 저장)
uv run --isolated --no-project --with python-docx --with pymupdf python paper/build_docx.py

# 3) 원고 검증
uv run --locked --with pymupdf python scripts/verify_paper_numbers.py paper/cascade_kics_draft.pdf   # -> ALL MATCH
bash paper/references/fetch_sources.sh                        # 인용 논문 PDF 다운로드 (저작권상 저장소에 없음)
uv run --no-project --with pymupdf python paper/references/verify_references.py paper/cascade_kics_draft.pdf --online   # -> ALL REFERENCES VERIFIED
```

LLM 점수를 처음부터 다시 만들려면 다음을 실행한다. 샘플링이 있어 점수는 조금씩 달라질 수 있다.

```bash
# 원본 데이터를 data/raw/에 받은 뒤 분할 생성 (출처는 아래 '데이터')
uv run --locked python data/build_datasets.py
uv run --locked python data/build_full_sets.py

# Gemma 3 4B: llama.cpp 서버, NN-PPI 부록 C 설정 (T=1.0, top-k=64, top-p=0.95)
llama-server -m google_gemma-3-4b-it-Q4_K_M.gguf --port 8080 -c 8192 -ngl 99
uv run --locked python scripts/score_gemma.py --dataset data/processed/clef_test.csv --out results/llm_scores/gemma/clef_test_scores.jsonl

# Claude Sonnet 5 / Haiku 4.5: Claude Code CLI로 40문장씩 배치 채점
uv run --locked python scripts/build_frontier_prompts.py
uv run --locked python scripts/build_frontier_prompts.py full_set.json
cd results/llm_scores && ls sonnet5/*.txt | xargs -P 6 -n 1 bash ../../scripts/run_frontier_batch.sh
ls sonnet5/*.txt | xargs -P 6 -n 1 bash ../../scripts/run_haiku_batch.sh
```

## 데이터

| 데이터셋 | 출처 | 라이선스 | 사용 분할 |
|---|---|---|---|
| CLEF 2024 CheckThat! Task 1 (영어) | [HF `iai-group/clef2024_checkthat_task1_en`](https://huggingface.co/datasets/iai-group/clef2024_checkthat_task1_en), 공식 test 정답은 [CheckThat! GitLab](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab) | CC BY-SA 4.0 | 보정: train에서 클래스 균형 2,406문장(Gemma 파싱 성공 2,405) / 테스트: dev 1,032 + dev-test 318 + 공식 test 341 = 1,691문장 (NN-PPI 재현 수치는 원 논문과 같은 dev-test 분할) |
| ClaimBuster | [Zenodo 3836810](https://zenodo.org/records/3836810) | CC BY 4.0 | 보정: 2012년 토론 1,314문장 / 테스트: 2016년 토론 2,745문장 (NN-PPI 원 논문은 2,740문장) |

`data/processed/`의 CSV는 위 원본에서 `data/build_datasets.py`, `data/build_full_sets.py`로 만든 파생물이며 원본 라이선스를 따른다.

CLEF 2024 영어 데이터는 ClaimBuster 말뭉치에서 가져왔다(Hasanain et al., CEUR-WS Vol. 3740, pp. 276–286). 실제로 CLEF 보정 문장 2,406개 중 2,405개가 ClaimBuster에 있다. 즉 두 데이터셋의 출처는 독립적이지 않다. ClaimBuster 테스트 2,745문장 중 313문장은 CLEF 보정 세트에도 있지만, 각 데이터셋은 자기 보정 세트만으로 학습·보정하므로 평가 누수는 없다. CLEF 안에서는 dev의 1문장이 보정 세트와 텍스트가 같다(1,691문장 중 1문장).

## 유의 사항

- **LLM 모델 ID**: 모든 Sonnet 채점(2026-09-25~26, 09-28)은 Claude Code 세션 기록상 `claude-sonnet-5`였다. 현재 CLI의 `--model sonnet` 별칭은 더 새로운 모델(`claude-sonnet-5-5`)을 가리키므로, 채점 스크립트는 모델 ID를 고정해 두었다.
- **Haiku 채점 조건**: Haiku 4.5는 `--restricted --strict-mcp-config` 모드로 채점했고(9월 Sonnet 채점과 CLI 모드가 다름), Claude Code 기본 설정에서 추론(thinking)을 길게 써서 배치당 비용·시간은 오히려 Sonnet 5보다 컸다. 그래서 비용 수치는 Sonnet 5로만 보고한다. 빈 응답을 낸 2배치는 다시 채점했다(`results/llm_scores/haiku45/failed/`).
- **비용·시간 추정 방식**: 테스트 배치 23개(5번째마다 하나)를 `claude-sonnet-5`에 다시 보내 응답 시간과 토큰을 쟀다. CLI가 덧붙이는 자체 시스템 프롬프트(19,385토큰)는 빈 프롬프트로 재서 뺐고, 1턴으로 끝난 호출만 썼다. 배치당 입력 약 1,540, 출력 약 370토큰, 평균 응답 4.3초. 정가(입력 \$2, 출력 \$10 / 100만 토큰)를 곱했고, 병렬 호출·캐싱·배치 할인은 반영하지 않았다.
- **배치 채점**: LLM은 40문장씩 채점했다. CLEF 80문장을 단건으로도 채점했는데 11건은 CLI 사용량 한도로 응답이 없었고, 나머지 69문장에서 ROC-AUC는 단건 0.991, 배치 0.990이었다. CLEF 테스트를 두 번 배치 채점한 점수의 상관계수는 0.973이다.
- **호출률 50%**: 보정 세트 교차검증 정확도가 50% 이후 CLEF 0.06%p, ClaimBuster 0.11%p만 올라 50%로 정했다. "0.1%p 이하"를 엄격한 규칙으로 쓰면 ClaimBuster는 60%가 되므로, 논문에는 규칙이 아니라 실제 증가값을 적었다.
- **통계**: 회차별 McNemar 검정은 다중 비교 보정을 하지 않았다(논문에 명시). 핵심 주장은 신뢰구간 기반이라 보정과 무관하다.
- **이전 표본**: 초기 실험은 비용 때문에 ClaimBuster 2016년 테스트 중 800문장과 CLEF dev-test 318문장만 썼다. 논문은 나머지를 추가로 채점한 전체 테스트셋 결과(`*_full`)를 보고한다.
- **Few-shot 예시**: NN-PPI는 few-shot 예시 6개를 공개하지 않았다. `cwcascade/prompt.py`의 예시는 원 논문의 6단계 기준에 맞춰 직접 작성했다.
- **선행 연구와의 관계**: 넘긴 입력에서 두 모델의 출력을 결합하는 캐스케이드는 일반 분류 과제에서 CAUC([arXiv:2609.11446](https://arxiv.org/abs/2609.11446))가 먼저 제안했다. 본 연구는 이를 팩트체크 필요성 탐지에 적용하면서 결합 가중치를 라벨로 학습한다. SRR([arXiv:2609.07786](https://arxiv.org/abs/2609.07786))은 넘길 입력을 학습으로 고르지만 넘긴 입력은 큰 모델의 답으로 교체한다.

## 다른 브랜치

`release`에는 논문에 쓰인 코드와 결과만 정리했다. 연구 과정의 기록은 다음 브랜치에 남아 있다(정리 전 폴더 구조).

| 브랜치 | 내용 |
|---|---|
| `jev-paper` | 상용 판정 모델 JEV를 첫 단계로 쓴 별도 2쪽 원고 (JEV → Sonnet 결합). 투고 여부 미정 |
| `ct22-tweets` | 세 번째 데이터셋 시험: CheckThat! 2022 Task 1A 영어 트윗. 미리 정한 기준 중 결합형 > NN-PPI 적용 전량 호출이 성립하지 않아 논문에 넣지 않음 |
| `full-benchmark` | 전체 테스트셋 확장과 원고 개정 이력 (병합됨) |
| `fusion-cascade` | 결합형 캐스케이드 실험의 원본 커밋 이력 |
| `nnppi-reproduction` | NN-PPI 재현과 신뢰구간 커버리지 보정 연구 |
| `prompt-reorder-ablation` | 프롬프트·채점 방식 변형 실험과 임베딩 분류기 발견 과정 |
| `safety-flag-extension` | 다른 도메인(Safety-Flag)으로의 전이 실험 |
| `topic-venue-research` | 주제 선정과 학회 적합성 검토 기록 |

## 참고 문헌 (원고와 같음)

1. P. Amatya, Venktesh V, and V. Setty, "Calibrating Small Language Models for Claim Check-Worthiness Detection," arXiv:2608.30731, 2026.
2. P. Amatya and V. Setty, "Multilingual Fact-Checking at Scale: Fine-Tuned Compact Models vs LLMs," arXiv:2606.08605, 2026.
3. M. Hasanain et al., "Overview of the CLEF-2024 CheckThat! Lab Task 1 on Check-Worthiness Estimation of Multigenre Content," CEUR-WS vol. 3740, pp. 276–286, 2024.
4. F. Arslan, N. Hassan, C. Li, and M. Tremayne, "A Benchmark Dataset of Check-Worthy Factual Claims," in Proc. ICWSM, vol. 14, pp. 821–829, 2020.
5. Z. Wang et al., "Signed Rescue Routing: Harm-Aware Cascades for Efficient LLM Inference," arXiv:2609.07786, 2026.
6. Y. Zhang et al., "Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous Model Collaboration," arXiv:2609.11446, 2026.
