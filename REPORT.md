# NN-PPI 재현 연구 — 최종 결과 보고서

**대상 학회**: KICS 2026 추계종합학술발표회 (A4 1~2쪽) · **마감**: 2026-10-02
**재현 대상**: Amatya et al., "Calibrating Small Language Models for Claim Check-Worthiness Detection" (NN-PPI, arXiv:2608.30731)

## 1. 문제 정의

NN-PPI는 소형 언어모델(SLM)의 판단을 이웃 잔차 평균으로 보정해 대형 LLM 수준까지 끌어올리는 재학습 없는 기법이다. 원 논문을 정밀 분석한 결과 두 가지 결함을 발견했다:

1. **신뢰구간(CI) 붕괴**: 명목 95% 대비 실측 커버리지 46.8~65%
2. **범위 이탈**: 잔차 보정식 θᵢ=ĉᵢ+r̄ᵢ가 구조상 [0,1]을 벗어날 수 있음에도 제약 없음

원 저자도 표준 보정기법(temperature/Platt scaling, isotonic regression)과의 비교를 향후 과제로 남겼다.

## 2. 재현 설정

| 항목 | 내용 |
|---|---|
| 데이터셋 | ClaimBuster (Zenodo 공개본, 2012 calib 1,314 / 2016 test 2,745), CLEF 2024 CheckThat! Task1 (공식 분할, calib 2,406 / test 318) |
| 모델 | Gemma 3 4B (원 논문과 동일), llama.cpp, few-shot 프롬프트 |
| 임베딩 | BAAI/bge-small-en-v1.5 (kNN 검색용) |
| 파싱 성공률 | 100% (Qwen3.5로 시도했다가 88.7%였던 것을 Gemma 3로 교체 후 해결) |

## 3. 제안 기법

기존 모델(J, frozen) + 학습이 필요 없는 두 가지 닫힌형(closed-form) 개입:

- **범위 제약(clip)**: θᵢ를 [0,1]로 제한. 하이퍼파라미터 없음, 구간 폭에 영향 없음.
- **유사도 가중치**: 이웃 잔차의 균등 평균 대신, 코사인 유사도를 softmax(온도 τ)로 선명화한 가중 평균 사용. τ는 보정 세트의 20% 홀드아웃에서 **커버리지 기준으로만** 선택(테스트셋 미사용).

$$\theta_i = \mathrm{clip}\Big(\hat c_i + \sum_j w_{ij}(Y_j-\hat c_j),\ 0,\ 1\Big),\quad w_{ij}=\mathrm{softmax}(\mathrm{sim}_{ij}/\tau)$$

## 4. 핵심 결과

### 4.1 헤드라인 — 범위 제약 + 유사도 가중치 (k=3)

| | 커버리지 | 폭 변화 | 가중 F1 |
|---|---|---|---|
| CLEF | 0.588 → **0.720** (+13.2%p) | +0.7% | 0.815 → **0.836** |
| ClaimBuster | 0.565 → **0.648** (+8.3%p) | +0.9% | 0.765 → **0.772** |

정확도·구간폭·커버리지 세 축 모두 개선. McNemar 검정으로 정확도 손실이 없음을 확인.

### 4.2 범위 제약 단독 효과 (하이퍼파라미터 0개)

두 데이터셋·k∈{3,5,10} 전 조건(6/6)에서 구간 폭 변화 없이 커버리지 **+4.4~9.1%p** 개선. 결정론적 연산이라 우연의 여지가 수학적으로 없음.

### 4.3 τ 선택의 견고성 검증

커버리지는 τ에 대해 **단조적으로 반응**(τ가 작을수록 커버리지 상승)하여, calib 홀드아웃에서의 선택이 test로 안정적으로 전이된다(6/6 조건 모두 개선). 반대로 ECE를 목적함수로 같은 절차를 적용하면 전이되지 않음(τ에 따라 비체계적으로 변동) — 초기에 관측한 "ECE 47% 개선"은 τ=0.1 임의 고정에서 나온 착시였음을 확인하고 폐기했다.

### 4.4 부정적 결과 — 구간폭 확장 계열의 구조적 한계

Split-conformal calibration 및 5가지 변형(ESS 분산 위 conformal, k 재선택, 분산/유사도 기준 Mondrian, 선택적 기권)을 시도했으나, 모두 구간 폭이 baseline 대비 **약 3배**로 수렴했다. 평균 폭(1.47~2.18)이 실측 데이터 범위(raw score 관측 범위 0.89)를 초과하여 개별 구간이 실질적 정보를 상실한다. 이는 **확률 파라미터의 CI를 이진 라벨 포함 여부로 평가**하는 지표 정의 자체가 요구하는 구조적 한계로 해석된다.

### 4.5 부가 발견 — 표준 보정기법의 무력화

LLM 원점수가 13~14개의 이산값으로 군집화되어(few-shot 예시 앵커링으로 추정), temperature/Platt scaling 같은 단조 변환 기반 보정기법은 **어떠한 분류 결정도 바꾸지 못함**(예측 변화 0건, 실측 확인). Brier score 기준 NN-PPI 계열이 표준 기법 대비 20~32% 우수.

## 4.6 채널 용량 진단 — 왜 사후보정에 천장이 있는가

`experiments/prompt-reorder-ablation/channel_capacity.py`로 원점수 채널이 실어 나르는 정보량을 직접 측정했다. **CLEF에서 오라클 상한(원점수의 13개 값을 라벨에 최적으로 매핑했을 때 달성 가능한 최댓값)이 실제 임계값 0.5 정확도(0.780)와 완전히 동일했다** — 상호정보량 0.258/0.924 bits(27.9%). 즉 이 원점수만으로는 **어떤 사후보정 기법도 78%를 넘을 수 없다는 수학적 상한**이 존재하며, §4.4의 conformal류 실패와 표준 보정기법 무력화(§4.5)가 전부 이 천장 아래에서 일어난 현상이었다. ClaimBuster는 더 극단적으로, claim 372개(13.6%)가 전부 `score=0.70`이라는 동일 심볼로 뭉개지고 그 안에서 `P(gold=1)=0.500`(완전한 동전던지기)이다 — 입력이 문자 그대로 같으므로 어떤 후처리로도 구별 불가능하다.

같은 50개 CLEF claim을 Gemma 3 4B와 Claude Sonnet 5(프론티어 대리)에 동일 프롬프트로 채점시켜 비교하면, Gemma는 라벨 엔트로피의 48%만, Sonnet은 94%를 점수에 실어 보낸다(오라클 상한 0.840 대 0.980). **프론티어와의 격차는 상당 부분 판단력이 아니라 출력 채널의 해상도 문제다.**

## 4.7 잔차식 자체의 개선 — 게이트+국소회귀 (오답률을 실제로 낮춘 유일한 사후보정)

원 잔차식(Eq.1, 균등평균+무제약 가산)을 두 가지로 대체했다(leakage-free, calib를 pool/tune으로 분리해 임계값 선정 후 test 평가):

- **신뢰도 게이트**: 이웃 라벨 순도(purity)·최대 유사도가 낮으면 보정 강도 λ∈[0,1]를 수축
- **국소 선형회귀(ridge=2.0)**: 이웃 15개로 $Y \approx a + b\hat c$ 국소 보정선을 학습(상수 가산이 아닌 구간별 보정), ridge로 과적합 방지

3시드 평균 결과: CLEF 오답률 19.2%→17.1%(-10.9%), 필요 폭 -6.4%; ClaimBuster 24.3%→22.4%(-7.7%), 필요 폭 -11.2%. **통계적으로 유의(McNemar CLEF p=0.0145, ClaimBuster p<0.0001)하지만 효과 크기는 작다**(부트스트랩 95% CI: CLEF [+1.3, +10.7]pp, ClaimBuster [+1.9, +4.0]pp).

## 4.8 프롬프트 필드 순서 — 원저자의 선행 연구(knowing-doing gap)와 동일 구조

원 프롬프트(Fig.2)는 JSON에서 `confidence_score`를 `justification`보다 먼저 요구한다. 자기회귀 생성상 점수가 근거보다 먼저 확정되므로, 근거가 점수를 교정할 기회가 구조적으로 없다 — 저자의 선행 연구("TicToc" KICS 제출본)가 보고한 knowing-doing gap(모델이 스스로 판단해놓고 그 판단대로 행동하지 않는 분리, 0.578→0.806 판단 상한 대비 실제 행동 일치율 52.2%)과 동일 구조다. 필드 순서를 `justification`→`confidence_score`로 뒤집어 재실험한 결과는 **데이터셋 의존적이었다**: CLEF는 가설대로 개선(22.0%→19.5% 오답, NN-PPI 결합 시 17.0%)됐으나, ClaimBuster는 악화(25.6%→29.4%)됐다. 원인 분석 결과 ClaimBuster(2016 대선토론 발화, 수사적 표현 밀도 높음)에서는 근거를 먼저 쓰게 하면 "~일 수도 있다"는 가정법으로 스스로를 설득해 체크가치를 과대평가하는 방향으로 체계적 편향(broke 사례의 94.7%가 상향 이동)이 생겼다. 원본·재정렬 점수의 불일치(|diff|) 자체를 신호로 쓰는 로지스틱 앙상블은 **두 데이터셋 모두에서 원본 단일 프롬프트를 이겼다**(22.0%→21.1%, 25.5%→24.5%; McNemar CLEF p=0.0088, ClaimBuster p<0.0001).

## 4.9 출력 채널 양자화 해제 (logprobs)

§4.6의 진단에 따라 verbalized score 대신 토큰 logprobs에서 직접 연속값을 복원(yesno: P(Yes)/(P(Yes)+P(No)), digit: 0-9 분포의 기대값)했다. 오라클 상한은 두 데이터셋 모두 상승했으나(CLEF 0.780→0.818, ClaimBuster 0.775→0.784) 효과는 작고 데이터셋 의존적이며, yesno는 확률이 두 극단(~1e-5, ~1-1e-7)에 포화돼 순진한 임계값 적합이 오히려 원점수보다 나쁜 정확도(0.748)를 낼 수 있음을 확인했다(logit 스케일로 읽어야 함).

## 4.10 헤드라인 — calib set을 직접 지도학습 신호로 사용 (가장 큰 개선)

문헌 검토 결과("100 Labelled Samples to Achieve Break-Even Performance", arXiv:2402.12819; "Fine-Tuned Small LLMs (Still) Significantly Outperform Zero-Shot GenAI Models", arXiv:2406.08660 — 후자는 ChatGPT·Claude Opus를 직접 비교 대상으로 포함), 라벨 100개 안팎이면 학습된 소형 분류기가 훨씬 큰 모델의 제로샷/퓨샷을 능가한다는 결과가 여러 독립 연구에서 재현됐다. 본 연구의 calib set(CLEF 2,406 / ClaimBuster 1,308)은 그 기준의 13~24배임에도, §3의 kNN 이웃 검색 용도로만 사용됐다.

이미 계산된 문장 임베딩(all-MiniLM-L6-v2) 위에 calib set으로 SVM(rbf)을 직접 학습(GPU·추가 LLM 호출 불필요, CPU 수 초)시켜 완전히 분리된 test set에서 평가한 결과:

| | LLM 원점수 | 임베딩 SVM | 상대 개선 |
|---|---|---|---|
| CLEF | 22.0% 오답 | 14.2~14.8% 오답 (AUC 0.928) | **-32~-35%** |
| ClaimBuster | 25.5% 오답 | 20.2% 오답 (AUC 0.830) | **-21%** |

SVM 확률 + 원본/재정렬 LLM 점수를 로지스틱 메타러너로 스택하면 CLEF 12.9%, ClaimBuster 19.9%까지 추가 개선. **검증**: calib를 8회 부트스트랩 재추출해도 오답률이 좁은 범위(CLEF 14.8~16.7%, ClaimBuster 21.5~23.3%, 표준편차 0.6~0.8pp)에서 안정적이며, McNemar 검정은 두 데이터셋 모두 강한 유의성을 보인다(CLEF p=0.0088, ClaimBuster p=5.4×10⁻⁸). 다수결 기준선(34.0%/73.5% 오답)보다 압도적으로 낫다.

**§4.7의 게이트+국소회귀·NN-PPI·logprob 채널을 이 SVM 확률에 추가로 스택해도 통계적으로 유의한 추가 개선은 없었다**(McNemar SVM-단독 대 전체결합: CLEF p=0.180, ClaimBuster p=0.460; 메타러너 계수는 SVM이 8~9인 반면 나머지 신호는 모두 3 미만). 이는 임베딩 분류기가 §3의 이웃 잔차보정 계열이 포착하던 정보를 이미 상위 집합으로 포함함을 시사한다.

**한계**: 이 방법은 더 이상 NN-PPI의 "재학습 없는 냉동결 LLM 후보정"이라는 전제 위에 있지 않다 — 별도의 지도학습 단계가 필요하다(다만 GPU·라벨링 비용 추가 없이 이미 보유한 calib 라벨로 수 초 내 완료됨). calib 크기에 비례해 개선폭이 커지는 경향(CLEF 2,406개 > ClaimBuster 1,308개)이 관측됐다.

### 4.10.1 프론티어(Sonnet) 대비 — 격차의 몇 %를 좁혔나

같은 50개 CLEF claim에 원 논문 프롬프트를 그대로 적용해 Claude Sonnet 5(프론티어 대리, GPT-5.2/Claude Opus 4.6 자체는 아님)로 채점한 소규모(n=50) 비교:

| | 오답 | F1 |
|---|---|---|
| Gemma 3 4B 원점수 | 10/50 (20.0%) | 0.800 |
| Gemma + 임베딩 SVM | 8/50 (16.0%) | 0.840 |
| Claude Sonnet 5 | 4/50 (8.0%) | 0.919 |

오답 개수 기준 Gemma-Sonnet 격차(6개)의 **33%**를 SVM이 회수했다(6→4). 전체 test set(318개, §4.10 본문)에서는 상대개선폭이 더 크지만(22.0%→12.9~14.8%) Sonnet 점수를 그 규모로는 확보하지 못해 직접 비교는 n=50 표본에 한정된다.

## 5. 한계

- Few-shot 예시 6개는 원 논문 비공개로 자체 제작 (재현 시 불가피한 차이)
- 단일 SLM(Gemma 3 4B)만 검증
- 프론티어 비교(§4.10.1)는 GPT-5.2/Claude Opus 4.6이 아닌 Claude Sonnet 5 대리 채점, n=50 소표본
- §4.10 임베딩 분류기는 원 논문의 "재학습 없는 냉동결 LLM" 전제를 벗어난 별도 방법론
- ClaimBuster는 연도 기반 분할을 재현했으나 원 논문과 소수 건수 차이 존재(전처리 방식 차이로 추정)

## 6. 재현 방법

```
git clone -b nnppi-reproduction git@github.com:leejb020313/KICS_paper_2609.git
python data/build_datasets.py            # ClaimBuster/CLEF 데이터 재구성
# GPU 머신에서: llama-server로 Gemma 3 4B 서빙 후
python src/nnppi/score_with_llm.py --dataset data/processed/clef_calib.csv --out results/clef_calib_scores.jsonl
# (4개 파일 반복)
python src/nnppi/run_experiment.py --dataset clef --embed-model BAAI/bge-small-en-v1.5
python src/nnppi/run_experiment.py --dataset claimbuster --embed-model BAAI/bge-small-en-v1.5
```

## 7. 산출물

- 코드(핵심 재현): `nnppi-reproduction` 브랜치
- 코드(§4.6~4.10 후속 실험): `prompt-reorder-ablation` 브랜치
  - `experiments/prompt-reorder-ablation/` — 채널 용량 진단, 게이트+국소회귀, 프롬프트 필드순서 앙상블
  - `experiments/logprob-channel/` — logprob 기반 채널 양자화 해제
  - `experiments/embedding-classifier/` — §4.10 헤드라인(임베딩 직접학습 분류기), kitchen-sink 결합 검증
- 논문 초안: `artifacts/nnppi-paper/nnppi_kics_draft.docx` (OpenResearch 프로젝트 아티팩트, 저자정보·표 삽입만 남음)
