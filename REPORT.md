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

## 5. 한계

- Few-shot 예시 6개는 원 논문 비공개로 자체 제작 (재현 시 불가피한 차이)
- 단일 SLM(Gemma 3 4B)만 검증
- 이산화 현상의 근본 원인은 미분석 (향후 과제)
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

- 코드: 이 저장소 (`nnppi-reproduction` 브랜치)
- 논문 초안: `artifacts/nnppi-paper/nnppi_kics_draft.docx` (OpenResearch 프로젝트 아티팩트, 저자정보·표 삽입만 남음)
