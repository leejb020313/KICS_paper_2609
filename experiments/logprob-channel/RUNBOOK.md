# 출력 채널 양자화 제거 (logprob 기반 연속 점수) — 실행 가이드

## 왜 이걸 하는가 — 측정된 근거

`../prompt-reorder-ablation/channel_capacity.py`로 원점수 채널의 정보량을 직접 쟀습니다.

**CLEF, Gemma 3 4B 원본 프롬프트:**
```
서로 다른 점수값        : 13개
임계값 0.5 정확도       : 0.780
오라클 상한(최적 매핑)  : 0.780   ← 완전히 동일
상호정보량              : 0.258 / 0.924 bits (27.9%)
```

**오라클 상한이 실제 정확도와 같습니다.** 이 원점수만 가지고는 어떤 사후보정
기법(temperature/Platt/isotonic/conformal/shrinkage/NN-PPI/우리 게이트+국소회귀)도
78%를 넘을 수 없습니다. 수학적 상한입니다.

**ClaimBuster에서는 더 명확합니다:**
```
score=0.70  n=372 (13.6%)  P(gold=1)=0.500   ← 완벽한 동전던지기
score=0.60  n=303 (11.1%)  P(gold=1)=0.366
```
372개 claim이 전부 `0.70`이라는 **동일 심볼**로 뭉개졌고 그 안에서 정답이 정확히 반반입니다.
입력이 문자 그대로 같은 값이므로 kNN도 conformal도 이 372개를 구별할 수 없습니다.

**같은 50개 claim, 모델만 바꿔서:**

| | Gemma 3 4B | Claude Sonnet 5 |
|---|---|---|
| 서로 다른 점수값 | 11개 | 18개 |
| 최대 버킷 집중도 | 30% | 12% |
| 상호정보량 | 0.484 bits (48%) | **0.945 bits (94%)** |
| 오라클 상한 | 0.840 | **0.980** |

프론티어 모델이 앞서는 건 판단력 자체보다 **출력 채널의 해상도**입니다.

## 가설

양자화는 **토큰이 샘플링되는 순간** 일어납니다. 모델 내부 상태는 연속입니다 —
`0.70`을 뱉을 때 그 자리의 토큰 분포에는 6, 8에도 확률 질량이 있습니다.
**logprobs를 읽으면 샘플러가 버린 연속값을 복원할 수 있고, 추가 연산은 0입니다.**

## 세 가지 모드

| 모드 | 방식 | 특징 |
|---|---|---|
| `yesno` | "Yes/No로 답하라" → `P(Yes)/(P(Yes)+P(No))` | 가장 깨끗함. 숫자·JSON·반올림 전부 우회 |
| `digit` | "0~9 한 자리로 답하라" → `E[d]/9` | 원 과제의 10점 척도 유지하면서 연속값 |
| `verbal` | 원 논문 프롬프트 + `Score: 0.` 로 프라이밍 → 소수 첫자리 기대값 | 원 논문에 가장 가까움 |

## 1. 스모크 테스트 먼저 (필수)

llama-server가 `top_logprobs`를 실제로 반환하는지 먼저 확인하세요. 안 되면 전체 실행이 통째로 낭비됩니다.

```
cd experiments/logprob-channel
pip install pandas requests
python smoke_test.py
```

**확인할 것:**
- `OK: logprobs are live` 가 떠야 함. `FAIL: ... returned NO logprobs` 면 llama.cpp 빌드가 지원 안 하는 것 → 최신 빌드로 교체하거나 네이티브 `/completion` + `n_probs`로 전환 필요(알려주시면 수정본 드립니다)
- 점수가 **연속값**이어야 함 (예: `0.6173`, `0.4482`). `0.60`/`0.70`처럼 둥근 값만 나오면 뭔가 잘못된 것
- `mass_on_poles` / `mass_on_digits` 가 1.0에 가까워야 함. 0.1 같은 값이면 모델이 다른 형식으로 답하고 있는 것

## 2. CLEF test부터 (318개, 제일 빠름)

```
python score_with_logprobs.py --mode yesno --dataset ../../data/processed/clef_test.csv --out results/clef_test_yesno.jsonl
python score_with_logprobs.py --mode digit --dataset ../../data/processed/clef_test.csv --out results/clef_test_digit.jsonl
```

`--max-tokens 4`라 기존 채점보다 **훨씬 빠릅니다** (기존은 justification 100단어까지 생성했음).

## 3. 채널 정보량 비교

```
python analyze_channel.py --dataset clef
```

**이 출력이 이번 실험의 판정입니다.** `ORACLE CEILING`과 `mutual info`가
verbalized(0.780 / 0.258 bits) 대비 올라갔는지만 보면 됩니다.
- 올라갔다 → 병목이 정말 출력 양자화였고, 그 위에 NN-PPI/우리 방법을 다시 얹으면 천장이 함께 올라감
- 그대로다 → 양자화가 아니라 모델의 판단 자체가 한계. 그것도 중요한 결론

## 4. 여기서 좋으면 calib까지 확장 (NN-PPI/우리 방법 재적용용)

```
python score_with_logprobs.py --mode yesno --dataset ../../data/processed/clef_calib.csv --out results/clef_calib_yesno.jsonl
python score_with_logprobs.py --mode yesno --dataset ../../data/processed/claimbuster_test.csv --out results/claimbuster_test_yesno.jsonl
python score_with_logprobs.py --mode yesno --dataset ../../data/processed/claimbuster_calib.csv --out results/claimbuster_calib_yesno.jsonl
python analyze_channel.py --dataset claimbuster
```

전부 재시작 가능(resumable)합니다.

## 5. 결과 커밋

```
git add experiments/logprob-channel/results/
git commit -m "Add logprob channel results"
git push
```

## 기대치

- **잘 되면**: 상한 0.780 → 0.85~0.9대. 그러면 지금까지 만든 모든 보정 기법이
  새 천장 아래에서 다시 작동할 공간이 생깁니다. 논문의 헤드라인이 "보정 기법 개선"에서
  **"소형 모델의 진짜 병목은 출력 양자화이며, 이를 제거하면 프론티어와의 격차 상당 부분이 사라진다"**로 바뀝니다.
- **안 되면**: 그것도 결정적 결과입니다. "verbalized confidence의 양자화는 증상이지 원인이 아니다"를
  실측으로 보인 것이고, 원 논문이 남긴 한계에 대한 정확한 반증이 됩니다.
