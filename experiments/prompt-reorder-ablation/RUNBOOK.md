# 프롬프트 필드 순서 뒤집기 ablation — 실행 가이드

## 가설

원 논문 프롬프트(`../../src/nnppi/prompt.py`)는 JSON 출력 스키마가
`confidence_score`를 `justification`보다 먼저 요구합니다. `results/clef_test_scores.jsonl`의
실제 `raw_response`를 확인해보면 모델이 정말로 항상 그 순서대로(점수 먼저, 근거 나중)
생성합니다 — 자기회귀 생성이라 점수가 근거를 반영할 수 없는 구조입니다.

`prompt_reordered.py`는 딱 하나만 바꿨습니다: `justification`을 먼저 쓰고 그다음
`confidence_score`를 쓰도록 스키마와 6개 few-shot 예시 순서를 뒤집었습니다. 기준/예시
내용 자체는 원본과 완전히 동일합니다. 이게 되면(모델이 자기가 쓴 근거에 실제로 점수를
근거해서 매기게 되면) 최소 정확도 오답률이 줄어들 것으로 예상합니다.

## 1. llama-server는 이미 떠 있는 것 재사용

`../../RUNBOOK_local_scoring.md`와 동일한 서버 그대로 씁니다 (Gemma 3 4B 기준으로
이미 재현 실험에서 쓰던 서버). 새로 켤 필요 없습니다.

## 2. 이 폴더에서 실행 (최소: CLEF test만 먼저 — 가장 저렴하게 가설 검증)

```
cd experiments/prompt-reorder-ablation
pip install pandas requests   # 이미 설치돼 있으면 생략
python score_with_llm_reordered.py --dataset ../../data/processed/clef_test.csv --out results/clef_test_scores_reordered.jsonl
```

CLEF test는 318개뿐이라 제일 빠르게 결과를 볼 수 있습니다. 여기서 먼저 오답률이
줄어드는지 확인하고, 될 것 같으면 아래 3번으로 범위를 넓히세요.

## 3. (선택, 여유 되면) calib까지 재채점 — NN-PPI/우리 방법까지 통으로 비교하려면 필요

```
python score_with_llm_reordered.py --dataset ../../data/processed/clef_calib.csv --out results/clef_calib_scores_reordered.jsonl
python score_with_llm_reordered.py --dataset ../../data/processed/claimbuster_test.csv --out results/claimbuster_test_scores_reordered.jsonl
python score_with_llm_reordered.py --dataset ../../data/processed/claimbuster_calib.csv --out results/claimbuster_calib_scores_reordered.jsonl
```

전부 재시작 가능(resumable)합니다 — 중단해도 이어서 진행됩니다.

## 4. 비교 실행

```
cd experiments/prompt-reorder-ablation
python compare_ablation.py --dataset clef
python compare_ablation.py --dataset claimbuster
```

- `results/*_test_scores_reordered.jsonl`만 있으면: 원본 vs 재정렬 프롬프트의
  **원점수 오답률/F1**만 비교합니다 (가설 자체를 가장 빠르게 검증).
- `results/*_calib_scores_reordered.jsonl`까지 있으면: 그 위에 원 논문 NN-PPI까지
  자동으로 적용해서 `../../REPORT.md`에 있는 숫자와 바로 비교 가능한 표까지 출력합니다.

## 5. 결과를 다시 저장소에 커밋

```
git add experiments/prompt-reorder-ablation/results/
git commit -m "Add reordered-prompt scoring results"
git push
```

## 기대하는 것 / 기대하지 않는 것

- **기대**: 원점수 단계의 오답률이 원본 대비 어느 정도 줄어드는 것 (지난 세션에서
  Gemma의 justification이 정답 신호를 맞게 짚고도 점수가 그걸 안 따라간 사례 6개를
  확인했고, 그게 필드 순서 때문이라는 가설).
- **기대하지 않는 것**: 이 하나로 프론티어 모델(Claude/GPT) 수준까지 완전히 따라잡는 것.
  같은 저자의 선행 연구(TicToc 논문)에서 유사한 개입이 간극의 72%만 회수했던 것처럼,
  부분적 개선일 가능성이 높습니다.
