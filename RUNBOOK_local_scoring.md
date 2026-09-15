# 로컬(RTX 3060) 채점 실행 안내

이 저장소는 데이터 준비 + 분석 코드까지 클라우드 세션에서 완성했지만,
Qwen3-4B few-shot 채점만은 GPU가 있는 로컬 머신에서 실행해야 합니다.

## 1. llama.cpp 서버 실행 (SETUP.md 기준)

```
llama-server -m Qwen3-4B-Q4_K_M.gguf --port 8080 -c 8192 -ngl 99
```

## 2. 이 저장소를 로컬에 pull/clone

## 3. 필요 패키지

```
pip install pandas requests
```

## 4. 4개 데이터셋 각각 채점 실행 (재시작 가능, 중단해도 이어서 진행됨)

```
python src/nnppi/score_with_llm.py --dataset data/processed/clef_calib.csv --out results/clef_calib_scores.jsonl
python src/nnppi/score_with_llm.py --dataset data/processed/clef_test.csv --out results/clef_test_scores.jsonl
python src/nnppi/score_with_llm.py --dataset data/processed/claimbuster_calib.csv --out results/claimbuster_calib_scores.jsonl
python src/nnppi/score_with_llm.py --dataset data/processed/claimbuster_test.csv --out results/claimbuster_test_scores.jsonl
```

ClaimBuster calib(1,314)+test(2,745) 합쳐서 4,059건, CLEF calib(2,406)+test(318) 합쳐서 2,724건 —
총 6,783건. Qwen3-4B 로컬 추론 기준으로 대략적인 소요 시간을 먼저 재보고(예: 20건),
전체 예상 시간을 가늠해서 필요하면 calib 세트를 줄이는 것도 고려하세요(D-17 안에 끝나야 함).

## 5. 결과를 다시 이 저장소로 가져오기

`results/*.jsonl` 4개 파일을 커밋하거나 다시 세션에 붙여넣어 주시면,
그 다음부터는(임베딩, NN-PPI, 대조군 보정기법 비교, 우리 개입 모듈) 전부 CPU로
클라우드 세션에서 이어서 진행할 수 있습니다.

## 참고: 프롬프트의 few-shot 예시 6개는 우리가 새로 만든 것입니다

원 논문(2608.30731)은 `{{examples}}` 자리만 언급하고 실제 6개 문구를 공개하지 않았습니다.
`src/nnppi/prompt.py`의 `FEW_SHOT_EXAMPLES`가 논문의 6단계 기준에 맞춰 우리가 만든 예시입니다 —
이건 원 논문과의 불가피한 차이점이니 논문 작성 시 명시해야 합니다.
