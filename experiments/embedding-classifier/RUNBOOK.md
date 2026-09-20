# 임베딩 직접 학습 분류기 — calib set을 kNN 검색이 아니라 학습 신호로 쓰기

## 배경 — 왜 이걸 안 하고 있었나

오늘 하루 종일 NN-PPI(원 논문)와 그 변형들(게이트, 국소회귀, 프롬프트 재정렬, logprob
채널)을 붙잡고 있었습니다. 이것들은 전부 **LLM이 뱉은 verbalized score를 기준점으로 두고
사후 보정**하는 방식입니다. 그런데 `../prompt-reorder-ablation/channel_capacity.py`로
측정해보니 그 원점수 채널 자체가 정보를 27~48%밖에 안 실어 나른다는 게 확인됐습니다
(자세한 내용은 `../prompt-reorder-ablation/RUNBOOK.md`). 사후보정은 원점수라는 천장 아래에
갇혀 있었던 겁니다.

문헌 검색(`orx discover`)으로 확인한 결과, 이 문제를 완전히 다른 축에서 푸는 방법이
이미 여러 논문에서 독립적으로 검증돼 있었습니다:

- **"Comparing Specialised Small and General Large Language Models on Text
  Classification: 100 Labelled Samples to Achieve Break-Even Performance"**
  (arXiv:2402.12819) — 라벨 100개만 있어도 특화된 작은 분류기가 일반 대형 LLM의
  제로샷/퓨샷을 따라잡거나 이깁니다.
- **"Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot
  Generative AI Models in Text Classification"** (arXiv:2406.08660) — ChatGPT
  (GPT-3.5/GPT-4)와 **Claude Opus를 직접** 파인튜닝된 소형 모델과 비교, 파인튜닝이 항상 이김.
- **"Language Models for Text Classification: Is In-Context Learning Enough?"**
  (arXiv:2403.17661) — 같은 결론.

우리 calib set은 CLEF 2,406개, ClaimBuster 1,308개 — 저 "100개" 기준점의 13~24배입니다.
그런데 지금까지 이 라벨들을 **NN-PPI의 kNN 이웃 검색용으로만** 썼지, 분류기를 직접
학습시키는 데는 한 번도 안 썼습니다. 원 논문(NN-PPI, arXiv:2608.30731) 스스로도
Appendix B에서 파인튜닝된 XLM-RoBERTa-Large가 few-shot LLM보다 낫다고 보고했지만
(84K 학습셋 기준), 그 방향을 우리 재현에서는 한 번도 안 밟아봤습니다.

## 방법

이미 계산해둔 `all-MiniLM-L6-v2` 문장 임베딩(NN-PPI 이웃 검색에 쓰던 것과 동일) 위에,
calib set의 (임베딩, 라벨) 쌍으로 가벼운 분류기를 직접 학습합니다. **추가 LLM 호출도,
GPU도, 파인튜닝도 필요 없습니다** — CPU에서 수 초 내로 끝납니다.

- `trained_classifier.py`: LogisticRegression / SVM(rbf) / MLP 비교, calib 5-fold CV로
  과적합 확인 후 test(원래부터 완전히 분리된 파일)에서 평가.
- `ensemble_with_embedding_clf.py`: SVM을 `probability=True`로 재학습해서 연속 확률을
  뽑고, 그 확률 + 원본 LLM 점수 + 재정렬 LLM 점수를 로지스틱 메타러너로 스택.

## 결과 (calib과 test는 원래부터 겹치는 claim이 0개인 별도 파일, 리키지 없음 확인됨)

| | LLM 원점수(baseline) | 임베딩 SVM 단독 | 스택(SVM+원본+재정렬) |
|---|---|---|---|
| CLEF (n_test=318) | 22.0% 오답 | 14.8% 오답 (AUC 0.928) | **12.9%** 오답 (상대 **-41.4%**) |
| ClaimBuster (n_test=2727) | 25.5% 오답 | 20.2% 오답 (AUC 0.830) | **19.9%** 오답 (상대 **-21.9%**) |

메타러너 계수(svm, orig, reordered) = CLEF `[9.124, 0.174, 0.497]`,
ClaimBuster `[8.411, -0.402, 0.706]` — **임베딩 분류기가 압도적 주력**이고 LLM 점수는
보조 신호로만 기여합니다.

## 오늘 다른 모든 개입과 비교하면

| 방법 | CLEF 오답률 개선(상대) | ClaimBuster 오답률 개선(상대) |
|---|---|---|
| 게이트+국소회귀 | -10.9% | -7.7% |
| 프롬프트 재정렬 | -11.4% (dataset-dependent) | +15.3%(악화) |
| 앙상블(원본+재정렬) | -4.1% | -3.9% |
| logprob 채널(digit) | 0% (정확도 동일) | +4.9%(악화) |
| **임베딩 분류기 스택** | **-41.4%** | **-21.9%** |

압도적으로 큽니다. 오늘 시도한 다른 모든 것을 합친 것보다 큽니다.

## 한계 — 정직하게

- 이건 더 이상 "NN-PPI 재현+개선"이 아니라 **완전히 다른 방법론(직접 지도학습)**입니다.
  논문에서는 이걸 NN-PPI의 대안/비교 기준선으로 명확히 구분해서 제시해야 합니다.
- calib set 크기에 의존적입니다 — ClaimBuster(1,308개)가 CLEF(2,406개)보다 개선폭이
  작은 것도 이와 일치합니다.
- 원 논문의 "재학습 없이 冷동결 LLM만으로" 라는 전제 자체를 벗어납니다. 이게 논문의
  기여를 "NN-PPI 개선"에서 "냉동결 LLM 후보정의 근본적 한계 + 대안 제시"로 격상시킵니다.
- SVM은 `probability=True`일 때 내부적으로 5-fold CV + Platt scaling을 쓰므로 계산이
  느립니다(ClaimBuster 1,308개 기준 수 초~수십 초). 더 큰 calib set에는 `LogisticRegression`이
  실용적입니다(성능 거의 동일, 훨씬 빠름).

## 재현

```
cd experiments/embedding-classifier
pip install scikit-learn sentence-transformers
python trained_classifier.py
python ensemble_with_embedding_clf.py
```

GPU도, 로컬 LLM 서버도 필요 없습니다. 이미 계산된 `results/*_scores.jsonl`(원본)과
`../prompt-reorder-ablation/results/*_scores_reordered.jsonl`만 있으면 됩니다.
