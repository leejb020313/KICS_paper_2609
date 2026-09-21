# 2026-09-21/22 세션 로그: 학회 수준 검증 + 추가 기여점 탐색

이 문서는 그날그날의 대화를 문서화한 것이 아니라, **재현 가능한 판단 근거**를 남기기 위한 것이다.
"왜 이 방향을 접었는지", "왜 이 방향은 아직 열려 있는지"를 다음 세션(또는 다른 사람)이
같은 검색을 반복하지 않고 이어갈 수 있게 하는 게 목적이다.

## 배경

오늘 세션의 대부분은 별개 프로젝트(`thesis-tictoc`, TicToc 경과시간 논문)의 "readout 기하"
헤드라인을 쫓다가 실패한 기록이다(`thesis-tictoc` 레포 `results/SUMMARY.md` §10-12,
그리고 이 프로젝트 메모리의 `project_kics_handoff_staleness.md` 참고). 그 실패 이후 두
가지를 했다:

1. **이 논문(NN-PPI 재현)이 KICS 학부/일반 세션 기준으로 충분한지 실측 검증**
2. **추가 기여점(§2.6 대체 또는 보강)을 위한 새 도메인 응용 탐색**

## 1. 학회 수준 검증 — 실제 논문 원문 대조

"내 첫 논문이 이딴거일 순 없어"라는 불안에 대해, 추측이 아니라 **실제로 KICS 학회
사이트(conf.kics.or.kr)를 크롤링하고 원문 PDF를 읽어서** 비교했다.

### 확인한 것

- **2026년도 하계종합학술발표회(2026s) 프로그램 전체**를 `conf.kics.or.kr/2026s/program`에서
  긁음. LLM 세션(20A), 학부논문 세션(36A-P, 36B-P) 등 약 100편의 실제 제목 확보.
- **학부논문 트랙 전체(36A-P, 36B-P, 약 70편)**: 전국 20개 이상 대학. 제목 패턴이 거의 전부
  "OO 기법 성능/민감도/영향 분석", "OO 시스템 설계 및 구현", "OO 구조 개선 연구"였다.
  "완전히 새로운 아키텍처 발명"류 제목은 하나도 없었다.
- **국립한밭대 오라 논문 2편 원문 전체를 다운로드해서 읽음**:
  - "시간 윈도우 기반 비콘 프레임 간격 분석을 통한 Fake AP 탐지" — Random Forest,
    Accuracy 0.99, 참고문헌 2개, **통계 검정 없음**, 단일 실험 환경.
  - "이중 경로 및 적응형 주파수 게이팅 기반 대규모 MIMO CSI 피드백 압축 기법"
    (DPAS-CsiNet) — IEEE TWC/TVT 게재 이력 있는 지도교수 랩, 4개 압축률×2개 시나리오
    비교. 결론에 스스로 "ablation study는 향후 과제"라고 명시. **이것도 통계 검정 없음.**
- **2025년도 하계종합학술발표회 우수논문상 수상자 인터뷰**(kics.or.kr 웹진) 확인:
  수상작은 "DSVT(기존 SOTA)의 구조적 빈틈을 찾아 고친" 논문이었고, 수상자 본인이
  후학에게 "SOTA를 맹목적으로 따르지 말고 빈틈을 봐라"고 조언함.
- **동료가 KICS 학부 경진대회에 낸 논문**(TTS speculative decoding, CosyVoice2 기반
  NAR drafter + 선행샘플 보정) 원문 전체 검토: 기술적으로는 더 야심차지만, 관련연구에서
  언급한 두 경쟁기법(FIRP, Domino)을 실험에서 전혀 비교하지 않았고, 이것도 통계 검정 없음.

### 결론

**이 프로젝트(NN-PPI 재현)의 초안이 실제로 대조한 모든 비교 대상보다 통계적으로 엄밀하다.**
McNemar exact test, 클러스터 부트스트랩 95% CI, leakage-free pool/tune 분리, 정직한
부정적 결과 보고(conformal 5종 실패) — 이 넷 중 어느 것도 위 어떤 실제 KICS 논문에도
없었다. 이건 자기위안이 아니라 원문 대조 결과다.

**따라서 초안은 이미 제출 가능한 완성도다.** 다음 섹션은 "더 큰 걸 찾아야 한다"는 압박
때문에 계속한 탐색이며, 필수가 아니라 보너스다.

## 2. 추가 기여점 탐색 — 시도하고 기각한 것들

기법 자체(clip + purity/유사도 신뢰도게이트 + ridge 국소회귀, 원고 식 2)를 **다른 도메인에
적용**해서 §2.6(현재는 임베딩 분류기 대조 문단)을 보강하거나 대체할 수 있는지 탐색했다.

### 기각 1 — 채널용량/오라클상한을 헤드라인으로 승격

REPORT.md §4.6("verbalized score가 13-14개 이산값으로 붕괴, CLEF 오라클 상한이 threshold
정확도와 정확히 같음")을 새 발견처럼 밀려고 했으나:

> **[The Score Granularity Gap in Black-Box LLM Classification](https://www.alphaxiv.org/abs/2606.22179)**
> (2026-06-20, Sun/Sun/Geng) — 9개 LLM × 3개 벤치마크(BoolQ/MNLI/PubMedQA)에서 정확히
> 같은 현상("verbalized confidence가 몇 개 안 되는 이산값으로 붕괴")을 **granularity gap**이라는
> 이름으로 이미 체계적으로 정리함. **기각.**

### 기각 2 — 위 논문의 벤치마크(BoolQ/MNLI/PubMedQA, Llama-3.1-8B)에 kNN 보정 적용

Score Granularity Gap 논문의 "weak model" 그룹에 Llama-3.1-8B가 있어 로컬 재현 가능하고,
그 논문이 시도한 7가지 confidence 구성법에 없는 8번째(kNN 잔차보정)를 추가하는 안.
처음엔 제목만 보고 "안전"이라고 잘못 판단했다가, 사용자가 재검증을 요구해서 전문을 읽음:

> **[Improved Confidence Estimates for Black-Box LLMs](https://www.alphaxiv.org/abs/2608.19323)**
> (2026-08-19, Layer 6 AI/TD Insurance) — reference set에서 k=20 최근접이웃을 뽑아
> 이웃 라벨 평균/표준편차 + 유사도 평균/표준편차를 feature로 만들고, 지도학습
> 분류기(Auto-ML: L1-로지스틱/Random Forest)로 confidence를 보정. **핵심 기전이
> 사실상 동일**(calibration set 이웃 → 보정). 데이터셋·평가지표(AUROC/ECE, granularity
> 아님)는 다르지만 심사자가 "저 논문을 다른 벤치마크에 옮긴 것 아니냐"고 물으면
> 반박이 궁색함. **기각.**

**교훈(메모리 `feedback_game_changer_pressure.md`에도 기록됨):** 제목/초록만 보고
"근접만 함, 배제 못함"이라고 판단한 게 틀렸다. 방법론 섹션을 실제로 읽어야
진짜 확인이다.

### 열려 있음 — WSF-ARG+ (혐오표현 검증가치)

> **[When Hate Meets Facts](https://www.alphaxiv.org/abs/2603.25269)**
> (2026-03-26, Ocampo/Caselli/Ceolin, CWI Amsterdam + Groningen) — 혐오표현+검증가치를
> 결합한 첫 데이터셋 WSF-ARG+를 공개(GitHub, 코드 포함). Llama-3.1-8B-Instruct,
> Qwen2.5-7B-Instruct 등 로컬 재현 가능한 모델 12종으로 테스트. **F1만 다룸 —
> "calibration"이라는 단어가 논문에 아예 안 나옴.** 검색 3바퀴(임베딩+키워드,
> 두 번 다른 표현으로) 전부 깨끗 — calibration/잔차보정을 이 데이터셋이나
> 이 주제에 결합한 후속연구 없음.

### 열려 있음, 오늘 새로 찾음, WSF-ARG+보다 유망 — Safety-Flag (콘텐츠 검열)

> **[Safety-Flag: A Unified Benchmark for the Reliability and Calibration of LLM
> Content Moderators](https://www.alphaxiv.org/abs/2609.19072)** (Yibo Hu, Illinois
> Institute of Technology, 2026년 9월) — BeaverTails/XSTest/Ethics/WildGuard/Aegis/
> ToxiChat/ToxiGen 7개 안전성 벤치마크를 하나의 flag/do-not-flag 프로토콜로 통합.
> 범용 LLM 6종 + 전용 guard 모델 4종(Llama Guard 등, 로컬 재현 가능한 크기)을
> 같은 아이템에 대해 평가하고 **아이템별 결정·confidence 점수를 코드와 함께
> 전부 공개**(GitHub: `yibo-hu-lab/safety-flag-benchmark`).
>
> **핵심**: 이 논문은 **calibration을 정면으로 다루면서도**, 시도한 보정 기법이
> "모델당 전역 temperature scaling 하나"뿐이다(ECE 2.8-6.0배 개선이라고 보고).
> 형님이 이미 검증한 **국소(이웃 기반) 보정은 시도되지 않았다.** WSF-ARG+보다
> 두 가지 점에서 더 낫다: (1) 벤치마크 자체가 "calibration"을 주제로 하므로
> NN-PPI 논문과 직결되는 비교가 자연스럽고, (2) 아이템별 점수가 이미 공개돼 있어
> 일부는 새로 LLM을 돌리지 않고도 시작할 수 있을 가능성이 있다(확인 필요).
>
> 검색 2바퀴(키워드+임베딩, "Safety-Flag" 자기 인용 제외하면 관련 결과 없음) 전부 깨끗.

## 3. 다음 세션에서 할 일 (아직 아무것도 실행 안 함)

1. **go/no-go부터 정한다.** 초안은 이미 제출 가능하므로(§1), 아래는 순수 보너스다.
   시간이 부족하면 건너뛰고 원고를 그대로 낸다.
2. 진행한다면 **Safety-Flag를 먼저** 본다(WSF-ARG+보다 fit이 좋음):
   - GitHub 레포에서 아이템별 confidence 점수·라벨 형식을 확인 — 재추론 없이
     쓸 수 있는지, 아니면 로컬 guard 모델(Llama Guard 등)로 다시 돌려야 하는지 결정.
   - 저자들의 temperature-scaling 결과(ECE 2.8-6.0배 개선)를 먼저 재현해서
     게이트로 삼는다(이 프로젝트의 모든 재현 단계와 같은 규율).
   - 게이트 통과하면, 형님의 clip+게이트+국소회귀(원고 식 2)를 얹어서
     temperature scaling보다 ECE·selective risk가 더 낮아지는지 확인.
3. Safety-Flag가 막히면 **WSF-ARG+로 폴백**(§2의 "열려 있음" 항목, 절차는 동일).
4. 어느 쪽이든 **"우리가 원리를 발명했다"가 아니라 "우리가 이미 검증한 기법이
   저자들이 시도 안 한 더 나은 보정임을 보였다"로 서술한다** — 이게 §1에서
   확인한 안전한 프레이밍이고, novelty 검색에서 살아남은 유일한 각도다.
