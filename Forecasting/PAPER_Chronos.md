# Chronos — 시계열을 언어처럼 토큰화했다가, 3년에 걸쳐 그 전제를 스스로 지워온 계보

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | Chronos: Learning the Language of Time Series |
| **저자** | Abdul Fatir Ansari*, Lorenzo Stella*, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, Yuyang Wang (*공동 1저자) |
| **소속** | AWS AI Labs (+ UC San Diego, Rutgers, ICSI/LBNL/UC Berkeley, NYU 겸직) |
| **공개일** | 2024-03-12 (arXiv v1) → 2024-11-04 (v3, 최종) |
| **학회** | **TMLR 2024** (Transactions on Machine Learning Research) |
| **분야** | Time Series Foundation Model(시계열 기반 모델), Zero-shot Forecasting(제로샷 예측), Probabilistic Forecasting(확률 예측) |
| **arXiv abstract** | https://arxiv.org/abs/2403.07815 |
| **arXiv HTML (v3)** | https://arxiv.org/html/2403.07815v3 |
| **OpenReview** | https://openreview.net/forum?id=DOz10U6BKO (저장소 README는 `gerNCVqqtR`로 링크) |
| **Chronos-2 기술보고서** | https://arxiv.org/abs/2510.15821 |
| **공식 코드** | https://github.com/amazon-science/chronos-forecasting (Apache-2.0) |
| **모델** | https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444 |
| **데이터** | https://huggingface.co/datasets/autogluon/chronos_datasets (+ `_extra`) |
| **벤치마크 도구** | https://github.com/autogluon/fev |
| **베이스 모델** | **T5** (`google/t5-efficient-*`) — 단 **가중치는 안 씀**, 구조만 빌리고 random init(무작위 초기화). GPT-2 변형도 실험 |
| **학습 자원** | AWS EC2, A100(40GB) 8장, 200K step |
| **리뷰 시점 코드 상태** | 커밋 `8589d19` (2026-08-14), 패키지 버전 2.3.1 |

> ⚠️ **문서 범위**: 이 문서는 논문(Chronos-1)만이 아니라 **저장소에 실제로 들어 있는 3세대(Chronos / Chronos-Bolt / Chronos-2) 전부**를 다룹니다. 저장소를 열어보면 논문에 해당하는 코드는 이제 셋 중 가장 낡고 가장 안 쓰이는 부분이기 때문입니다.

---

## 📖 주요 용어 사전 (Glossary)

*이 논문은 NLP 용어를 시계열에 그대로 이식하기 때문에, "언어 쪽 단어가 시계열에서 무엇을 뜻하게 되는지"를 매칭해두면 본문이 거의 다 읽힙니다.*

### 언어 → 시계열 용어 이식

| 용어 | 풀이 |
|---|---|
| **tokenization(토큰화)** | 연속적인 실숫값 시계열을 **정수 토큰 번호**로 바꾸는 것. Chronos의 심장. |
| **quantization(양자화)** | 실수 범위를 여러 칸(bin)으로 잘라 각 값을 "몇 번째 칸"으로 바꾸는 것. Chronos는 4096칸. |
| **vocabulary(어휘)** | 모델이 쓸 수 있는 토큰 번호의 전체 목록. 여기서는 특수토큰 2개 + 값 칸 4093개. |
| **cross-entropy loss(교차 엔트로피 손실)** | "정답 칸 번호를 맞히는" 분류 손실. Chronos-1은 **거리 개념이 없는** 이 손실을 쓴다 — 3번 칸을 4번으로 틀리든 4000번으로 틀리든 벌점이 같다. |
| **autoregressive sampling(자기회귀 샘플링)** | 한 칸씩 뽑아 이어붙이며 미래를 생성. Chronos-1은 이걸 20번 반복해 20개 경로(sample path)를 만든다. |

### 예측 문제의 구조

| 용어 | 풀이 |
|---|---|
| **zero-shot forecasting(제로샷 예측)** | 그 데이터로 **학습한 적 없이** 바로 예측. 이 논문의 주 무대. |
| **in-domain(도메인 내)** | 학습 코퍼스에 포함된 데이터로 평가. 15개 데이터셋. |
| **probabilistic forecast(확률 예측)** | 값 하나가 아니라 **분포**를 내놓는 것. "80% 확률로 100~120 사이" 같은 예측 구간. |
| **quantile(분위수)** | 분포를 대표하는 지점들. 0.1분위 = 하위 10% 지점. Bolt/Chronos-2는 0.1~0.9의 9개를 직접 회귀한다. |
| **covariate(공변량)** | 타깃 말고 예측에 도움이 되는 부가 시계열. 미래를 아는 것(예: 세일 일정)과 모르는 것이 있다. |
| **direct multi-step forecasting(직접 다중스텝 예측)** | 미래 여러 칸을 **한 번에** 뽑는 방식. 자기회귀 루프가 없어 빠르고 오차 누적이 없다. Bolt의 핵심. |

### 정규화(scaling) — 3세대에 걸쳐 계속 바뀌는 부분

| 용어 | 풀이 |
|---|---|
| **mean scaling(평균 스케일링)** | 절댓값 평균으로 나누기만 한다. **중심을 빼지 않는다**(Chronos-1). |
| **standardization(표준화)** | 평균을 빼고 표준편차로 나눈다(Bolt, Chronos-2). instance normalization이라고도 부른다. |
| **arcsinh** | 큰 값은 압축하고 작은 값은 거의 그대로 두는 부드러운 변환. Chronos-2의 선택적 옵션. |

### 평가 지표

| 용어 | 풀이 |
|---|---|
| **MASE (Mean Absolute Scaled Error)** | 점 예측 오차를 계절적 naive 예측 대비로 정규화한 값. **계열의 크기에 무관**. |
| **WQL (Weighted Quantile Loss)** | 확률 예측 품질. 분위수 예측이 얼마나 잘 맞는지. **계열의 크기에 의존**. |
| **aggregated relative score(집계 상대점수)** | 각 데이터셋에서 seasonal naive 대비 비율을 구하고 기하평균. **1보다 작으면 naive보다 좋음.** 저장소의 공식 결과 형식. |

### 데이터 증강 (Chronos-1의 두 기둥)

| 용어 | 풀이 |
|---|---|
| **TSMixup** | 실제 시계열 여러 개를 무작위 비율로 **볼록 결합(convex combination)** 해 새 시계열을 만드는 증강. 1000만 개 생성. |
| **KernelSynth** | Gaussian Process(가우시안 과정)의 커널을 무작위로 조합(더하기/곱하기)해 **완전 합성** 시계열 생성. 100만 개. |

---

## 📝 논문 요약 (TL;DR)

**한 줄**: 시계열 값을 스케일링 + 양자화로 **정수 토큰**으로 바꾸고, 구조를 하나도 안 고친 T5에 교차 엔트로피로 학습시켰더니, 그 데이터를 본 적 없어도(zero-shot) 그 데이터 전용으로 학습된 모델과 겨룰 수준이 됐다.

**문제**: 예측 모델은 보통 데이터셋마다 새로 학습해야 한다. 데이터셋이 1000개면 모델도 1000개다. LLM처럼 **하나의 사전학습 모델로 아무 시계열이나** 예측할 수 없을까?

**해결책**:
1. **토큰화** — 시계열을 절댓값 평균으로 나누고(mean scaling), [-15, +15] 범위를 4093칸으로 잘라 정수로 바꾼다.
2. **구조 재사용** — T5를 그대로 쓴다. 시계열 전용 부품을 하나도 안 만든다.
3. **데이터 증강** — 실데이터 28개가 부족하니 TSMixup 1000만 + KernelSynth 100만으로 불린다(9:1 비율).
4. **확률 예측** — 20번 샘플링해 경로 20개를 만들고 거기서 분위수를 뽑는다.

**검증**: 42개 데이터셋(in-domain 15 + zero-shot 27). 제로샷에서 그 데이터 전용 학습 모델과 대등하거나 우세.

**주목할 음성 결과(negative result)**: T5의 **언어 사전학습 가중치를 쓰면 오히려 손해다.** 무작위 초기화가 최종 손실도 낮고 성능도 같거나 낫다(논문 §5.6, Figure 8·9). 제목이 "Learning the Language of Time Series"인데, 정작 **언어 지식은 전이되지 않는다**는 걸 논문 스스로 보여준다. 이식된 것은 지식이 아니라 **구조와 학습 레시피**뿐이다.

---

## 🎯 핵심 기여 (Contributions)

1. **최소주의 프레임워크**: 시계열 전용 구조를 만들지 않고 토큰화만으로 LM을 예측기로 전용. 재현·확장이 쉽다.
2. **데이터 증강 2종**: TSMixup(실데이터 혼합)과 KernelSynth(GP 합성)로 학습 코퍼스를 인위적으로 확장. 제로샷 성능 향상의 실질적 원인.
3. **대규모 공개 평가**: 42개 데이터셋 벤치마크 + 모델·데이터·평가 결과 전면 공개.
4. **음성 결과 보고**: LLM 초기화 무용, 어휘 크기 트레이드오프, 토큰화의 정밀도 붕괴 조건을 **스스로 문서화**(§5.6, §5.7).

---

## 🧩 주요 알고리즘 설명

### 3.1 토큰화 — 논문 전체가 이 40줄에 걸려 있다

*이 절을 두는 이유: Chronos의 성능도, 한계도, 이후 두 세대가 이걸 갈아엎은 이유도 전부 여기서 나온다.*

코드: `src/chronos/chronos.py`의 `MeanScaleUniformBins` 클래스.

```
1) scale = mean(|x|)                    # 절댓값 평균. 중심(평균)을 빼지 않는다!
2) x_scaled = x / scale
3) token = bucketize(x_scaled, 경계 4094개) + 2   # [-15, +15]를 4093칸으로
4) 결측(NaN) → PAD 토큰
```

*(말로 풀면: 계열의 "전형적 크기"로 나눠 −15~+15 범위에 밀어 넣고, 그 범위를 4093칸 자로 재서 몇 번째 칸인지를 토큰 번호로 삼는다.)*

역변환은 각 칸의 중심값을 꺼내 `scale`을 다시 곱한다.

**여기서 파생되는 3가지 구조적 한계** (셋 다 논문 §5.7이 스스로 인정):

| 한계 | 왜 생기나 | 언제 터지나 |
|---|---|---|
| **overflow(범위 초과)** | 표현 가능 범위가 ±15×scale로 고정 | 희소 스파이크 계열. scale이 작으면 스파이크가 잘려나감 |
| **정밀도 손실** | 칸 간격이 30×scale/4093으로 고정 | **평균이 크고 변동이 작은 계열.** 예: 1000 근처에서 ±1 흔들리면 4093칸 중 서너 개만 씀 |
| **거리 개념 없음** | 교차 엔트로피는 칸 번호 간 거리를 모름 | 3번을 4번으로 틀리든 4000번으로 틀리든 벌점이 같음 |

세 번째는 논문이 "충분한 데이터로 인접 칸의 확률 구조를 알아서 배운다"고 옹호하지만, **Bolt와 Chronos-2가 둘 다 pinball loss(분위 손실)로 갈아탄 것**이 사실상의 답이다.

### 3.2 3세대 구조 비교 — 저장소를 열면 보이는 것

*이 절을 두는 이유: README만 보면 "Chronos 패밀리"로 뭉뚱그려지지만, 코드상 셋은 손실 함수부터 구조까지 다른 별개 모델이다.*

| | **Chronos** (2024-03) | **Chronos-Bolt** (2024-11) | **Chronos-2** (2025-10) |
|---|---|---|---|
| 파일 | `chronos.py` (560줄) | `chronos_bolt.py` (667줄) | `chronos2/` (약 3,200줄) |
| 발상 | 시계열을 **언어처럼 토큰화** | **패치** + 직접 다중스텝 | **인코더 전용** + 그룹 어텐션 |
| 백본 | T5 seq2seq (통째) | T5 encoder-decoder 개조 | 자체 구현(T5 스타일 + RoPE) |
| 공개 크기 | 8M / 20M / 46M / 200M / 710M | 9M / 21M / 48M / 205M | 28M / 120M |
| 입력 단위 | 값 1개 = 토큰 1개 | 16스텝 패치 = 토큰 1개 | 패치 = 토큰 1개 |
| 입력 특징 | 토큰 ID | 값 + 마스크 (패치×2) | **시간 인코딩 + 값 + 마스크** (패치×3) |
| 스케일링 | mean scaling(중심화 없음) | 표준화 | 표준화 + arcsinh 옵션 |
| 손실 | cross-entropy | pinball(분위) | pinball(분위) |
| 출력 | 20회 샘플링 → 경로 20개 | 9분위 **1회** 직접 회귀 | 9분위 × 패치 |
| 다변량·공변량 | ❌ | ❌ | ✅ (`group_ids` 마스크) |
| 학습 코드 | ✅ `scripts/training/train.py` | ❌ **없음** | 파인튜닝(`fit()`)만 |

### 3.3 Chronos-Bolt — 250배 빠른 진짜 이유

*이 절을 두는 이유: "250배"라는 숫자만 보면 커널 최적화 같은 걸 상상하게 되는데, 실제로는 "LLM 흉내를 그만둔 것"이 전부다.*

바뀐 건 셋뿐이고, 셋 다 자기회귀를 제거하는 방향이다.

1. **패치화** (`Patch` 클래스): 16스텝을 묶어 토큰 1개. 컨텍스트 2048 → 토큰 128개. 시퀀스 길이가 16분의 1.
2. **디코더 1스텝** (`decode()`): decoder start 토큰 **하나만** 넣고 은닉상태 `(batch, 1, d_model)` **하나만** 받는다. 그걸 `output_patch_embedding`(ResidualBlock 한 장)에 넣어 **9분위 × 64스텝 = 576개 값을 한 번에** 뱉는다. 자기회귀 루프 소멸.
3. **분위 회귀**: 샘플링 20회 → 결정론적 1회.

```
[컨텍스트 2048]
   → 표준화 → 16칸씩 패치 → 값+마스크 concat
   → ResidualBlock → 토큰 128개
   → T5 인코더
   → 디코더 1스텝 (start 토큰 1개)
   → 은닉 (B,1,d) → ResidualBlock → (B, 9분위, 64스텝)   ← 한 방
```

### 3.4 Chronos-2 — 마스크 하나로 다변량·공변량을 통일

*이 절을 두는 이유: Chronos-1 논문 §6.1이 "공변량/다변량은 조합이 너무 많아 단일 모델로 다루기 어렵다"고 미해결 과제로 남긴 것을, 18개월 뒤 같은 팀이 **구조 변경 없이 마스크만으로** 풀었기 때문.*

인코더 블록(`chronos2/model.py:48`)은 3층이다.

```
Chronos2EncoderBlock
 ├─ TimeSelfAttention   (RoPE 사용)   ← 시간축: 같은 계열의 패치들끼리
 ├─ GroupSelfAttention  (RoPE 없음)   ← 변량축: 같은 그룹의 계열들끼리
 └─ FeedForward
```

`GroupSelfAttention`(`chronos2/layers.py:388`)은 `rearrange(x, "batch time d -> time batch d")`로 축을 뒤집은 뒤 **배치축을 따라** 어텐션한다. 코드 주석도 솔직하다 — *"배치축에는 자연스러운 순서가 없으므로 RoPE를 쓰지 않는다."*

**`group_ids`가 마법의 스위치다.** 같은 그룹 id끼리만 섞이도록 마스크를 걸면, **하나의 배치 안에 이질적인 과제를 섞을 수 있다**:

| `group_ids` | 미래 공변량 | 무슨 과제가 되나 |
|---|---|---|
| `[0, 0]` | 둘 다 NaN | 두 계열의 **다변량 공동 예측** |
| `[1, 1, 1]` | 앞 둘만 값 제공 | 앞 둘은 **미래를 아는 공변량**, 셋째가 타깃 |
| `[2]` | NaN | **단변량 독립 예측** |

구조를 바꾸지 않고 **입력 마스크만으로** 세 과제를 통일했다. 미래 패치도 컨텍스트와 **같은 임베딩 층**을 태워 시퀀스 뒤에 붙이고, 마지막 `num_output_patches`개 은닉상태만 잘라 분위 헤드에 넣는다 — **디코더가 아예 없다.**

---

## 📊 실험 요약

### 4.1 벤치마크 구성

*이 표를 보는 이유: "42개 데이터셋"이라는 숫자가 어떻게 나뉘는지 알아야 in-domain / zero-shot 수치를 오해하지 않는다.*

| 벤치마크 | 개수 | 내용 | 저장소 config |
|---|---|---|---|
| **Benchmark I (in-domain)** | 15개 | 학습 코퍼스에 포함. m4_daily/hourly/monthly/weekly, electricity_15min, taxi_30min, uber_tlc 등 | `configs/in-domain.yaml` |
| **Benchmark II (zero-shot)** | 27개 | 학습 때 **본 적 없음**. ETTh, ETTm, m5, exchange_rate, monash 계열 다수 | `configs/zero-shot.yaml` |

### 4.2 공식 결과 실측 (저장소 `scripts/evaluation/results/` 직접 집계)

*이 표를 만든 이유: 논문 그림(Figure 7b)의 스케일링 주장과 저장소가 공개한 실제 숫자가 어긋나는지 확인하기 위해.*

seasonal naive 대비 집계 상대점수. **낮을수록 좋음.**

| 모델 | ID-MASE | ID-WQL | ZS-MASE | ZS-WQL |
|---|---|---|---|---|
| chronos-t5-tiny (8M) | 0.7649 | 0.6289 | 0.8705 | 0.7109 |
| chronos-t5-mini (20M) | 0.7250 | 0.5965 | 0.8412 | 0.6888 |
| chronos-t5-small (46M) | 0.7296 | 0.6087 | 0.8304 | 0.6650 |
| chronos-t5-base (200M) | 0.7008 | 0.5786 | **0.8155** | **0.6425** |
| chronos-t5-large (710M) | **0.6945** | **0.5597** | 0.8214 | 0.6505 |
| chronos-bolt-tiny (9M) | 0.7403 | 0.5734 | 0.8445 | 0.6679 |
| chronos-bolt-mini (21M) | 0.7268 | 0.5651 | 0.8222 | 0.6442 |
| chronos-bolt-small (48M) | 0.7031 | 0.5444 | 0.8192 | 0.6356 |
| chronos-bolt-base (205M) | 0.6800 | 0.5339 | **0.7915** | **0.6241** |

### 4.3 Bolt vs T5 동급 비교 (양수 = Bolt가 좋음)

| 크기 | ID-MASE | ID-WQL | ZS-MASE | ZS-WQL |
|---|---|---|---|---|
| tiny | +3.2% | +8.8% | +3.0% | +6.1% |
| mini | **−0.3%** | +5.3% | +2.3% | +6.5% |
| small | +3.6% | +10.6% | +1.3% | +4.4% |
| base | +3.0% | +7.7% | +3.0% | +2.9% |

8개 조합 평균 4.6%로 README의 "5% lower error"에 얼추 맞지만, **이득이 WQL(확률 예측)에 몰려 있다.** MASE(점 예측)만 보면 평균 2.4%이고, bolt-mini는 in-domain에서 오히려 0.3% 나쁘다. 정직한 서술은 *"확률 예측이 5~10% 좋아지고 점 예측은 2~3%"*이다.

### 4.4 논문이 보고한 하이퍼파라미터 분석 (§5.6)

*이 절을 두는 이유: 이 논문에서 가장 재사용 가치가 높은 부분이 메인 결과가 아니라 이 음성 결과들이기 때문.*

| 실험 | 결론 |
|---|---|
| **LLM 초기화** | **무용하거나 해롭다.** 무작위 초기화가 최종 학습 손실이 더 낮다. Base·Large는 LM 가중치로 시작하면 초반만 빠르고 결국 더 높은 손실에 수렴 |
| **TSMixup** | in-domain은 그대로, **zero-shot만 개선**. 즉 다양성 확보용 |
| **KernelSynth 비율** | 소량(약 10%) 섞을 때 zero-shot이 추가 개선 |
| **컨텍스트 길이** | 1024까지는 개선, 그 이상은 포화/악화. 논문은 "고빈도 데이터셋이 평가셋에 부족해서일 수 있다"고 유보 |
| **어휘 크기** | MASE는 커질수록 개선, **WQL은 커지면 악화**. 정밀도↑ ↔ 칸당 데이터↓ 트레이드오프 |

---

## 🔍 급소 — 확인된 문제들

### 5.1 [논문 ↔ 공개 결과 충돌] "모델이 클수록 좋다"가 zero-shot에서 성립하지 않는다 ⭐

논문 §5.6 "Model size" 항목은 이렇게 말한다:

> *"downstream model performance — it improves with the model size **for both in-domain and zero-shot benchmarks** ... These trends suggest that even larger models may improve performance further."*

그런데 **같은 팀이 저장소에 공개한 공식 평가 CSV**(→ 4.2)를 보면:

| | ZS-MASE | ZS-WQL |
|---|---|---|
| chronos-t5-base (200M) | **0.8155** | **0.6425** |
| chronos-t5-large (710M) | 0.8214 (더 나쁨) | 0.6505 (더 나쁨) |

**710M이 200M보다 zero-shot에서 두 지표 모두 나쁘다.** 파라미터를 3.5배 늘린 값이 제로샷에서는 마이너스다. in-domain에서만 large가 이긴다(0.6945 / 0.5597) — 이건 **더 큰 모델이 학습 코퍼스에 더 과적합했다**는 전형적 패턴으로 읽힌다.

논문의 Figure 7b가 다른 집계 방식이었을 가능성은 있지만, **저장소가 재현용으로 공개한 숫자로는 논문의 스케일링 주장이 재현되지 않는다.** "even larger models may improve further"라는 전망은 이 데이터로는 지지되지 않는다.

### 5.2 [재현성] 논문의 데이터 증강 절반이 코드에 없다

논문의 두 기둥은 TSMixup과 KernelSynth인데:

| | 코드 | 상태 |
|---|---|---|
| KernelSynth | `scripts/kernel-synth.py` (200줄) | ✅ 있음 |
| **TSMixup** | — | ❌ **없음** |

학습 config(`chronos-t5-small.yaml`)는 이미 만들어진 `/home/ubuntu/tsmixup-data.arrow`를 가리킬 뿐이다. §5.6이 "TSMixup이 zero-shot 개선의 원인"이라고 밝힌 그 증강의 생성 코드가 없다.

### 5.3 [재현성] 지금 실제로 쓰이는 두 모델은 사전학습 재현이 불가능하다

| 모델 | 사전학습 코드 | 파인튜닝 |
|---|---|---|
| Chronos-1 | ✅ `scripts/training/train.py` | ✅ (같은 스크립트) |
| **Chronos-Bolt** | ❌ 없음 | ❌ 없음 |
| **Chronos-2** | ❌ 없음 | ✅ `Chronos2Pipeline.fit()` (full / LoRA) |

즉 **논문 모델만 재현 가능하고, 실제 권장 모델 두 개는 추론과 파인튜닝만 된다.** 게다가 `scripts/README.md`는 스스로 경고한다 — *"이 문서는 2024년 3월 모델용으로 쓰였고 최신이 아닐 수 있습니다."*

### 5.4 [검증 불가 주장] "250배 빠름 / 20배 메모리 효율"에 벤치마크가 없다

README 최상단의 대표 주장인데, 저장소에 **속도·메모리 측정 스크립트가 없다.** 출처는 AWS 블로그 글이다. 구조상(자기회귀 20회 × 64스텝 → 1회 forward) 큰 폭의 개선이 있는 건 명백하지만, **250이라는 숫자 자체는 재현 경로가 없다.**

### 5.5 [코드 버그] Chronos-Bolt 손실 함수의 축 주석이 뒤바뀌어 있다

`src/chronos/chronos_bolt.py:373-375`:

```python
loss = loss.mean(dim=-2)  # Mean over prediction horizon    ← 실제로는 분위축
loss = loss.sum(dim=-1)   # Sum over quantile levels        ← 실제로는 호라이즌축
loss = loss.mean()
```

텐서 모양은 `(batch_size, num_quantiles, prediction_length)`이므로 `dim=-2`는 **분위축**, `dim=-1`은 **호라이즌축**이다. 주석이 정확히 반대다.

같은 저장소의 `chronos2/model.py:558`은 제대로 되어 있다:
```python
loss = loss.mean(dim=-1).sum(dim=-1).mean()   # mean over horizon, sum over quantiles — 주석도 정확
```

**영향**: 수치적으로는 상수배(Q=9, H=64이므로 약 7.1배) 차이라 학습률에 흡수된다. 하지만 **두 세대의 손실 스케일이 7배 다르다**는 뜻이므로, Bolt의 학습률 설정을 Chronos-2에 그대로 옮기면 어긋난다. 주석만 믿고 읽는 사람은 반드시 헷갈린다.

### 5.6 [설계 결함, 저장소가 스스로 인정] Chronos-1의 장기 예측은 불확실성이 붕괴한다

`chronos.py:507`:
```python
context_tensor = torch.cat([context_tensor, prediction.median(dim=1).values], dim=-1)
```

예측 길이가 모델의 64스텝을 넘으면 롤아웃(rollout)하는데, **컨텍스트에 붙이는 게 중앙값 하나뿐**이다. 20개 경로를 만들어 놓고 그중 가운데 하나만 이어붙이니, 두 번째 블록부터는 "확신에 찬 하나의 미래"를 전제로 예측한다 → 예측 구간이 실제보다 좁아진다.

이건 추측이 아니라 **저장소가 명시적으로 인정한 내용**이다. `chronos2/pipeline.py:409` 주석:
> *"Note that this effectively leads to shrinking of the probability space but it is **better heuristic than just using the median to unroll, which leads to uncertainty collapse.**"*

세대별 대응:

| 세대 | 롤아웃 방식 |
|---|---|
| Chronos-1 | 중앙값 1개만 이어붙임 → **붕괴** |
| Chronos-Bolt | 9분위 전부를 각각 이어붙여 9×9=81개 "샘플" → 경험 분위수로 9개 축약 |
| Chronos-2 | 위와 같되 **각 경로에 확률질량 가중치**(사다리꼴 근사)를 부여해 `weighted_quantile`로 축약 |

### 5.7 [사소] 어휘의 한 칸이 도달 불가능하다

토크나이저 경계의 첫 값이 `-1e20`이라 `bucketize` 결과가 항상 1 이상이고, 특수토큰 오프셋 2를 더하면 최소 토큰 id가 3이다. **id 2번은 어떤 입력으로도 나오지 않는다.** 어휘 4096 중 실사용은 4093개. 성능 영향은 없지만 "4096 vocabulary"라는 표현은 정확히는 4093이다.

### 5.8 [문서화 누락] Chronos-2의 손실이 결측 타깃을 조용히 감가한다

`chronos2/model.py`의 `_compute_loss`는 마스킹된 미래 타깃을 **분모에 포함한 채** `mean(dim=-1)`을 한다. 즉 미래 결측이 많은 계열은 자동으로 손실 가중치가 낮아진다. 의도된 설계일 수 있으나 문서에는 한 줄도 없다. (Bolt는 `sum`이라 이 감가가 없다 → 두 세대의 결측 처리 철학이 다르다.)

---

## 🛠 코드 품질 총평

*이 절을 두는 이유: 급소를 여럿 지적했으니, 이 저장소를 실제로 쓸지 말지 판단할 균형점이 필요하다.*

**연구 코드로서는 상위권이다.**

| 항목 | 평가 |
|---|---|
| 테스트 | 3,219줄 + 더미 체크포인트 4종(chronos / bolt / chronos2 / chronos2-lora) + CI |
| 라이브러리 호환 | `transformers` v4/v5 양쪽을 `_TRANSFORMERS_V5` 분기로 대응. v5에서 `_tied_weights_keys`가 list→dict로 바뀐 것까지 처리 |
| 가독성 | 축 변환을 전부 `einops` 문자열(`"batch time d -> time batch d"`)로 명시 — 시계열 코드에서 가장 헷갈리는 부분을 자기 문서화 |
| 실무 접점 | `predict_df`(pandas 입출력), `predict_fev`, SageMaker 배포 노트북, AutoGluon 연동 |
| 오류 처리 | `_validate_input`이 조용히 넘어가지 않고 구체적 메시지와 함께 예외를 던짐 |
| 의존성 | torch / transformers / accelerate / numpy / einops / pandas — 가볍다 |
| 라이선스 | Apache-2.0 (모델·코드 모두) |

---

## 💬 Q&A

### Q1. iTransformer는 "시간축 어텐션이 해롭다"고 했는데, Chronos-2는 왜 다시 쓰나? ⭐

이게 두 논문을 겹쳐 볼 때 가장 흥미로운 지점이다.

**[[PAPER_iTransformer]]의 결론**: 시간축 어텐션은 유해하다. 절제 실험에서 시간축에 어텐션을 넣으면 Traffic MSE가 0.428 → 0.913으로 **2배 폭발**했다. 그래서 시간축을 통째로 접어 토큰 1개로 만들고, 어텐션은 변량축에만 남겼다.

**Chronos-2의 선택**: 시간축 어텐션(`TimeSelfAttention`)과 변량축 어텐션(`GroupSelfAttention`)을 **둘 다** 쓴다. iTransformer 표에서 최악(0.913)이었던 "Attention / Attention" 조합이다.

무엇이 달라졌나:

| | iTransformer | Chronos-2 |
|---|---|---|
| 시간축 토큰 | 시계열 전체 = 토큰 1개 (어텐션 자체가 없음) | 패치 단위 토큰 + RoPE |
| 변량축 | 어텐션 | 어텐션 (그룹 마스크) |
| 학습 데이터 | ETT 학습 샘플 8,545개 등 **단일 데이터셋** | 대규모 다중 도메인 **사전학습 코퍼스** |
| 변량 개수 | 데이터셋마다 고정 | 배치 안에서 그룹별 가변 |

**핵심 차이는 데이터 규모다.** iTransformer의 "시간축 어텐션은 해롭다"는 결론은 *샘플 수천 개짜리 벤치마크에서* 성립한다. 어텐션은 파라미터 대비 데이터가 부족하면 과적합하고, 그때는 선형/FFN이 이긴다 — 이건 iTransformer 부록 G.1이 인용한 *"CI가 CD를 이기는 이유는 샘플 희소성"* 논의와 정확히 같은 메커니즘이다.

Chronos는 그 전제를 데이터로 걷어냈다. 그러자 시간축 어텐션이 다시 쓸모 있어졌고, iTransformer가 정착시킨 변량축 어텐션은 **버려진 게 아니라 흡수됐다** — `GroupSelfAttention`이라는 이름으로.

**거꾸로 iTransformer가 여전히 이기는 지점**도 명확하다. Chronos-2의 그룹 어텐션은 **매 패치 위치마다** 변량끼리 섞으므로 비용이 (시간축 토큰 수 × 변량²)로 붙는다. iTransformer는 시간축을 이미 접었으니 변량²만 낸다. 그리고 iTransformer의 "변량 20%만 학습하고 전체 예측" 트릭은 Chronos 쪽에 대응물이 없다.

### Q2. Chronos-1 논문이 남긴 미해결 과제 중 무엇이 풀렸나?

논문 §6.1이 명시한 한계와, 저장소 코드가 실제로 도달한 지점:

| 논문이 남긴 과제 | 현재 상태 |
|---|---|
| "공변량/다변량은 **조합이 너무 많아** 단일 모델로 다루기 어렵다. task-specific adapter나 LightGBM 스태킹이 대안일 수 있다" | **풀렸다.** Chronos-2가 어댑터도 스태킹도 없이 `group_ids` **마스크 하나**로 통일 |
| "추론이 느리다. 양자화·speculative decoding 같은 NLP 기법이 도움될 것" | **다르게 풀렸다.** NLP 기법을 가져온 게 아니라 **자기회귀 자체를 버려서**(Bolt) 해결 |
| "인코더 표현이 분류·이상탐지 등에 범용적일 것" | 미검증. `embed()` API는 세 세대 모두 제공하지만 다운스트림 실험은 없음 |
| "불규칙 샘플링 시계열" | 미해결 |

두 번째가 특히 아이러니하다. 논문은 "언어 모델 프레임워크를 썼으니 NLP의 발전이 그대로 전이된다"를 장점으로 내세웠는데, **실제 개선은 언어 모델 프레임워크를 버리는 방향에서 나왔다.**

### Q3. 그래서 어떤 걸 써야 하나?

| 상황 | 권장 |
|---|---|
| 그냥 잘 되는 제로샷 예측이 필요 | **chronos-bolt-base** 또는 **chronos-2**. 논문 모델(chronos-t5-*)을 새로 쓸 이유는 없다 |
| 공변량이 있음 (판촉 일정, 날씨 예보 등) | **chronos-2** 전용. 다른 세대는 지원 자체가 없다 |
| 여러 계열이 서로 영향을 줌 (다변량) | **chronos-2** + `group_ids` |
| CPU만 있음 / 지연시간 민감 | bolt-tiny(9M) ~ bolt-small(48M). t5 계열은 샘플링 20회라 느리다 |
| 자체 데이터로 사전학습부터 하고 싶음 | **Chronos-1만 가능**(`train.py`). Bolt·Chronos-2는 코드 없음 |
| 도메인 데이터로 미세조정 | chronos-2 `fit()` (full 또는 LoRA). 기본 학습률 1e-6이 매우 낮은 점 주의 |
| 논문 재현이 목적 | TSMixup 코드 부재로 **완전 재현 불가**. KernelSynth 부분만 가능 |

### Q4. "언어처럼 다룬다"는 프레이밍은 결국 맞았나?

**부분적으로만 맞았고, 저자들이 스스로 철회하는 중이다.**

| 주장 | 검증 결과 |
|---|---|
| LLM **가중치**가 전이된다 | ❌ 논문 §5.6이 직접 반증. 무작위 초기화가 더 낫다 |
| LLM **구조**를 재사용할 수 있다 | ✅ 맞다. T5를 거의 그대로 씀 |
| LLM **학습 레시피**(대규모 코퍼스 + 사전학습 + 제로샷)가 통한다 | ✅ 이게 진짜 기여 |
| LLM **토큰화·손실**(양자화 + 교차 엔트로피)이 적절하다 | ❌ Bolt와 Chronos-2가 둘 다 폐기 |
| LLM **자기회귀 생성**이 적절하다 | ❌ Bolt가 폐기 |

남은 건 "구조"와 "레시피"뿐이다. **"시계열의 언어를 배운다"는 제목이 실제로 입증한 것은 "언어 모델의 학습 방법론을 배운다"였다.**

---

## 🎓 무엇을 가져갈 것인가

### 남는 것

1. **사전학습 레시피는 전이되고, 사전학습 가중치는 전이되지 않는다.** 모달리티가 바뀌면 LLM 가중치는 도움이 안 된다는 걸 통제 실험으로 보여준 드문 사례(§5.6).
2. **합성 데이터가 제로샷 일반화를 산다.** TSMixup과 KernelSynth 둘 다 in-domain은 그대로 두고 zero-shot만 올렸다.
3. **자기회귀를 버리면 250배가 나온다.** 시퀀스 길이를 16분의 1로 줄이고 디코더를 1스텝으로 만드는 것만으로. 생성 모델을 예측 모델로 쓸 때 가장 먼저 의심할 지점.
4. **마스크로 과제를 통일할 수 있다.** 다변량/공변량/단변량을 구조 분기가 아니라 `group_ids` 하나로 처리한 Chronos-2의 설계는 다른 도메인에도 이식 가능한 발상이다.
5. **저장소가 자기 이전 세대의 결함을 코드 주석으로 남긴다.** `"which leads to uncertainty collapse"` 같은 문장은 논문 어디에도 없다 — **코드를 읽어야만 보이는 진실**의 좋은 예.

### 주의해서 읽을 것 (인용하기 전에)

- 스케일링 주장("클수록 좋다")은 **저장소의 공개 CSV로는 zero-shot에서 재현되지 않는다**(→ 5.1).
- "250배 빠름"은 재현 스크립트가 없다(→ 5.4).
- Bolt의 "5% 개선"은 **확률 예측 쪽 이득**이고, 점 예측은 2~3%다(→ 4.3).
- 논문 모델(Chronos-1)의 결과를 지금 인용하는 건 이미 낡았다. 실사용 권장 모델은 Bolt/Chronos-2이고, 그 둘은 논문이 없거나(Bolt) 별도 기술보고서(Chronos-2)다.

---

## 🧾 한 줄 요약 (전체)

**Chronos는 시계열을 정수 토큰으로 바꿔 T5에 먹이는 최소주의 프레임워크로 제로샷 예측의 실용성을 증명했지만, 정작 그 "언어 모델 흉내"의 세 요소(사전학습 가중치·양자화 토큰화·자기회귀 생성)를 후속 두 세대가 차례로 폐기했다 — 남은 것은 구조와 대규모 사전학습 레시피뿐이고, 그 위에 iTransformer가 정착시킨 변량축 어텐션이 `group_ids` 마스크로 흡수되면서 다변량·공변량 통합이 완성됐다.**

---

## 🔗 관련 문서 / 메모리

- [PAPER_iTransformer.md](PAPER_iTransformer.md) — 변량축 어텐션의 원류. **소규모 지도학습에서는 "시간축 어텐션이 해롭다"였던 결론이 대규모 사전학습에서 뒤집히는 대비**(→ Q1)
- [PAPER_TimesFM.md](PAPER_TimesFM.md) — 같은 시기 구글의 시계열 foundation model. **정확히 반대 노선**: Chronos는 LM 구조를 빌려 토큰화로 우겨넣었고(그리고 두 세대에 걸쳐 그걸 되물렸고), TimesFM은 처음부터 시계열 전용 패치 구조를 from-scratch로 설계. TimesFM 3.0의 variate attention 도입은 Chronos-2의 `GroupSelfAttention`과 같은 방향의 수렴
- [[reference_pretrained_backbone_reuse_landscape]] — Chronos-1의 "T5 구조만 빌리고 가중치는 무작위 초기화"는 백본 재사용 분기 중 **구조만 재사용** 사례. §5.6의 LLM 초기화 무용 결과가 이 분류의 근거
- 저장소 리뷰 대상 커밋: `8589d19` (2026-08-14), `chronos-forecasting` 2.3.1
