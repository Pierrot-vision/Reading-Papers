# Gemma 4 Technical Report

## 📋 메타 정보

| 항목 | 내용 |
|---|---|
| **제목** | Gemma 4 Technical Report |
| **저자** | Gemma Team (Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem 외 295명) |
| **소속** | Google DeepMind |
| **공개일** | 2026-07-02 |
| **분야** | open-weight(공개 가중치) natively multimodal LLM — 텍스트 + 이미지 + 오디오 |
| **arXiv** | [abs](https://arxiv.org/abs/2607.02770) · [html](https://arxiv.org/html/2607.02770v1) |
| **라이선스** | **Apache 2.0** (Gemma 1~3의 커스텀 "Gemma Terms of Use"에서 전환) |
| **분류** | cs.CL, cs.AI |
| **선행 논문** | Gemma 3 / Gemma 3n (per-layer embeddings 계승) |
| **핵심 인용** | Barbero et al., ICLR 2025 — "Round and round we go! What makes rotary positional encodings useful?" (pp-RoPE의 출처) |

---

## 📖 주요 용어 사전 (Glossary)

### 아키텍처
- **dense(밀집) 모델**: 모든 파라미터가 매 토큰마다 전부 계산에 참여하는 표준 Transformer. Gemma 4의 E2B·E4B·12B·31B가 여기 해당.
- **MoE (Mixture-of-Experts, 전문가 혼합)**: 여러 개의 FFN "전문가" 중 일부만 골라 쓰는 구조. 총 파라미터는 크지만 실제 계산량(활성 파라미터)은 작음. Gemma 4의 26B-A4B가 유일한 MoE (26B 총 / 3.8B 활성).
- **PLE (per-layer embeddings, 레이어별 임베딩)**: Gemma 3n에서 가져온 기법. 임베딩 테이블을 레이어마다 따로 두되 실제 연산(matmul)에는 참여시키지 않아, **총 파라미터는 크지만 유효(effective) 파라미터는 작게** 만듦. E2B가 총 5B인데 유효 2.3B인 이유.
- **GQA / local-global attention(로컬-글로벌 어텐션)**: 대부분의 레이어는 가까운 토큰만 보는 sliding window(슬라이딩 윈도우) local attention으로, 소수 레이어만 전체를 보는 global attention으로 두는 설계. Gemma 4는 로컬:글로벌 = **4:1 (E2B)** / **5:1 (나머지 전부)**.
- **KV cache(키-값 캐시)**: autoregressive(자기회귀) 생성 시 이미 계산한 key·value 텐서를 저장해 두는 메모리. 컨텍스트가 길어질수록 이게 병목이 됨. 이 논문 효율 개선의 주 타깃.
- **encoder-free(인코더 없는)**: 이미지·오디오를 전용 인코더(ViT, Conformer)로 먼저 압축하지 않고, raw 패치를 곧장 LLM 임베딩 공간에 투영하는 설계. Gemma 4 **12B 전용**.
- **drafter head / MTP (multi-token prediction, 다중 토큰 예측)**: 본체 모델 옆에 붙인 작은 보조 모델. 다음 토큰 여러 개를 미리 "초안"으로 뽑아 본체가 한 번에 검증하게 함 → speculative decoding(추측 디코딩)으로 생성 속도 가속.

### 핵심 개념
- **RoPE (rotary positional embedding, 회전 위치 임베딩)**: 토큰의 위치 정보를, query·key 벡터를 위치에 비례한 각도만큼 "회전"시켜 넣는 방식. 차원 쌍마다 회전 주파수가 다름.
- **partial RoPE / pp-RoPE (부분 회전 RoPE)**: head 차원의 **일부만 회전**시키고 나머지는 회전 없이(NoPE) 두는 변형. Gemma 4는 global attention 레이어에 `p=0.25` (= 25%만 회전) 적용. 출처는 Barbero et al. 2025 — RoPE의 저주파 회전 성분은 위치 정보를 거의 나르지 않으면서 의미 채널만 갉아먹는다는 관찰.
- **values=keys (키를 값으로 재사용)**: global attention 레이어에서 value projection(값 투영 행렬)을 아예 두지 않고, key 텐서를 그대로 value로 사용. 원문 표현 그대로 `values=keys`.
- **thinking mode(사고 모드)**: 답변을 내놓기 전에 추론 과정(reasoning trace)을 먼저 출력하는 모드. 시스템 턴의 제어 토큰으로 켬.

### 평가
- **RULER**: 긴 컨텍스트에서 정보를 찾아내는 능력을 재는 벤치마크 (needle-in-a-haystack 계열의 확장판).
- **LOFT**: 긴 문맥 안에서 검색(retrieval) 성능을 재는 벤치마크. Recall@k로 측정.
- **Codeforces Elo**: 경쟁 프로그래밍 문제 풀이 실력을 Elo 점수로 환산한 지표.
- **LMArena Elo**: 사람이 두 모델의 답변을 블라인드 비교해 매기는 순위 점수.

---

## 🎯 논문 요약 (TL;DR)

**한 줄**: Gemma 4는 새 학습 알고리즘 논문이 아니라 **"메모리·추론 효율 설계 모음집"**이다. 성능 점프의 근거를 학습법이 아니라 세 가지 구조 선택 — (1) KV cache를 37.5% 줄이는 attention 설계, (2) 12B에서 vision/audio encoder를 통째로 없앤 encoder-free 구조, (3) thinking mode 추가 — 에 두고 있다.

**핵심 문제**: 공개 가중치 모델은 프론티어 모델과 성능으로 겨루면서도 **모바일·단일 GPU에서 돌아가야 한다**. 이때 실질적 병목은 파라미터 수가 아니라 긴 컨텍스트에서 폭증하는 KV cache 메모리, 그리고 멀티모달 인코더가 차지하는 별도 메모리 조각(memory fragmentation)이다.

**해결책**: 파라미터를 줄이는 대신 **캐시해야 할 텐서 자체를 줄인다**. global attention에서 value를 없애고(`values=keys`), RoPE를 25%만 걸어 회전 안 한 차원은 key와 value가 동일해지게 만든다 → global KV cache 37.5% 절감(§3.1). 멀티모달은 12B에서 550M ViT를 **35M 행렬곱 하나**로 대체(§3.2).

**검증**: LMArena 텍스트에서 31B가 Elo 1451(43위)로 "10배 큰 모델과 겨룬다"고 주장. thinking mode 기준 AIME 2026 89.2, Codeforces Elo 2150, MMLU-Pro 85.2. 128k 컨텍스트에서 RULER 96.4.

**⚠️ 다만**: 위 세 기법 중 **어느 것도 통제된 ablation(제거 실험)으로 격리 검증되지 않았다.** 학습 관련 정보(사전학습 토큰 수, distillation 여부, RL 알고리즘, MoE 라우팅 설계)는 리포트에 **아예 없다**. 자세한 검증 결과는 §6.

---

## 🏆 핵심 기여 (Contributions)

1. **pp-RoPE + values=keys 조합으로 global KV cache 37.5% 절감** — 도메인 무관하게 이식 가능한, 이 리포트에서 가장 재사용 가치 높은 아이디어.
2. **12B의 encoder-free 멀티모달** — 550M vision encoder를 35M matmul로, Conformer audio encoder를 직접 투영으로 대체.
3. **thinking mode 도입** — Gemma 3 대비 가장 큰 기능적 차이.
4. **모바일 겨냥 양자화 + MTP drafter head** — E2B를 0.8GB까지 압축, 262k vocab projection을 top-k 클러스터로 축소.
5. **Apache 2.0 라이선스로 전환** — 벤치마크 숫자보다 실무적으로 큰 뉴스.

---

## 🧱 모델 라인업

*모델마다 "어디에 배포할 것인가"가 다르고, 그에 따라 인코더 유무와 파라미터 배치가 갈리므로 먼저 전체 지형부터 본다.*

| 모델 | 형태 | 파라미터 | Vision Encoder | Audio Encoder | Drafter |
|---|---|---|---|---|---|
| **E2B** | Dense | 2.3B 유효 / 5B 총 | 150M ViT | 305M Conformer | 76M |
| **E4B** | Dense | 4.5B 유효 / 8B 총 | 150M ViT | 305M Conformer | 77M |
| **12B** | Dense | 12B | **없음 (35M matmul)** | **없음 (직접 투영)** | 400M |
| **26B-A4B** | MoE | 26B 총 / 3.8B 활성 | 550M ViT | — | 430M |
| **31B** | Dense | 31B | 550M ViT | — | 500M |

- 어휘(vocabulary)는 전 모델 **262k** 공유. SentencePiece tokenizer (split digits, preserved whitespace, byte-level encodings).
- E2B/E4B의 "유효 vs 총" 격차는 **PLE(per-layer embeddings)** 때문 — 5B 중 2.3B만 실제 연산에 참여하고, 나머지 임베딩 테이블은 메모리에만 얹혀 있음. 모바일에서 유리한 배치.
- 정규화는 RMSNorm 기반 **pre-norm + post-norm** 병용, 안정성을 위해 **QK-Norm** 사용.
- 오디오는 **E2B·E4B·12B에만** 존재 (26B-A4B·31B는 텍스트+비전).

### Table 1 (파라미터 분해, 리포트 원표)

| Model | Audio Enc | Vision Enc | Embedder | Einsums | Drafter |
|---|---|---|---|---|---|
| E2B | 305M | 150M | 400M + 2,340M | 1,870M | 76M |
| E4B | 305M | 150M | 670M + 2,820M | 3,940M | 77M |
| 12B | — | — | 1,000M | 10,890M | 400M |
| 26B-A4B | — | 550M | 740M | 24,500M / 2,800M (active) | 430M |
| 31B | — | 550M | 1,410M | 29,290M | 500M |

> E2B/E4B의 두 번째 embedder 숫자(2,340M / 2,820M)가 per-layer embeddings.
> ⚠️ 26B-A4B의 활성 파라미터가 본문에선 "3.8B activated", Table 1 einsums에선 "2,800M active"로 **불일치**한다. 전자는 embedder 포함, 후자는 einsum만 센 것으로 보이지만 리포트가 명시하지 않는다.

### Vision Encoder 스펙 (Table 10)

| 크기 | 적용 모델 | dim | MLP | heads | layers | patch |
|---|---|---|---|---|---|---|
| 550M | 26B-A4B, 31B | 1152 | 4304 | 16 | 27 | 16 |
| 150M | E2B, E4B | 768 | 3072 | 12 | 16 | 16 |

- 위치 정보: **axial 2D-RoPE + 2D absolute positional embeddings** 병용.
- 이미지 토큰 수는 **70 / 140 / 280 / 560 / 1120** 중에서 선택 (종횡비 보존 리사이즈 + 풀링).
- Audio encoder(305M, E2B/E4B): downsampling conv 2층 + **Conformer 12층**, 40ms 청크 Mel filterbank 입력. **사전학습 동안 동결(frozen)**. Gemma 3n의 680M에서 55% 감축.

---

## 🔬 주요 알고리즘 설명

### 3.1 KV cache 37.5% 절감 — 이 리포트에서 제일 예쁜 부분

*긴 컨텍스트에서 진짜 메모리를 잡아먹는 건 파라미터가 아니라 KV cache다. 그래서 "파라미터를 줄이자"가 아니라 "캐시해야 할 텐서 자체를 없애자"로 접근한다.*

리포트는 이걸 한 문단에 툭 던지고 지나가는데, 산수를 맞춰보면 설계 의도가 아주 깔끔하게 드러난다. 세 가지가 겹쳐 있다.

**리포트 원문 (Long-context efficiency 절)**:
> "Our local to global attention ratio patterns follow Gemma Team [2025a], that is, 4-to-1 local attention blocks for E2B and 5-to-1 for the rest. We improve memory efficiency by re-using keys as values in the global attention layers (except in E2B and E4B), i.e., values=keys. We encode position with pp-RoPE with p=0.25 on global attention layers and with RoPE on local attention layers, effectively reducing the global KV cache by 37.5%. The RoPE frequencies are set to 1M and 10k on global and local attention layers, respectively. Finally, we share the KV cache with ratios of 20/35 and 18/42 for the E2B and E4B model."

#### ① value를 아예 없앴다 (`values=keys`)

global attention 레이어에서 value projection(값 투영 행렬)을 두지 않고, key 텐서를 그대로 value로 쓴다. E2B/E4B는 제외 → 즉 **12B, 26B-A4B, 31B에 적용**.

#### ② pp-RoPE with p=0.25

인용은 **Barbero et al., ICLR 2025 "Round and round we go! What makes rotary positional encodings useful?"**. 그 논문의 요지는 RoPE의 **저주파 회전 성분이 실은 위치 정보를 거의 안 나르고 오히려 의미 채널을 갉아먹는다**는 것이고, 처방이 **head 차원의 일부만 회전시키고 나머지는 회전 없이(NoPE) 두는 partial RoPE**다. `p=0.25`면 25%만 회전.

#### ③ 왜 정확히 37.5%인가 — 산수 재구성

리포트는 산수를 안 써놨지만 딱 맞아떨어진다.

기존엔 head 하나당 **key(차원 d) + value(차원 d) = 2d**를 캐시해야 한다.

그런데:
- value가 key와 같아졌고 (`values=keys`),
- key의 **75%는 회전조차 안 되니** value와 완전히 동일한 값이다.

그러면 저장할 것은:
```
회전 전 key 벡터 전체        →  d      (이게 곧 value)
회전된 25% 부분만 추가로      →  0.25d
──────────────────────────────────────
총                            1.25d
```

`1.25d / 2d = 0.625` → **정확히 37.5% 절감**.

> ⚠️ 이 유도는 **본 문서 작성 시 맞춘 재구성**이고 리포트에 명시되어 있지 않다. 다만 수치가 우연이라기엔 너무 정확하다. (회전된 25%를 저장하지 않고 매번 재계산하면 d만 저장 → 50% 절감이 되지만, 저자들은 저장하는 쪽을 택한 것으로 보인다.)

#### ④ 나머지 장치

- RoPE 주파수를 **global 1M / local 10k**로 갈라놨다.
- E2B/E4B는 추가로 레이어 간 **KV cache 공유** (비율 20/35, 18/42).
  ⚠️ 이 비율이 레이어 인덱스인지 head 인덱스인지 원문이 말해주지 않는다.

---

### 3.2 12B의 encoder 제거 (encoder-free)

*전용 인코더는 별도 메모리 조각(memory fragmentation)을 만들고 배포를 복잡하게 한다. 백본이 충분히 크다면 인코더가 하던 일을 직접 배울 수 있지 않을까 — 이게 12B의 도박이다.*

#### Vision

**550M vision encoder를 35M짜리 행렬곱(matmul) 하나로 대체**한다.

- 입력: **48×48×3 RGB 패치**
- 처리: 단일 large matmul (35M 파라미터)로 LLM 임베딩 공간에 직접 투영
- 공간 정보: **2D coordinate-based positional embeddings**를 패치 표현에 직접 더해서 챙김

원문:
> "Gemma 4 12B takes in 48×48×3 RGB patches, but replaces the 550M vision encoder by a single large matmul (35M parameters)."
> "Spatial awareness is maintained by adding 2D coordinate-based positional embeddings directly to the patch representations."

#### Audio

같은 철학. 16kHz raw 오디오를 **40ms 청크**로 잘라 **640차원 벡터**로 만든 뒤, **별도 위치 인코딩 없이** LLM 임베딩 공간에 직접 투영한다.

> "Raw audio is segmented into 40ms chunks at 16kHz, resulting in 640-dimensional vectors per chunk. These are projected directly into the LLM embedding space."

효과로 "별도 인코더가 필요 없어지고 memory fragmentation이 줄어든다"고 서술.

#### 이 흐름의 위치

이건 [[paper_tuna_2]] (raw RGB를 `Conv2d` 한 줄로 Qwen2.5-7B에 직접 주입), [[paper_sensenova_u1]] (2층 conv raw RGB 주입), [[paper_hidream_o1_image]] (VAE·별도 텍스트 인코더 없이 VLM backbone)와 **정확히 같은 가설** 위에 서 있다:

> *스케일이 충분하면 전용 인코더는 백본이 흡수할 수 있는 귀납 편향(inductive bias)에 불과하다.*

TUNA-2가 "scale에서 인코더 역전(crossover)"을 주장했는데, Gemma 4가 12B에서 이걸 **프로덕션 모델로 실행**했다는 게 의미가 있다.

#### ⚠️ 결정적인 게 빠졌다

리포트에 **인코더 있는 12B vs 없는 12B 통제 비교표가 없다.** 12B의 비전 성능 숫자(1120 토큰 기준 MMMU-Pro 69.1, MATH-Vision 79.7)는 있지만 대조군이 없으니, "인코더를 빼도 괜찮다"는 주장이 **논문 내부 증거로 지지되지 않는다**.

게다가 인코더를 뺀 게 하필 **12B 하나뿐**이고 26B-A4B/31B는 550M ViT를 그대로 유지한다. 스케일이 커질수록 인코더를 빼야 유리하다는 crossover 가설과는 **오히려 반대 방향의 배치**라, 왜 12B에서만 했는지 설명이 필요한데 없다.

---

### 3.3 Thinking mode

*수학·코딩처럼 중간 단계가 필요한 문제에서, 답을 바로 뱉게 하지 말고 생각을 먼저 적게 하면 정확도가 오른다.*

- 답변 전에 reasoning trace(추론 흔적)를 먼저 출력하는 모드.
- 시스템 턴(leading system turn)의 제어 토큰 `<|think|>`으로 토글, 출력 형식은 `<|channel>thought ...<channel|>` (Table 11).
- 원문: "By outputting a reasoning trace before the response, models demonstrate improved capabilities in reasoning-heavy domains such as mathematics and coding."
- Gemma 3 대비 "가장 큰 차이(A significant difference)"로 명시.

**⚠️ 그런데 이걸 어떻게 학습시켰는지가 리포트에 없다.** thinking budget 파라미터도, RLVR 같은 추론 전용 RL도 언급이 없다. 그냥 "Gemma 3와 유사한 post-training"이라고만 하고 넘어간다.

---

### 3.4 MTP drafter head (§2.6)

*생성 속도를 올리려면 본체를 매 토큰 한 번씩 돌리는 대신, 작은 모델이 여러 토큰을 미리 찍고 본체가 한 번에 검증하면 된다.*

MTP 쪽은 그래도 구체적이다.

- 구조: **별도 embedder + 4층 Transformer 블록**이 본체 모델의 KV에 **cross-attend** 한다.
  - 블록 구성: local attention 3층 + global attention 1층
  - 모델 차원: **256** (E2B/E4B), **1024** (26B-A4B/31B)
- E2B/E4B 최적화: 262k 전체 vocab projection을 **top-k 클러스터 연산**으로 대체
  → 최종 행렬곱이 `d×262,000`에서 `d×4,096`으로 축소, "preserving a similar acceptance rate".

원문:
> "The MTP head generates future tokens sequentially using a separate embedder and a 4-layer Transformer block that cross-attends to the KVs of the main model."

**⚠️ 구체적인 acceptance rate(%)도, speedup/latency 수치도 리포트에 없다.**

---

## 📊 실험 요약

### 인간 선호 평가 (Table 4)

*"벤치마크 말고 실제 사람이 어느 답을 고르는가"를 보는 지표.*

| 모델 | LMArena Text Elo | 순위 |
|---|---|---|
| Gemma 4 31B | 1451 ± 8 | 43위 |
| Gemma 4 26B-A4B | 1438 ± 8 | — |

"10배 큰 모델과 겨루며, dense 오픈소스 카테고리에서 선두"라고 주장.

### 추론 벤치마크 (Table 5, thinking mode)

| 벤치마크 | 31B | E2B |
|---|---|---|
| MMLU-Pro | 85.2 | 60.0 |
| AIME 2026 | 89.2 | 37.5 |
| Codeforces Elo | 2150 | 633 |

### 멀티모달 비전 (Table 6, 최대 해상도)

| 벤치마크 | 31B | 12B (encoder-free, 1120 토큰) | E2B |
|---|---|---|---|
| MMMU-Pro | 76.9 | 69.1 | 44.2 |
| MATH-Vision | 85.6 | 79.7 | 52.4 |

### 오디오 (Table 7)

| 벤치마크 | Gemma 4 E2B | Gemma 3n 대비 |
|---|---|---|
| CoVoST 번역 (CorpusBLEU 평균) | 35.4 | +12% |
| FLEURS ASR (WER 평균) | 0.090 | −17% (개선) |

### 롱컨텍스트 (Table 9, 128k)

| 벤치마크 | 31B | E4B |
|---|---|---|
| RULER Accuracy | 96.4 | 89.8 |
| LOFT Text Retrieval Recall@k | 79.5 | 66.3 |

- E4B가 **32k RULER에서 Gemma 3 27B를 추월** (96.4 vs 91.1).
- ⚠️ 모델별 **최대 지원 컨텍스트 길이 선언이 리포트에 없다.** 128k까지 평가만 있고, Table 9에 "~128k (Half book)", "~256k (Full book)" 항목이 보인다.

### 메모리 & 양자화 (Table 3, 32k 컨텍스트, 텍스트 전용)

*모바일 배포를 진심으로 겨냥한 숫자들.*

| 모델 | bf16 | 양자화 후 |
|---|---|---|
| E2B | 4.6 GB | **0.8 GB** (int2/int4 혼합 weight + int8 activation) |
| 12B | 24.0 GB | 7.65 GB (Q4_0) |
| 31B | 64.0 GB | 19.2 GB |

- KV cache 추가 오버헤드는 0.05~1.10 GB로 미미.
- Audio encoder: 디스크 기준 390MB → 87MB (**78% 감축**, 양자화 후).
- Vision encoder: quantization-aware training으로 **latency 44% 감축**.

---

## 🕳️ 이 리포트의 빈칸 — 솔직히 말해서

*논문을 읽고 "무엇을 알 수 있는가"만큼 "무엇을 알 수 없는가"도 기록해 둬야 나중에 재현을 시도할 때 헛수고를 안 한다.*

읽으면서 반복적으로 걸리는 게, **학습에 관해 사실상 아무것도 말하지 않는다**는 점이다. 원문 문자열을 직접 검색해서 확인한 결과:

| 항목 | 확인 결과 |
|---|---|
| `distill`, `teacher` | **한 번도 안 나옴.** Gemma 2·3의 정체성이 대형 교사로부터의 **knowledge distillation(지식 증류)**이었는데, Gemma 4는 이 얘기를 아예 하지 않는다. 안 썼다는 건지 말 안 하겠다는 건지 알 수 없다. |
| 사전학습 토큰 수 | **없음.** "trillion"이라는 단어조차 안 나온다. 데이터 설명은 "웹/코드/이미지/오디오, 컷오프 2025년 1월"이 전부. |
| MoE 26B-A4B 설계 | **expert 개수, top-k, shared expert, load balancing loss, router 구조 전부 없음.** `router`라는 단어가 본문에 등장하지 않는다. **MoE 모델을 내놓으면서 MoE 설계를 안 썼다.** |
| RL 알고리즘 | RLHF/BOND/WARM/WARP/RLVR — **하나도 없음.** reward model 언급도 없음. "similar post-training approach as in Gemma 3"라는 포인터만 존재. |
| 통제된 ablation 표 | **없음.** pp-RoPE, key-as-value, encoder 제거 어느 것도 격리 검증되지 않음. |
| 최대 컨텍스트 길이 | **선언 없음.** 128k까지 평가만 존재. |
| MTP acceptance rate / speedup | **없음.** |
| 언어 백본 하이퍼파라미터 표 (layers/dim/heads/FFN) | **없음.** Table 1은 파라미터 총량만, Table 10은 vision encoder만. |

확인된 post-training 정보는 데이터 필터링뿐이다:
> "We filter examples that show certain personal information, unsafe or toxic model outputs, mistaken self-identification data, and duplicated examples."
> + in-context attribution / hedging / refusals 데이터가 factuality를 개선했다는 서술.

**즉 이건 논문이 아니라 릴리즈 노트에 가깝다.** 재현 가능한 건 아키텍처뿐이고, 성능을 만든 요인은 공개되지 않았다. §3의 세 기법이 실제로 얼마나 기여했는지는 이 문서로는 알 수 없고, **가중치가 Apache 2.0으로 풀렸으니 검증은 커뮤니티 몫으로 넘어갔다.**

### 안전성 / 배포

- "Gemma 4 undergoes the same rigorous safety evaluations as Gemini models."
- CSAM / dangerous content / hate speech / harassment 정책. train-time mitigation + post-training eval.
- "major improvements in every category", "minimal policy violations".
- 라이선스: **Apache 2.0** — "empowering developers and researchers everywhere to build upon, customize, and extend these capabilities." 배포 플랫폼(HuggingFace/Kaggle/Vertex)은 리포트에 명시 없음.

---

## 💬 Q&A

### Q1. 왜 하필 37.5%인가? 산수가 맞는가?

맞는다. §3.1 ③의 유도 참조 — `values=keys`로 value를 없애고, pp-RoPE(p=0.25)로 75%의 차원이 회전조차 안 하니 그 부분은 key와 value가 동일한 값이 된다. 회전 전 key 전체(d) + 회전된 25%(0.25d) = 1.25d를 저장하면 되고, 원래 2d 대비 정확히 37.5% 절감이다.

단, 이 유도는 리포트에 명시되지 않은 **재구성**이다. 리포트는 결과 수치만 던진다.

### Q2. Gemma 4에서 이미지 생성 연구자가 가져갈 만한 게 있나?

**`pp-RoPE + values=keys` 조합**이다. 이건 LLM 전용 트릭이 아니라, **긴 시퀀스를 다루는 DiT 계열에도 그대로 이식 가능**하다.

특히 텍스트+이미지 토큰을 concat해서 joint self-attention을 돌리는 **single-stream 구조** — [[paper_lumina_image_2]] 의 Unified Next-DiT, [[paper_z_image]] 의 S3-DiT 계보 — 에서 시퀀스가 길어질 때 KV 메모리가 병목인데, *"회전 안 하는 차원은 key와 value가 같은 값이니 한 번만 저장한다"* 는 관찰은 **도메인 무관**하다.

그리고 이 아이디어의 진짜 출처는 Gemma 4가 아니라 **Barbero et al., ICLR 2025**이므로, 파고들 거라면 그쪽을 봐야 한다.

### Q3. encoder-free 12B는 TUNA-2의 crossover 가설을 입증하는가?

**아니다. 오히려 증거로 삼기 어렵다.**

두 가지 이유:

1. **대조군이 없다.** 인코더 있는 12B와의 통제 비교표가 리포트에 없다. TUNA-2는 백본(Qwen2.5-7B)을 고정한 채 비전 입력단만 3가지(VAE+SigLIP / SigLIP만 / Conv2d만)로 바꾼 3-way ablation을 돌렸다. Gemma 4는 그걸 안 했다.

2. **배치 방향이 반대다.** crossover 가설은 "스케일이 커질수록 encoder-free가 유리해진다"인데, Gemma 4는 **가장 큰 26B-A4B·31B에 550M ViT를 유지**하고 중간 크기 12B에서만 인코더를 뺐다. 가설대로면 31B에서 뺐어야 한다.

가장 그럴듯한 해석은 **12B가 "단일 GPU 배포 타깃"이라 memory fragmentation 제거의 실익이 가장 컸다**는 엔지니어링 판단이지, 표현학습 우위 주장은 아니라는 것이다. 다만 이것도 리포트가 말해주지 않는다.

### Q4. Gemma 3n과 뭐가 다른가?

| 항목 | Gemma 3n | Gemma 4 |
|---|---|---|
| PLE (per-layer embeddings) | 도입 | **계승** (E2B/E4B) |
| MatFormer / nested / elastic | 사용 | **언급 없음** (리포트에서 확인 불가) |
| Audio encoder | 680M | **305M** (55% 감축) |
| Thinking mode | 없음 | **있음** |
| MoE | 없음 | **26B-A4B 추가** |
| 라이선스 | Gemma Terms | **Apache 2.0** |

### Q5. 26B-A4B는 활성 파라미터가 3.8B인가 2.8B인가?

**리포트가 스스로 모순된다.** 본문은 "3.8B activated", Table 1의 einsums 열은 "2,800M (active)"라고 쓴다. 차이 1B는 embedder(740M) + drafter(430M) 등을 포함했는지 여부로 설명될 여지가 있으나, 리포트가 명시하지 않는다. 인용할 때 주의.

---

## 🎬 한 줄 요약 (전체)

> **Gemma 4는 "무엇을 학습시켰는가"는 침묵하고 "무엇을 캐시하지 않을 것인가"만 말하는 리포트다.** `values=keys` + `pp-RoPE(p=0.25)`로 global KV cache를 정확히 37.5% 덜어내고, 12B에서는 550M vision encoder를 35M 행렬곱으로 갈아치웠다. 세 개의 구조 기법 중 어느 것도 ablation으로 검증되지 않았고 distillation·토큰 수·MoE 라우팅·RL은 한 글자도 없지만, **가중치가 Apache 2.0으로 풀렸으므로 검증은 이제 커뮤니티의 몫이다.**

---

## 🔗 관련 메모리 / 문서

- [[paper_tuna_2]] — encoder-free의 통제된 3-way ablation. Gemma 4에 없는 바로 그 실험.
- [[paper_sensenova_u1]] — 2층 conv로 raw RGB 직접 주입, native 통합 멀티모달.
- [[paper_hidream_o1_image]] — VAE·별도 텍스트 인코더 없이 VLM backbone 재사용.
- [[reference_pretrained_backbone_reuse_landscape]] — 백본 재사용 패러다임 분류.
- [[paper_lumina_image_2]], [[paper_z_image]] — single-stream joint self-attention 계보. §3.1의 KV 절감 기법 이식 대상.
- [[paper_nucleus_image]] — MoE diffusion. Gemma 4가 침묵한 routing 설계를 정면으로 다룬 논문.
