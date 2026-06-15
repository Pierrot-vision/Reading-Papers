# PAPER: Lumina-Image 2.0 — 텍스트와 이미지를 한 줄로 합친 통합 효율 생성 모델

## 0. 이 문서를 읽는 법

이 문서는 Lumina-Image 2.0 논문(arXiv 2503.21758)과 공개 코드를 처음 보는 사람이 흐름을 놓치지 않도록 정리한 리뷰입니다. 전작 [PAPER_Lumina-Next.md](PAPER_Lumina-Next.md) 와 후속 형제 모델 [PAPER_Z-Image.md](PAPER_Z-Image.md) 를 양옆에 두고, **"무엇이 바뀌었나 / 무엇이 같나"** 를 축으로 읽으면 가장 빠릅니다.

핵심 메시지 한 줄:

> **Lumina-Image 2.0 은 텍스트 토큰과 이미지 토큰을 한 시퀀스로 이어붙여(concat) 같은 트랜스포머가 통째로 처리하는(joint self-attention) "Unified Next-DiT" 다. 여기에 전용 캡셔너(UniCap)와 3단계 점진 학습, CFG 추론 가속을 더해, 2.6B 라는 작은 몸집으로 FLUX/SD3 급 품질에 도달했다.**

### 신뢰도 표기 규칙 (이 문서 한정)

논문 본문/그림을 직접 정독한 게 아니라 일부는 웹 요약에 의존했으므로, 출처별로 신뢰도를 구분해 표시합니다.

- 🟢 **코드 검증** — 공개 repo `models/model.py` 의 실제 config 로 확인한 사실.
- 🟡 **논문 요약 기반** — 논문 abstract/본문 요약에서 얻었으나 원문 표·그림으로 재확인이 권장되는 수치(특히 GPU-days, 벤치마크 점수).

---

## 1. 메타 정보

| 항목 | 내용 |
|---|---|
| 논문 제목 | Lumina-Image 2.0: A Unified and Efficient Image Generative Framework |
| 1저자 / 소속 | Qi Qin 외 22인 / Alpha-VLLM (Shanghai AI Lab) 계열 |
| 공개일 | 2025-03-27 (arXiv v1), 21 pages · 12 figures |
| 분야 | Text-to-Image / Diffusion Transformer / Flow Matching |
| 논문 링크 | https://arxiv.org/abs/2503.21758 |
| 공식 코드 | https://github.com/Alpha-VLLM/Lumina-Image-2.0 |
| 파라미터 | 2.6B 🟢 |
| 텍스트 인코더 | Gemma-2-2B 🟢 |
| VAE | FLUX-VAE (16채널) 🟢 |
| 선행 논문 | Lumina-Next (Next-DiT, [[paper_lumina_next]]) |
| 후속 형제 | Z-Image (S3-DiT, [[paper_z_image]]) — 같은 single-stream 계열 |

### 위치 짚기 (왜 이 논문이 중요한가)

*왜 이 절을 두나: 이 모델이 계보의 어느 매듭인지 먼저 박아두면, 이후 모든 변경점이 "왜 바뀌었는지"로 읽힌다.*

- **Lumina-Next → Lumina-Image 2.0 = 구조 혁신의 매듭.** 텍스트를 옆에서 주입(cross-attention)하던 방식을 버리고, 텍스트·이미지를 한 줄로 합쳐 통째로 보는(joint self-attention) **single-stream** 으로 전환한 지점.
- 이 single-stream 통합 설계를 **공개적으로 먼저 제시**(2025-03)했고, 8개월 뒤 [[paper_z_image]] 의 S3-DiT(2025-11)가 같은 패러다임을 더 큰 체급으로 잇는다.
- SD3 의 dual-stream MMDiT 와 비교하면, 2.0 은 텍스트·이미지가 **같은 가중치**를 공유하는 single-stream 이라 한 걸음 더 통합 쪽으로 나아간 설계.

---

## 2. 한눈에 보기 (TL;DR)

*왜 이 절을 두나: 바쁜 사람이 이 절만 읽고도 핵심 4개를 가져갈 수 있게.*

전작 [[paper_lumina_next]] 의 처방(Sandwich Norm, QK-Norm, Flow Matching)은 **그대로 토대로 남기고**, 네 가지를 새로 더하거나 갈아끼웠습니다.

1. **Unified Next-DiT (통합 아키텍처)** — 텍스트·이미지 토큰을 concat → joint self-attention. cross-attention 분리 폐기.
2. **Refiner 도입** — 백본에 넣기 전, 이미지 토큰은 `noise_refiner`, 텍스트 토큰은 `context_refiner` 로 각각 다듬어 모달리티 격차를 메움.
3. **UniCap (통합 캡셔너)** — 한 이미지에 길이·관점·언어가 다른 여러 캡션을 생성. "캡션 품질이 곧 모델 스케일링" 이라는 발견.
4. **추론 효율** — CFG-Truncation(20%+ 가속) + CFG-Renormalization(과포화 억제).

결과: **GenEval 0.73 / DPG 87.2** 🟡 로, 전작 Lumina-Next(0.46 / 75.66)를 크게 앞서고 SD3-medium·Sana 를 상회.

---

## 3. 핵심 기여 (Contributions)

*왜 이 절을 두나: 논문이 "우리가 새로 한 것"이라 주장하는 칸을, 네트워크 / 데이터 / 추론으로 나눠 한 번에 본다.*

| 분류 | 기여 | 한 줄 설명 |
|---|---|---|
| 네트워크 | **Unified Next-DiT** | 텍스트+이미지를 한 시퀀스로 joint self-attention. causal LLM 의 단방향 편향 완화, 정렬 향상, 확장 용이 |
| 네트워크 | **noise/context Refiner** | single-stream 합치기 전 모달리티별 2층 정리로 학습 안정화 |
| 네트워크 | **3D mRoPE** | 텍스트 길이 + 이미지 h + w 를 세 축으로 인코딩 (2D → 3D 확장) |
| 데이터 | **UniCap** | 다중 입자도·관점·언어 캡션. 캡션 길이를 "스케일링 수단"으로 활용 |
| 학습 | **3단계 점진 학습** | 256² → 1024² → 고품질 튜닝, 고해상 단계에 저주파 보존 보조 손실 |
| 추론 | **CFG-Trunc / CFG-Renorm** | 초반 CFG 생략으로 가속 + 큰 guidance 과포화 억제 |

---

## 4. 아키텍처 — Unified Next-DiT

*왜 이 절을 두나: 이 논문의 정체성이 여기 다 들어 있다. 전작과의 가장 큰 차이가 "텍스트를 어떻게 섞느냐" 한 가지이기 때문.*

### 4.0 전체 그림 한 장 (Figure 2)

*왜 이 절을 두나: 세부로 들어가기 전, 논문이 직접 그린 파이프라인 지도를 먼저 머리에 넣으면 4~8장이 이 그림의 부분 확대로 읽힌다.*

![Figure 2 — Overview of Lumina-Image 2.0](https://arxiv.org/html/2503.21758v1/x2.png)

> **캡션 원문:** "Overview of Lumina-Image 2.0, which consists of Unified Captioner and Unified Next-DiT. The Unified Captioner re-captions web-crawled and synthetic images to construct hierarchical text-image pairs, which are then used to optimize Unified Next-DiT with our efficient training strategy."

그림은 좌우 두 덩어리와 그 사이 데이터 흐름으로 읽으면 됩니다.

```
[웹 크롤 + 합성 이미지]
        │
        ▼
   ┌──────────────────────────┐
   │  Unified Captioner (UniCap)│   ← 왼쪽: 데이터를 만드는 쪽 (6장)
   │  한 이미지 → 긴/중간/짧은 캡션 │
   │         + 다관점 + 다국어     │
   └──────────────────────────┘
        │  계층적(hierarchical) image-text 쌍
        ▼
   다단계 점진 학습 (256² → 1024²)   ← 가운데: 학습 전략 (7장)
        │
        ▼
   ┌───────────────────────────────────────┐
   │  Unified Next-DiT                       │   ← 오른쪽: 모델 (4장)
   │  [텍스트 임베딩 + noisy latent] → concat │
   │  → joint self-attention + mRoPE         │
   │  → velocity 예측                         │
   └───────────────────────────────────────┘
```

- **왼쪽(UniCap):** 웹·합성 이미지를 다시 캡션해 길이·관점·언어가 다른 고품질 쌍을 만든다 → 6장.
- **가운데(데이터→학습):** 이 계층적 쌍이 다단계 점진 학습(256→1024)으로 흘러 모델을 최적화한다 → 7장.
- **오른쪽(Unified Next-DiT):** 텍스트 임베딩과 noisy latent 를 concat 해 joint self-attention + mRoPE 로 처리한다 → 4장.

한마디로 **"좋은 캡션으로 좋은 데이터를 만들어(왼쪽), 통합 DiT 를 효율적으로 학습한다(오른쪽)"** 는 논문 전체를 한 장에 압축한 지도다. "캡션 품질이 곧 수렴·생성 충실도를 좌우한다"는 핵심 주장이 이 그림의 메시지.

### 4.1 가장 큰 변화 — cross-attention 분리 → joint self-attention

전작 Next-DiT 는 한 레이어 안에서 **셀프 어텐션(이미지끼리)** 과 **크로스 어텐션(이미지→텍스트)** 을 *분리*해 두고, 텍스트 영향은 `gate.tanh()` 의 zero-init 으로 천천히 켰습니다 ([Lumina-Next.md 4.4절](PAPER_Lumina-Next.md)).

2.0 은 이 구조를 버립니다.

```
[ Lumina-Next ]  이미지 토큰 ──self-attn──┐
                                          ├─ 한 레이어, 분리
                 텍스트(Gemma) ─cross-attn─┘   (텍스트는 zero-init gate로 옆에서 주입)

[ Lumina-Image 2.0 ]  [텍스트 토큰][이미지 토큰]  ── concat ──> 같은 블록 joint self-attention
                          ↑ 둘이 대등하게 서로를 본다
```

**왜 좋아지나 (논문 논리):** 전작은 causal LLM(Gemma)에서 나온 텍스트 표현을 한 방향으로만 주입해 단방향 편향이 남았는데, joint self-attention 으로 묶으면 이 편향이 완화되고 텍스트-이미지 정렬(prompt adherence)이 좋아진다. DPG 가 75.66 → 87.20 으로 점프한 주된 이유로 제시됨.

### 4.2 모달리티 격차를 메우는 Refiner

텍스트와 이미지는 분포가 너무 달라서 그냥 concat 하면 충돌합니다. 그래서 백본에 넣기 **전에** 각 모달리티를 따로 다듬는 2층짜리 블록을 둡니다. 🟢

- **`noise_refiner`** — 이미지(노이즈 latent) 토큰용, `modulation=True` (timestep 조건 받음), 2층.
- **`context_refiner`** — 텍스트 토큰용, `modulation=False`, 2층.

> 이 noise/context refiner 는 [[paper_z_image]] 의 Noise Refiner / Context Refiner 와 **이름·역할·층수까지 동일**하다. single-stream 계열이 공유하는 안정화 장치다.

### 4.3 실제 코드 config 🟢

공개 repo `models/model.py` 의 팩토리 함수에서 직접 확인한 값입니다.

| 모델 | dim | layers | heads | kv | refiner | axes_dims (RoPE) | axes_lens |
|---|---|---|---|---|---|---|---|
| **NextDiT_2B_GQA_patch2_Adaln_Refiner** (배포본) | 2304 | 26 | 24 | 8 | 2 | [32, 32, 32] | [300, 512, 512] |
| NextDiT_3B_..Refiner | 2592 | 30 | 24 | 8 | 2 | [36, 36, 36] | [300, 512, 512] |
| NextDiT_4B_..Refiner | 2880 | 32 | 24 | 8 | 2 | [40, 40, 40] | [300, 512, 512] |
| NextDiT_7B_..Refiner | 3840 | 32 | 32 | 8 | 2 | [40, 40, 40] | [300, 512, 512] |

배포 모델(2.6B)의 파생 수치:
- **head_dim = 2304 / 24 = 96**, 이를 RoPE 세 축에 **32 / 32 / 32** 로 균등 분배.
- **axes_lens [300, 512, 512]** = 텍스트 토큰 최대 300, 이미지 높이 512, 너비 512 토큰까지.
- **patch = 2**, GQA(24 query head → 8 KV head).
- 토큰 concat 순서: `padded_full_embed[:cap_len] = 텍스트`, 이어서 `cap_len : cap_len+img_len = 이미지`.

### 4.4 3D mRoPE — 텍스트 축이 생긴 이유

*왜 이 절을 두나: 2D→3D 변화가 단순 버전업이 아니라 "통합의 직접 결과" 임을 짚기 위해.*

전작은 이미지만 시퀀스에 있어 2D RoPE(h, w)로 충분했습니다([Lumina-Next.md 3.2절](PAPER_Lumina-Next.md)). 2.0 은 한 시퀀스 안에 **텍스트까지** 들어오므로, 위치 인코딩에 **텍스트 길이 축이 추가**됩니다 → 3D multimodal-RoPE.

- 2.6B: 세 축에 **균등 분배 [32, 32, 32]**.
- 참고로 [[paper_z_image]] 는 [32, 48, 48] 로 **공간(h·w)에 더 많이** 배분 → Z-Image 가 공간 해상도 표현에 무게를 더 둔 설계.

### 4.5 계승된 것 (Next 와 동일)

Sandwich Norm, QK-Norm, RMSNorm(eps 1e-5), tanh-gated residual, Flow Matching(Linear path, velocity 예측), LLM 의 `hidden_states[-2]` 사용, patch=2 — [Lumina-Next.md 3.1·3.4·3.5절](PAPER_Lumina-Next.md)의 안정화 처방은 그대로 살아 있습니다. **바뀐 건 텍스트 결합 방식 / refiner / RoPE / VAE 네 가지뿐.**

---

## 5. forward 흐름 한 번

*왜 이 절을 두나: 토큰이 입력에서 velocity 까지 흐르는 순서를 한눈에 보면 4장의 부품들이 어디서 작동하는지 정리된다.*

```
1. prompt
     -> Gemma-2-2B
     -> hidden_states[-2]
     -> projection
     -> 텍스트 토큰

2. 이미지 latent (16ch, FLUX-VAE)
     -> patchify (p=2)
     -> 이미지 토큰
     -> noise_refiner 2층 (timestep 조건)

3. 텍스트 토큰
     -> context_refiner 2층

4. [텍스트 토큰][이미지 토큰]
     -> concat
     -> 3D mRoPE 부여
     -> Backbone 26층 joint self-attention (Sandwich Norm + QK-Norm + tanh-gate)

5. 출력에서 이미지 토큰만 선택
     -> final layer
     -> unpatchify
     -> velocity 예측
```

velocity = "지금 noisy latent 가 다음에 어느 방향으로 가야 하는가" (Flow Matching).

---

## 6. UniCap — "캡션 품질이 곧 모델 스케일링" 🟡

*왜 이 절을 두나: 전작이 "데이터 출처 불명"으로 비워뒀던 칸을, 2.0 은 데이터를 모델로 정제하는 정면 공략으로 채운다.*

- **베이스:** Qwen2-VL-7B 를 캡셔닝 전용으로 파인튜닝.
- **다중 입자도(multi-granularity):** GPT-4o 가 만든 아주 상세한 설명을 오픈소스 LLM 이 중간·짧은·태그형으로 요약. 길이가 다른 여러 버전 보유.
- **다중 관점(multi-perspective):** 스타일 / 주 객체 / 모든 객체의 속성 / 공간 관계를 따로 기술.
- **다국어(multi-lingual):** 영어·중국어 (Gemma 인코더 덕에 독·일·러는 zero-shot).
- **native 해상도 캡션:** 경쟁 시스템처럼 저해상도로 줄여 보지 않고 원본 크기로 봐서 디테일을 놓치지 않음.

**핵심 발견:** 캡션이 정밀·상세할수록 모델 수렴이 눈에 띄게 빨라진다 → **캡션 길이가 사실상 파라미터를 키우는 것과 같은 효과**. 작은 모델로 큰 모델을 이기는 비결의 절반.

> 같은 발상이 [[paper_z_image]] 의 Z-Captioner(tag/short/medium/long/simulated prompt 5종) 로 이어진다.

---

## 7. 3단계 점진 학습 🟡

*왜 이 절을 두나: 전작이 비워뒀던 학습 자원표가 여기서 채워진다. (단, 아래 GPU-days·step 수치는 원문 표로 재확인 권장)*

| 단계 | 해상도 | 데이터 | step | batch | A100-days |
|---|---|---|---|---|---|
| 저해상 사전학습 | 256² | 100M | 144K | 1024 | 191 |
| 고해상 | 1024² | 10M | 40K | 512 | 176 |
| 고품질(HQ) 튜닝 | 1024² | 1M | 15K | 512 | 224 |

- 학습률 전 단계 **2e-4** 고정.
- **고해상 단계 보조 손실(auxiliary loss):** latent 를 4배 다운샘플한 것으로 추가 손실 → "저주파(전역 구도)는 보존하면서 고주파(디테일)를 배움". [Lumina-Next.md 3.3절](PAPER_Lumina-Next.md)의 *저주파=구도 / 고주파=디테일* 철학이 손실 쪽으로 옮겨온 형태.
- **2종 system prompt** 를 캡션 앞에 붙여 합성(미적) / 실사 데이터를 구분 학습 → 추론 때 스타일 핸들.

전체 약 591 A100-days 수준 🟡 — 2.6B 치고 효율적.

---

## 8. 추론 가속 🟡

*왜 이 절을 두나: 전작의 sigmoid time-shift + midpoint 위에, 2.0 은 CFG 자체를 손보는 새 트릭을 얹는다.*

- **CFG-Renormalization (CFG-Renorm):** 큰 guidance scale 에서 생기는 **과포화(oversaturation)** 를, 수정된 velocity 크기를 conditional velocity 크기로 다시 맞춰 억제.
- **CFG-Truncation (CFG-Trunc):** denoising **초반(노이즈 많은 구간, t < 임계 α)에는 CFG 계산을 생략**하고 후반에만 적용 → **20%+ 가속**. "초반엔 대략적 구도라 텍스트 가이드가 덜 중요하다"는 직관. ([Lumina-Next.md 3.3절](PAPER_Lumina-Next.md)의 "t≈1 은 전역 레이아웃 단계" 관찰과 같은 결.)
- **솔버:** Midpoint / Euler / **DPM-Solver** 지원. Flow-DPM(14~20 NFE)은 빠르나 불안정, TeaCache 는 흐려짐 → **최종 파이프라인은 CFG-Renorm + CFG-Trunc 조합**.
- diffusers 기본값: `guidance_scale=4.0`, `num_inference_steps=50`.

> 주의: 이 논문에는 **증류(distillation)도 RLHF 도 없다.** [[paper_z_image]] 의 Decoupled DMD + 2-stage RLHF, [[paper_longcat_image]] 의 3단 RLHF 와는 다른 노선 — 2.0 은 "데이터(캡션) + 아키텍처"로 승부.

---

## 9. 벤치마크 🟡

*왜 이 절을 두나: "2.6B 가 정말 큰 모델급이냐"를 숫자로 확인. (점수는 원문 표 재확인 권장)*

| 모델 | 파라미터 | GenEval | DPG | 비고 |
|---|---|---|---|---|
| **Lumina-Image 2.0** | 2.6B | 0.73 | **87.20** | DPG 전 항목 최상위 |
| Lumina-Next | 1.7B | 0.46 | 75.66 | 전작 |
| SD3-medium | 2B | 0.62 | 84.08 | |
| Sana-1.6B ([[paper_sana_1_5]]) | 1.6B | 0.66 | 84.80 | |
| Janus-Pro-7B | 7B | 0.80 | — | 학술 1위지만 아레나는 낮음 |

- **DPG 87.20** (Entity 91.97 / Relation 94.85 / Attribute 90.20) — 전부 최고. 통합 아키텍처의 정렬 능력 지표.
- **GenEval 0.73** (Two-Object 0.87, Counting 0.67, Color 0.62).
- **T2I-CompBench** 0.7417 (Color 0.8211 최고).
- 사람 평가 아레나: Artificial Analysis ~982, Rapidata 969(Alignment 1031 로 FLUX Pro 다음 2위).
- AGI-Eval(중국어) 0.4545, 전작 Lumina-Next 0.3229 를 크게 상회.
- 논문이 짚는 메시지: **학술 벤치마크 1위(Janus-Pro)와 사용자 아레나 순위가 어긋난다** → GenEval 같은 지표만 좇으면 안 된다.

**한계 (논문 자인):** 인체 같은 복잡 구조, 희귀 개념, 군중 장면, 길고 복잡한 텍스트 렌더링에 약함.

---

## 10. Lumina-Next 와의 비교 (분기점)

*왜 이 절을 두나: 계보에서 "가장 크게 바뀐 매듭"이 Next → 2.0 사이임을 명확히.*

| 항목 | Lumina-Next | Lumina-Image 2.0 | 변화 의미 |
|---|---|---|---|
| 텍스트 결합 | cross-attn 분리 + zero-init gate | **concat → joint self-attn** | 단방향 편향 완화, 정렬↑, 확장 용이 |
| refiner | **없음** | noise 2 + context 2 | single-stream 안정화 |
| RoPE | 2D (h, w) | **3D mRoPE** [32,32,32] | 텍스트 축 추가 |
| VAE | 4채널 SD-VAE | **16채널 FLUX-VAE** | 재구성 품질↑ |
| 파라미터 | 1.7B | 2.6B | |
| layers / heads | 24 / 32 | 26 / 24 | |
| 텍스트 인코더 | Gemma-2B | Gemma-2-2B | |
| 계승 | — | Sandwich Norm·QK-Norm·tanh-gate·FM 그대로 | 안정화 처방 유지 |

핵심: **① cross-attn → joint self-attn, ② refiner 도입, ③ 2D→3D RoPE, ④ 4→16ch VAE.** 나머지는 전작 그대로.

---

## 11. Z-Image · FLUX.2 와의 비교 (4-모델 표)

*왜 이 절을 두나: 후속 형제 Z-Image 가 "구조가 같냐", 그리고 같은 시기 FLUX.2 와는 어떻게 다른가에 한 번에 답하기 위해.*

Lumina-Next 수치는 [Lumina-Next.md 4.7절](PAPER_Lumina-Next.md), Lumina-Image 2.0·Z-Image 는 실제 코드 기준 🟢, FLUX.2 는 공개 정보 기준 🟡 (dim/heads/층수/RoPE 축 등은 공식 비공개).

| 항목 | **Lumina-Next** | **Lumina-Image 2.0** | **Z-Image (S3-DiT)** | **FLUX.2 [dev]** |
|---|---|---|---|---|
| 공개 | 2024-06 | 2025-03 | 2025-11 | 2025-11-25 |
| 파라미터 | 1.7B | 2.6B | 6.15B | **32B** |
| **스트림 구조** | cross-attn 분리 | **pure single-stream** | **pure single-stream** | **hybrid: dual→single** |
| backbone 층 | 24 | 26 | 30 | 미공개 |
| **refiner** | **없음** | noise 2 + context 2 | noise 2 + context 2 | (double block 이 대체) |
| dim | 2304 | 2304 | 3840 | 미공개 |
| heads / kv | 32 / 8 | 24 / 8 | 32 / 8 | 미공개 |
| head_dim | 72 | 96 | 128 | 미공개 |
| **텍스트 결합** | cross-attn 분리 | concat → joint self-attn (가중치 공유) | concat → joint self-attn (가중치 공유) | dual(가중치 분리, **joint self-attn**)→single |
| **RoPE** | **2D** (h, w) | **3D** [32,32,32] 균등 | **3D** [32,48,48] 공간 가중 | RoPE (축 미공개) |
| patch | 2 | 2 | 2 | 미공개 |
| Norm | Sandwich + QK-Norm + tanh-gate | 동일 | 동일 | 미공개 |
| 텍스트 인코더 | Gemma-2B | Gemma-2-2B | Qwen3-4B | **Mistral-3 24B (VLM)** |
| 프롬프트 컨텍스트 | 짧음 | — | 512 토큰 | **~32K 토큰** |
| VAE | 4채널 SD-VAE | 16채널 FLUX-VAE | 16채널 Flux VAE | 16채널 신규 FLUX.2 VAE |
| 학습목표 | Flow Matching | 동일 | 동일 | Rectified Flow (= FM) |
| 편집(I2I) 통합 | ✗ | 확장 가능 | 사전학습 내장 | gen+edit 단일 아키텍처 |
| 증류 / few-step | 없음 | 없음 (CFG-Trunc/Renorm) | **Decoupled DMD → 8-step Turbo** | Turbo 변종 있음 |
| RLHF | 없음 | 없음 | **DMDR + DPO + GRPO** | 공개 정보 없음 |

### 계보 / 진영 한 줄

```
[pure single-stream 진영]                         [hybrid dual→single 진영]
Lumina-Image 2.0(2.6B) ─같은 골격─ Z-Image(6B)            FLUX.2 [dev](32B)
  concat→joint self-attn   refiner·3D RoPE 동일           DoubleStream→SingleStream
  Gemma-2-2B               Qwen3-4B                        Mistral-3 24B VLM (FLUX.1 계보)

Lumina-Next(1.7B): cross-attn 분리 — 위 두 진영 어디도 아닌 옛 방식
```

- **Next → 2.0 = 구조 혁신** (cross-attn → joint self-attn, refiner 도입, 2D→3D RoPE, 4→16ch VAE).
- **2.0 → Z-Image = 스케일·파이프라인 강화** (층·폭 키우고, RoPE 를 공간에 더 배분, 네트워크 밖에서 증류·RLHF 추가).
- **2.0/Z-Image ↔ FLUX.2 = 진영 자체가 다름.** 단, 차이는 *attention 종류*가 아니라 *가중치 공유 여부*다. **single-stream(2.0·Z)도 dual-stream/MM-DiT(FLUX.2)도 attention 은 둘 다 concat 후 joint self-attention** 이다. 차이는 텍스트·이미지가 **같은 가중치**를 쓰느냐(single) vs **모달리티별 분리 가중치**(own QKV/MLP/AdaLN)를 쓰느냐(dual). FLUX.2 는 앞쪽 dual-stream 블록(분리 가중치) → 뒤쪽 single-stream 블록(공유)으로 가는 FLUX.1 계보(= [Z-Image.md 6.2절](PAPER_Z-Image.md)의 dual-stream 진영). 텍스트 인코더도 2~4B LLM 이 아니라 24B VLM. ※ **진짜 cross-attention(이미지 Q → 텍스트 K/V 비대칭)을 쓰는 건 이 비교에서 Lumina-Next 뿐**이다.
- 결론: **블록 토폴로지(noise/context refiner → concat → joint backbone)는 2.0 과 Z-Image 가 동일**. 다른 건 체급·RoPE 배분·후처리뿐. FLUX.2 는 표제만 "통합 효율 모델"이고 구조 계열이 다르다.

---

## 12. 💬 Q&A

### Q1. "Unified" 가 정확히 뭐가 통합된 건가?
**텍스트 토큰과 이미지 토큰이 한 시퀀스로 합쳐져 같은 트랜스포머 블록이 처리**된다는 뜻. 전작은 둘을 cross-attention 으로 분리했는데, 2.0 은 concat 해서 joint self-attention 으로 본다. 덤으로 이 통합 덕에 나중에 편집(I2I) 같은 작업을 "이미지 토큰을 하나 더 끼워넣는" 식으로 확장하기 쉽다.

### Q2. Z-Image 와 네트워크가 같나?
**블록 골격은 같다(코드 검증).** 둘 다 noise_refiner + context_refiner → concat → joint backbone. 차이는 체급(2.6B vs 6B), RoPE 차원 배분([32,32,32] vs [32,48,48]), 그리고 증류·RLHF 유무(네트워크가 아니라 학습 파이프라인 차이). → 11장 표.

### Q3. 증류(Turbo) 버전이 있나?
**이 논문에는 없다.** few-step 증류와 RLHF 는 후속 [[paper_z_image]] 의 몫. 2.0 은 50-step + CFG 트릭(Trunc/Renorm) 노선.

### Q4. 왜 LLM 의 마지막 직전 layer(`hidden_states[-2]`)를 쓰나?
마지막 layer 는 다음 토큰 예측에 특화돼 noise 가 강하고, 한 단계 앞이 문맥 정보가 더 풍부하기 때문. 전작과 동일한 관례([Lumina-Next.md Q5](PAPER_Lumina-Next.md)).

### Q5. 이 문서 수치 중 어디까지 믿어도 되나?
🟢 표시(아키텍처 config: dim/layers/heads/refiner/RoPE 축)는 공개 코드로 검증한 확정값. 🟡 표시(GPU-days, 학습 step, 벤치마크 점수)는 논문 요약 기반이라 발표용으로 쓰기 전 **원문 표·그림으로 재확인 권장**.

### Q6. 2.0(2.6B)과 Z-Image(6.15B)는 백본 층이 26 vs 30 으로 4층밖에 차이 안 나는데 왜 파라미터가 2배 넘게 차이나나?
**파라미터는 "층 수"가 아니라 "층의 폭(dim)"이 좌우하기 때문.** 트랜스포머 한 층의 가중치는 거의 다 `(폭 × 폭)` 정사각 행렬(Q·K·V·O, FFN up·down)이라 **폭의 제곱에 비례**한다. 층을 하나 더 쌓는 건 같은 크기를 한 장 더 얹는 덧셈이지만, 폭을 키우는 건 제곱 곱셈이다.

```
파라미터 비 ≈ (폭 비)²  ×  (층 비)
            ≈ (3840/2304)²  ×  (30/26)
            ≈ 1.67²  ×  1.15  ≈  2.78 × 1.15  ≈  3.2배
```

실제 비율은 6.15B / 2.6B ≈ 2.4배. 계산한 3.2배보다 작은 건 Z-Image 가 GQA 로 attention 을 일부 아끼고 FFN 비율·refiner 구성이 달라서지만, **차이의 대부분은 층 수(+15%)가 아니라 폭(2304→3840, +67%, 제곱이라 블록당 ≈2.8배)에서 온다.** (이 비교는 DiT 백본만 센 것 — 텍스트 인코더·VAE 는 외부·동결이라 제외.)

### Q7. 같은 시기 FLUX.2 와는 구조가 같나?
**다른 계열이다 — 단 차이는 attention 종류가 아니라 가중치 공유 여부.** 2.0·Z-Image 는 텍스트·이미지가 **같은 가중치**를 쓰는 **pure single-stream**, FLUX.2 는 앞쪽이 **모달리티별 분리 가중치(own QKV/MLP/AdaLN)** 인 **dual-stream 블록** → 뒤쪽 single-stream 으로 가는 FLUX.1/MM-DiT 계보다. **둘 다 attention 자체는 concat 후 joint self-attention 이고 cross-attention 이 아니다** (진짜 cross-attn 은 Lumina-Next 뿐). 텍스트 인코더도 24B VLM(Mistral-3), 체급 32B 로 약 5~12배 크다. → 11장 표 참고.

---

## 13. 한 줄 요약 (전체)

> **Lumina-Image 2.0 = "Lumina-Next 의 안정화 처방(Sandwich/QK-Norm/FM)은 유지하되, 텍스트를 옆에서 주입하던 cross-attention 을 버리고 텍스트·이미지를 한 줄로 합쳐 통째로 보는 single-stream(Unified Next-DiT)으로 전환" + "refiner·3D RoPE·16ch VAE·UniCap·CFG 가속" 으로 2.6B 가 큰 모델급 정렬(DPG 87.2)을 달성한 모델.** 이 single-stream 통합 설계가 곧 후속 Z-Image(S3-DiT)의 직접 선례다.

---

## 14. 관련 메모리 링크

- [[paper_lumina_next]] — 직계 전작. cross-attn 분리·2D RoPE·4ch VAE 에서 2.0 이 무엇을 바꿨는지의 기준선
- [[paper_z_image]] — 같은 single-stream 계열 후속 형제(S3-DiT). refiner·3D RoPE 동일, 체급·증류·RLHF 만 추가
- [[paper_dreamlite]] — 또 다른 효율형 통합 모델(온디바이스). 비교 관점
- [[paper_longcat_image]] — Flux 계열 dual+single 혼합 + 3단 RLHF. 통합 방식이 다른 대조군
- [[paper_min_snr]] — Flow Matching velocity 예측이 ε 가중치 문제를 자동 해소하는 배경
- [[reference_pretrained_backbone_reuse_landscape]] — Gemma / Qwen 텍스트 인코더 재사용 패러다임 분류
