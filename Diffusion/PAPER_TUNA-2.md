# TUNA-2: Pixel Embeddings Beat Vision Encoders for Unified Understanding and Generation

## 📋 메타 정보

| 항목 | 내용 |
|---|---|
| **제목** | Tuna-2: Pixel Embeddings Beat Vision Encoders for Unified Understanding and Generation |
| **소속** | Meta + HKU(홍콩대) + University of Waterloo(워털루대) |
| **공개일** | 2026-04 |
| **분야** | unified multimodal model(통합 멀티모달 모델, UMM) — 이미지 이해(understanding) + 생성(generation)을 한 모델로 |
| **arXiv** | [abs](https://arxiv.org/abs/2604.24763) |
| **코드** | [facebookresearch/tuna-2](https://github.com/facebookresearch/tuna-2) |
| **선행 논문** | Tuna (CVPR 2026, [2501.10441](https://arxiv.org/abs/2501.10441)) |
| **외부 모델/데이터** | Qwen2.5-7B-Instruct(LLM 백본) · WAN2.2 VAE(비교군) · SigLIP2(비교군) |

---

## 📖 주요 용어 사전 (Glossary)

### 아키텍처
- **UMM (unified multimodal model, 통합 멀티모달 모델)**: 텍스트와 이미지를 하나의 LLM 시퀀스 안에서 같이 처리해, 이미지를 "이해"하고 "생성"하는 일을 한 모델로 합친 것. (예: BAGEL, Show-o2). TUNA-2는 그중 **native UMM** — 이미지를 별도 모듈이 아니라 LLM 토큰 그 자체로 다룸.
- **vision encoder(비전 인코더)**: 이미지를 LLM이 받기 좋은 "의미 있는 벡터"로 미리 압축하는 사전학습 모듈. SigLIP·CLIP·VAE가 대표. TUNA-2의 핵심 주장은 **이걸 0개로 없애자**는 것.
- **SigLIP2**: 이미지-텍스트 contrastive(대조 학습)로 정렬된 비전 인코더. "이건 얼굴, 저건 하늘" 식의 의미 정렬된 표현을 만듦. TUNA-2의 비교군.
- **VAE (variational autoencoder, 변분 오토인코더)**: 이미지를 작은 잠재(latent)로 압축(encode)했다가 복원(decode)하는 모듈. latent diffusion(잠재 확산)의 1단계.
- **SimplePatchEmbedding(단순 패치 임베딩)**: TUNA-2의 비전 입력단. `Conv2d(3→3584, kernel=16, stride=16) + RMSNorm` 단 한 줄. 비전 인코더 전체를 이 2.76M 파라미터짜리 Conv 하나로 대체.
- **diffusion head(디퓨전 헤드)**: LLM 본체 위에 얹은 16-layer DiT(adaLN modulation). 노이즈 낀 이미지를 받아 클린 이미지를 회귀하는 생성 전담부.
- **encoder-free(인코더 없는)**: 비전 인코더를 두지 않고 raw RGB 픽셀을 LLM에 직접 흘리는 설계. TUNA-2(variant C)의 정체성.

### 핵심 개념
- **native(네이티브) 처리**: 이미지를 인코더로 추상화하지 않고, raw RGB 패치를 LLM 토큰으로 그대로 넣어 7B 파라미터가 "의미 추출"까지 직접 배우게 하는 것.
- **JiT (Just-image-Transformer) x0-prediction(클린 픽셀 직접 예측)**: 노이즈에서 클린 이미지로 가는 방향(velocity)을 맞추는 대신, **클린 이미지 x0 자체를 직접 회귀**하는 학습 타깃. 픽셀 공간에서 분산 폭주를 피함.
- **velocity prediction(속도 예측)**: rectified flow(직선 흐름)의 표준 타깃. `noise_scale·ε − x0` 방향을 맞춤. 픽셀 공간에선 noise_scale에 비례해 분산이 커지는 약점.
- **JiTNoiseScheduler**: timestep을 lognormal(P_mean=−0.8)로 뽑아 **작은 t(노이즈 적은 구간, 디테일 학습)** 를 더 자주 보게 하는 노이즈 스케줄러.
- **DeTok-style mask token(마스크 토큰)**: 학습 중 일부 패치를 학습 가능한 mask 벡터로 무작위 치환하는 정규화. TUNA-2는 음수 min ratio 트릭으로 70% 샘플은 마스킹을 아예 스킵.
- **block-diagonal attention(블록 대각 어텐션)**: 한 시퀀스 안에서 텍스트 영역은 causal(앞쪽만 보기), 이미지 영역은 bidirectional(양쪽 보기)로 묶는 마스크.

### 비교/평가
- **crossover(교차점)**: 학습 초반엔 인코더 기반이 앞서다가, 충분히 학습하면 encoder-free가 역전하는 지점. 이 논문 결론의 핵심 조건.
- **fine-grained perception(세밀 인지)**: 작은 글자·세부 패턴 읽기 등 디테일 의존 과제. encoder-free의 역전 폭이 가장 큰 영역.
- **NTP (next-token prediction, 다음 토큰 예측)**: 이해(understanding) 쪽 손실. 텍스트 토큰을 맞춤.

---

## 🎯 논문 요약 (TL;DR)

**한 줄**: 비전 인코더(SigLIP/VAE)를 0개로 없애고 raw RGB 픽셀을 `Conv2d` 한 줄로 LLM(Qwen2.5-7B)에 직접 흘려도, **충분히 키우면 인코더 있는 쪽을 이긴다**는 것을 같은 백본 위 통제된 3-way ablation으로 실증.

**핵심 문제**: 기존 native UMM은 거의 다 사전학습 비전 인코더를 앞단에 둔다. 그런데 이 인코더는 contrastive loss로 학습된 **lossy(손실) 압축기** — 픽셀 디테일을 버린다. 저자들은 이 버려진 디테일이 모델 성능의 **천장(ceiling)** 이 된다고 본다.

**해결책**: 인코더를 떼어내고 raw 픽셀을 LLM에 직접 넣는다(encoder-free). 인코더가 하던 "의미 추출"을 7B 파라미터 LLM 본체가 직접 학습하게 한다. 생성 쪽은 VAE decoder까지 없애고 디퓨전 헤드가 **픽셀을 직접 회귀**(JiT x0-prediction)한다.

**검증**: 백본(Qwen2.5-7B)을 고정한 채 비전 입력단만 3가지(VAE+SigLIP / SigLIP만 / Conv2d만)로 바꿔 비교. 결과 — 학습 초반엔 인코더 기반이 빨리 수렴하지만, **scale에서 encoder-free가 역전**하며, 특히 fine-grained 인지에서 우위가 크다.

---

## 🔑 핵심 기여 (Contributions)

1. **encoder-free native UMM 실증**: 비전 인코더 없이 raw 픽셀만으로도, 충분한 scale에서 인코더 기반을 이긴다는 것을 처음으로 통제 실험으로 보임.
2. **통제된 3-way ablation**: 백본을 Qwen2.5-7B로 고정하고 **입력단만** 교체(A: VAE+SigLIP, B: SigLIP, C: Conv2d). "인코더 유무"의 효과를 깨끗하게 분리.
3. **SimplePatchEmbedding**: 400M짜리 SigLIP을 2.76M 파라미터 `Conv2d + RMSNorm` 한 줄로 대체하는 미니멀 입력단.
4. **디코더까지 제거**: VAE decoder 없이 디퓨전 헤드가 픽셀을 직접 출력(JiT x0-prediction). "인코더도 디코더도 lossy"라는 일관된 철학.
5. **crossover 현상 규명**: 인코더는 좋은 출발점이지만 scale에서 천장이 된다는 시간 의존적(scale-dependent) 결론을 명시.

---

## 🧩 주요 알고리즘 설명

### 0️⃣ 전체 구조 — 인코더 0개 + 통합 시퀀스 + 디퓨전 헤드

*왜 보나: TUNA-2의 본질은 "raw RGB 픽셀이 LLM 안에서 그대로 흐른다"는 것. 입력·중간·출력 어디에도 인코더/디코더가 없다는 걸 한눈에 봐야 나머지 알고리즘이 이해된다.*

variant C(=TUNA-2 본체)의 T2I(text-to-image) 흐름:

```
text_tokens (B, L_text)            image_latents=RGB (B,3,1,512,512)        t (B,)
     │                                      │                                │
     ▼                                      ▼                                ▼
[Qwen2.5 embed_tokens]          [SimplePatchEmbedding]                  [TimestepEmb]
(B, L_text, 3584)               Conv2d(3→3584,k=16,s=16)+RMSNorm        sinusoid+MLP
   ★ img_pad 슬롯이              (B, 1024, 3584)  ★ 비전 인코더 0개      (B, 3584)
   곧 image_embeds로 덮임             │                                     │
     │        ┌──────────────────────┘                                     │
     ▼        ▼                                                            │
   [_prepare_input] modality_positions 따라 img_embeds를                   │
   text 시퀀스의 [boi]+[img_pad×N]+[eoi] 자리에 삽입 + time_embed prepend   │
     │                                                                     │
     ▼                                                                     │
   input_embeds (B, L, 3584),  L = L_text + 1024(@512²)                    │
     │                                                                     │
     ▼                                                                     │
   ┌ Qwen2.5-7B (28 layers) ── block-diagonal attention ────┐             │
   │  text=causal, img=bidirectional, text→img 보임, img→text 가림         │
   │  3D RoPE: time × height × width                         │             │
   └────────────────────────────────────────────────────────┘             │
     │                                                                     │
   ┌─┴───────────────┐                                                     │
   ▼                 ▼                                                     │
logits(NTP)     last_hidden (B, L, 3584)                                   │
(이해)               │                                                     │
                     ▼                                                     │
   ┌ Diffusion Head (16-layer DiT, adaLN modulation) ◀──── time_embed ─────┘
   │  adaLN(time)→shift/scale/gate ×2 · Self-Attn(RoPE) · SwiGLU FFN
   └──────────────────┘
                     │
                     ▼
   [FinalLayer] AdaLN 2-param + Linear(3584 → 16²·3=768) → [unpatchify]
                     │
                     ▼
   x0_pred (=클린 픽셀 직접, VAE decoder 없이 그대로 사용)

   loss = ntp_coeff·CE(logits, text_labels) + flow_coeff·MSE(x0_pred, x0)·w(t)
```

핵심: 512² 이미지 → **1024 토큰(32×32 패치)**. 세 변종 모두 토큰 수는 같다. 다른 건 **"각 토큰이 무엇을 담는가"** — 인코더 변종은 의미 정렬된 압축 표현, encoder-free는 raw RGB 패치 그대로.

---

### 1️⃣ SimplePatchEmbedding — 인코더 자리를 Conv2d 한 줄로

*왜 이걸 두나: 인코더가 미리 압축하면 디테일이 lossy하게 깎인다. 압축을 LLM 본체에 맡기려면 입력단은 "raw 픽셀을 자르기만" 하면 충분하다는 게 TUNA-2의 가설.*

**비유**: 기존 UMM은 편지를 스캐너(SigLIP)로 OCR해서 "텍스트화된 요약"만 봉투에 넣는다. TUNA-2는 편지를 16×16 조각으로 잘라 **원본 그대로** 봉투에 넣고 LLM이 직접 읽게 한다.

**3단계 분해** ([patch_embed.py](https://github.com/facebookresearch/tuna-2)):
```
입력: pixel_values (B, 3, 512, 512)  RGB [-1,1]
[1] Conv2d(3→3584, k=16, s=16)  → (B, 3584, 32, 32)   ※ 16×16 블록을 겹침 없이 3584차원으로
[2] flatten+transpose           → (B, 1024, 3584)     ※ LLM 입력 시퀀스 형식
[3] RMSNorm(3584)               → (B, 1024, 3584)     ※ Qwen2.5 input_embeds 자리에 삽입
※ 학습 파라미터 ≈ 3×16×16×3584 + 3584 ≈ 2.76M  (SigLIP 400M의 약 1/150)
```

**대안 비교**:

| 비전 입력 | 사전학습 weight | 학습 파라미터 | 1024 토큰까지 비용 | 정보 흐름 |
|---|---|---|---|---|
| VAE+SigLIP (A) | WAN2.2 VAE 250M + SigLIP2 400M | ~5M | VAE forward + SigLIP 26층 | 압축된 의미 |
| SigLIP만 (B) | SigLIP2 400M | ~5M | SigLIP 26층 | 정렬된 표현 |
| **Conv2d만 (C)** | **없음** | **~2.76M** | **conv 1번** | **raw 픽셀** |

> **한 줄**: SimplePatchEmbedding = "Conv2d 1개 + RMSNorm 1개로 비전 인코더 통째 대체". 인코더가 하던 의미 추출을 7B LLM이 흡수.

---

### 2️⃣ 디코더 부재 — 픽셀을 직접 출력

*왜 이렇게 하나: 인코더가 lossy면 복원하는 VAE decoder도 똑같이 lossy다. 출력 공간을 처음부터 픽셀로 두면 decode 단계 자체가 불필요.*

**비유**: 기존 디퓨전은 DiT가 latent 잉크 패턴을 그리고 → VAE decoder(인쇄소)가 종이에 인쇄한다(2단계). TUNA-2는 디퓨전 헤드가 **종이에 직접** 그린다(인쇄소 제거).

```
[기존 latent diffusion]  ε → DiT(t) → latent z0 → VAE.decode → pixel    ← decode가 별도 50~250M
[TUNA-2]                 ε → LLM+DiffHead(t) → x0(=픽셀 그대로)          ← 이미 픽셀이라 decode 불필요
```

코드의 `t2i_pixel` 분기는 `images = latents` 한 줄이 전부 — denorm([-1,1]→[0,255])만 거쳐 PIL. 단, 이건 **7B 백본이 디코더 자리까지 흡수 가능할 때만** 성립(작은 모델엔 부담 폭증).

> **한 줄**: 디코더 부재 = "7B 파라미터가 디코더 역할까지 흡수". 인코더·디코더 둘 다 lossy라는 일관된 철학.

---

### 3️⃣ JiT x0-prediction loss — velocity 대신 클린 픽셀 직접 회귀

*왜 velocity를 안 쓰나: 픽셀 공간에서 velocity 타깃은 noise_scale에 비례해 분산이 폭주한다. 클린값 x0는 항상 [-1,1] 안이라 안정적.*

**비유**: velocity = "지금 어느 방향·속도로 가야 하나"를 매 step 새로 묻기. x0 = "목적지 좌표가 어디냐"를 매 step 같은 답으로 묻기.

**학습 절차** ([jit_utils.py](https://github.com/facebookresearch/tuna-2)):
```
x0 : 클린 픽셀,  ε ~ N(0,I),  t ~ lognormal(P_mean=-0.8, P_std=0.8)→[0,1],  noise_scale=2.0
x_t      = (1-t)·x0 + t·noise_scale·ε        ← 직선 보간(linear interpolation)
x0_pred  = model(x_t, t)                     ← ★ 클린 픽셀 직접 예측
loss_flow = MSE(x0_pred, x0)·w(t)            ← w(t)는 SNR 함수
loss_ntp  = CE(logits, text_labels)          ← 이해 쪽
```

**왜 픽셀에서 x0이 안정한가** (수치 예시, x0=0.5, ε=0.3, t=0.5):
```
noise_scale=2.0 → velocity = 2.0·0.3 - 0.5 = 0.10,   x0 타깃 = 0.5
noise_scale=5.0 → velocity = 5.0·0.3 - 0.5 = 1.00 (진폭 10배),   x0 타깃 = 0.5 (불변)
```
→ noise_scale를 키우면 velocity 분산이 직격당하지만 x0 타깃은 그대로. TUNA-2가 noise_scale=2.0으로 노이즈를 키운 것도 x0-prediction이라 가능한 선택.

| 예측 타깃 | 학습 안정성 | step 외삽 | 픽셀 공간 적합도 |
|---|---|---|---|
| velocity | 좋음 | DPM 14~20 step | △ high-freq 폭주 위험 |
| noise(ε) | △ t≈1 불안정 | △ 50+ step | △ |
| **x0 (JiT)** | **매우 좋음** | **큰 보폭 안전** | **✅ 자연스러움** |

> **한 줄**: JiT x0-prediction = "클린 픽셀 직접 회귀로 픽셀 공간의 velocity 분산 폭주 회피".

---

### 4️⃣ JiTNoiseScheduler — lognormal로 디테일 구간을 더 자주

*왜 이렇게 뽑나: 인코더 없이 디테일을 직접 배워야 하므로, 노이즈가 적어 디테일이 보이는 작은 t 구간을 더 자주 학습시킨다.*

**비유**: uniform t = 셔터 속도를 골고루 샘플링. lognormal(P_mean=−0.8) = "빠른 셔터(작은 t, 노이즈 적음)"를 더 자주 — 디테일 학습 강조.

```
self.jit_noise_scheduler = JiTNoiseScheduler(P_mean=-0.8, P_std=0.8, noise_scale=2.0, t_eps=5e-2)
[샘플링]  σ = exp(randn·P_std + P_mean);  t = σ/(1+σ);  t = clamp(t, t_eps, 1-t_eps)
```
P_mean=−0.8이면 mode가 t≈0.31 부근으로 작은 t 쪽으로 치우침 → x_t가 거의 클린 → 디테일 회귀에 집중.

> **한 줄**: JiTNoiseScheduler = "lognormal P_mean=−0.8로 디테일 학습 구간을 더 자주 샘플링".

---

### 5️⃣ DeTok-style mask token — 음수 min ratio로 stochastic 마스킹

*왜 두나: encoder-free 모델은 raw 픽셀 의존도가 커서, 약한 마스킹 정규화로 학습 분포를 다양화한다. 단 너무 자주 가리면 불안정해 대부분 샘플은 스킵한다.*

**비유**: BERT식 마스킹은 매번 "정확히 30% 가린다". TUNA-2는 매 샘플 "0~30% 또는 아예 안 함"을 무작위로 — `min=-0.7`이라는 음수가 "안 함"을 의미.

```python
# config: masked_image_ratio_min=-0.7, masked_image_ratio=0.3
for b in range(B):
    mask_ratio = random.uniform(-0.7, 0.3)
    num_masked = int(N * mask_ratio)        # 음수면 아래 if False → 마스킹 스킵
    if num_masked > 0:
        idx = np.random.choice(N, num_masked, replace=False)
        image_embeds[b, idx] = self.mask_token
```
- `ratio<0` 확률 = 0.7/(0.7+0.3) = **70% 스킵**
- `ratio≥0` 확률 30%, 이때만 0~30% 균등 마스킹
- 실제 평균 마스킹 ≈ 0.30 × 0.15 = **4.5%** 만 적용

→ "마스킹은 보조 정규화일 뿐, 메인은 클린 학습"이라는 우선순위를 음수 트릭으로 표현.

> **한 줄**: DeTok mask + 음수 min ratio = "70% 샘플은 스킵, 30%만 0~30% 마스킹"하는 가벼운 정규화.

---

## 📊 실험 요약

*왜 보나: 이 논문의 가치는 단일 SOTA 숫자가 아니라 "백본 고정·입력단만 교체"라는 통제 실험과 그 결과인 crossover 곡선에 있다.*

### 통제된 3-way ablation (같은 Qwen2.5-7B)

| 변종 | 비전 입력 | 디퓨전 타깃 | 손실 |
|---|---|---|---|
| **A. Tuna** | RGB → WAN2.2 VAE → SigLIP2 | VAE latent | velocity (rectified flow) |
| **B. Tuna-R** | RGB → SigLIP2 (26층) | 픽셀 3ch | JiT x0-prediction |
| **C. Tuna-2** | RGB → Conv2d(3→3584) + RMSNorm | 픽셀 3ch | JiT x0-prediction |

### 핵심 결과 — crossover

논문 abstract가 직접 인정하는 시간 의존적 결론:

- **학습 초반**: 인코더 기반(A/B)이 **빨리 수렴**. 사전학습 표현이 좋은 출발점이기 때문. (*"encoder-based variant converges faster in early pretraining"*)
- **충분히 학습 후**: encoder-free(C)가 **역전**. 인코더의 lossy 압축이 천장이 됨.
- **fine-grained 인지**: 작은 글자·세밀 패턴 등 디테일 의존 과제에서 역전 폭이 가장 큼. 인코더가 버린 게 바로 그 디테일이라서.

→ 결론은 무조건적 "인코더 나쁨"이 **아니라**, **"데이터·step·모델 크기를 충분히 쓸 수 있으면 인코더의 압축 손실이 손해로 바뀐다"** 는 조건부 주장이다.

---

## 💬 Q&A

### Q1. "인코더를 없앤다"는 게 정확히 무슨 의미인가?

비전 입력단 3가지를 "벗겨내는" 과정으로 보면 명확하다:
```
A. VAE+SigLIP : RGB → [VAE 250M] → 32×32×48 → [SigLIP2 400M]  → VAE latent (velocity)
B. SigLIP만   : RGB → [SigLIP2 400M, 26층]                    → 픽셀 (x0)
C. Conv2d만   : RGB → [Conv2d + RMSNorm, 2.76M]               → 픽셀 (x0)
```
C는 사전학습 weight가 0개. 인코더가 하던 "의미 추출"을 LLM 본체(7B)가 직접 학습한다. 토큰 수(1024)는 세 변종 모두 같고, 다른 건 토큰이 담는 내용(압축된 의미 vs raw RGB)뿐이다.

### Q2. 픽셀 공간인데 왜 velocity가 아니라 x0을 예측하나?

velocity 타깃 `noise_scale·ε − x0`는 노이즈 진폭에 직접 비례해, 큰 noise_scale에서 분산이 폭주한다(Q 알고리즘 3️⃣ 수치 예시). x0은 항상 [-1,1] 안이라 분산이 안정적. 그래서 TUNA-2는 노이즈를 키우면서(noise_scale=2.0) 동시에 x0-prediction을 써서 안정성을 확보했다. 둘은 한 묶음 설계다.

### Q3. 이게 latent diffusion(FLUX, SD)과 근본적으로 뭐가 다른가?

두 군데서 다르다. ① 입력단 — latent diffusion은 VAE로 압축한 latent에서 작업, TUNA-2는 raw 픽셀. ② 출력단 — latent diffusion은 VAE decoder로 픽셀 복원, TUNA-2는 디퓨전 헤드가 픽셀 직접 출력. 즉 **인코더와 디코더를 둘 다 제거**하고 그 일을 LLM 본체에 흡수시킨 게 차이. 단 이 흡수는 7B 백본 + 막대한 학습량이 있어야 성립한다.

### Q4. 논문 제목만 보면 "인코더는 무조건 나쁘다"인데, 진짜 그런가?

아니다. 본문은 **scale 조건부**다. 초반엔 인코더가 이기고(저자도 인정), crossover 이후에 역전한다. 그 crossover 지점이 어디냐가 실무 채택을 좌우하는데 — 작은 모델·적은 학습 예산이라면 인코더가 끝까지 유리할 수 있다. 제목은 도발적이지만 결론은 조건부라는 점을 분리해 읽어야 한다.

### Q5. 이 논문의 약점은?

1. **"충분한 scale"의 비용이 숨겨짐** — encoder-free가 이기려면 7B 백본 + 막대한 학습이 필요. 복잡도를 인코더에서 LLM 본체로 **옮긴** 것이지 없앤 게 아니다. 같은 FLOPs 예산 비교가 약함.
2. **crossover 지점이 흐릿** — "언젠가 역전"은 알겠으나 정확한 step/데이터 규모가 불명확.
3. **타깃 entangle 가능성** — variant A는 velocity, C는 x0을 쓴다. "인코더 효과"와 "예측 타깃 효과"가 일부 섞였을 수 있다.
4. **일반화 범위** — Qwen2.5-7B라는 특정 조합의 결론. 1~2B 백본에서도 같은 crossover가 나는지는 미검증.

---

## 📝 한 줄 요약 (전체)

> **TUNA-2 = "비전 인코더 0개 + 디코더 0개 + Qwen2.5-7B 위 16-layer DiT 헤드 + JiT x0-prediction"**. raw RGB 패치를 LLM에 그대로 흘려, **충분한 scale에서 인코더 기반을 역전**한다는 걸 백본 고정 3-way ablation으로 실증한 논문. 주장은 도발적이고 실험 설계는 깔끔하지만, 결론이 대규모 학습 예산이라는 **비싼 입장권에 강하게 묶여 있다**(scale-dependent)는 게 핵심 한계.

---

## 🔗 관련 메모리 링크

- [[paper_pixeldit]] — VAE 제거 픽셀 확산. TUNA-2의 "디코더 부재"와 같은 동기(인코더/디코더는 lossy), 단 PixelDiT는 순수 생성 모델
- [[paper_repa]] — 인코더의 의미 표현을 보조 손실로 끌어오는 정반대 접근(인코더 활용 극대화)
- [[paper_qwen_image]] / [[paper_z_image]] — 동결 LLM/인코더 재사용 계보. TUNA-2는 그 인코더 의존을 정면으로 부정
- [[reference_pretrained_backbone_reuse_landscape]] — 사전학습 백본 재사용 3분기. TUNA-2는 "인코더 미사용" 극단 사례
