# PAPER: Z-Image - 6B DiT로 큰 이미지 생성 모델을 따라잡는 효율 설계

## 0. 이 문서를 읽는 법

이 문서는 Z-Image 논문과 공개 추론 코드를 처음 읽는 사람이 흐름을 놓치지 않도록 다시 정리한 리뷰입니다.

핵심 목표는 하나입니다.

> **Z-Image는 20B-80B급 대형 이미지 생성 모델을 무작정 따라 키우지 않고, 6B DiT + 좋은 데이터 큐레이션 + 효율적인 학습/증류로 경쟁력 있는 성능을 만든 모델이다.**

이 문서는 DMD, DMD2 문서와 같은 방식으로 읽기 쉽게 구성했습니다.

1. **큰 그림**: Z-Image가 해결하려는 문제
2. **등장인물**: 텍스트 인코더, DiT, VAE, teacher/student
3. **모델 구조**: Single-Stream DiT가 왜 중요한지
4. **데이터와 학습**: 작은 모델을 똑똑하게 만드는 재료
5. **증류와 RLHF**: Turbo가 왜 8 step으로 동작하는지
6. **공개 코드 흐름**: prompt가 이미지가 되는 과정
7. **실험 결과와 FAQ**: 숫자와 헷갈리는 점

수식은 GitHub에서 깨지지 않도록 LaTeX 대신 일반 텍스트 표기로 적었습니다.

```text
복잡한 LaTeX 수식 대신:
L_DMDR = L_DM + lambda * L_RL
```

---

## 1. 메타 정보

| 항목 | 내용 |
|---|---|
| 논문 | Z-Image: An Efficient Image Generation Foundation Model |
| 저자 | Tongyi MAI Team, Alibaba |
| arXiv | https://arxiv.org/abs/2511.22699 |
| PDF | https://arxiv.org/pdf/2511.22699 |
| 공식 사이트 | https://tongyi-mai.github.io/Z-Image-blog/ |
| 공식 코드 | https://github.com/Tongyi-MAI/Z-Image |
| Hugging Face | `Tongyi-MAI/Z-Image`, `Tongyi-MAI/Z-Image-Turbo` |
| 분야 | Text-to-Image, Image Editing, Diffusion Transformer |
| 공개 범위 | 추론 코드와 가중치 공개. 학습 코드, 학습 데이터, 보상 모델은 비공개 |

---

## 2. 한 문장 요약

> **Z-Image는 "더 큰 모델"이 아니라 "더 효율적인 전체 파이프라인"으로 성능을 만든 6B 이미지 생성 모델이다.**

조금 더 쉽게 말하면:

> **텍스트 이해는 Qwen3-4B에게 맡기고, 이미지는 6B Single-Stream DiT가 만들며, 데이터 큐레이션과 Decoupled DMD 증류로 작은 모델의 한계를 보완한 시스템이다.**

Z-Image를 하나의 모델이라고만 보면 이해가 어렵습니다. 정확히는 아래 네 가지가 같이 맞물린 결과입니다.

```text
좋은 데이터
  + 효율적인 DiT 구조
  + 단계별 학습 커리큘럼
  + few-step 증류와 RLHF
  = 6B로 큰 모델급 성능
```

---

## 3. Z-Image가 해결하려는 문제

최근 이미지 생성 모델은 점점 커지고 있습니다.

```text
Qwen-Image      약 20B
FLUX.2 dev      약 32B
HunyuanImage    약 80B
Z-Image          6B
```

모델을 크게 만들면 성능은 좋아질 수 있지만 문제가 있습니다.

| 문제 | 왜 문제인가 |
|---|---|
| 학습 비용 증가 | 학계나 작은 팀이 재현하기 어려움 |
| 추론 비용 증가 | 사용자에게 이미지를 제공할 때 비용이 커짐 |
| synthetic distillation 의존 | 상용 모델이 만든 이미지를 다시 학습 데이터로 쓰는 우회로가 많아짐 |
| 오픈소스 격차 | 공개 모델이 점점 무겁고 다루기 어려워짐 |

Z-Image의 출발점은 이 질문입니다.

> **모델을 20B, 30B, 80B로 키우지 않고도 좋은 이미지 생성 모델을 만들 수 있을까?**

저자의 답은 "가능하다"입니다. 대신 모델 하나만 잘 만드는 것이 아니라, 데이터부터 추론까지 전체를 효율화합니다.

---

## 4. 전체 구조 한눈에 보기

Z-Image는 크게 세 모델 조각으로 돌아갑니다.

```text
prompt
  -> Qwen3-4B text encoder
  -> text feature
  -> S3-DiT image generator
  -> latent image
  -> Flux VAE decoder
  -> RGB image
```

각 조각의 역할은 다음과 같습니다.

| 구성 요소 | 역할 |
|---|---|
| Qwen3-4B | prompt를 이해해서 텍스트 feature를 만듦 |
| S3-DiT | 텍스트 feature와 noisy image latent를 보고 denoising 방향을 예측 |
| Flux VAE | DiT가 만든 latent를 실제 RGB 이미지로 복원 |
| FlowMatchEuler scheduler | 여러 timestep에서 latent를 조금씩 업데이트 |

Z-Image 계열 모델은 아래처럼 나뉩니다.

| 모델 | 추론 step | CFG | 용도 | 추가 학습 |
|---|---:|---|---|---|
| Z-Image-Omni-Base | 50 | 사용 | 사전학습 base | 쉬움 |
| Z-Image | 50 | 사용 | 일반 고품질 T2I | 쉬움 |
| Z-Image-Turbo | 8 | 미사용 | 빠른 생성 | 권장 안 함 |
| Z-Image-Edit | 50 | 사용 | 이미지 편집 | 쉬움 |

가장 중요한 구분은 이것입니다.

```text
Z-Image      = 품질 중심 50-step 모델
Z-Image-Turbo = 속도 중심 8-step distilled 모델
```

---

## 5. 등장인물 정리

### 5.1 Text encoder, Qwen3-4B

Z-Image는 텍스트 이해를 DiT 안에서 처음부터 배우게 하지 않습니다. 이미 강한 LLM인 Qwen3-4B를 텍스트 인코더로 사용합니다.

```text
prompt
  -> Qwen3-4B
  -> hidden_states[-2]
  -> linear projection
  -> text token features
```

여기서 중요한 점:

| 항목 | 의미 |
|---|---|
| `hidden_states[-2]` | 마지막 직전 layer의 hidden state를 사용 |
| 2560 -> 3840 | Qwen feature 차원을 DiT hidden dimension에 맞춤 |
| max length 512 | prompt를 최대 512 token까지 처리 |
| reasoning mode | Qwen의 reasoning hidden state를 활용 |

쉽게 말하면:

> **Qwen이 prompt를 "이해"하고, DiT는 그 이해 결과를 보고 이미지를 만든다.**

### 5.2 Image generator, S3-DiT

S3-DiT는 Z-Image의 핵심 생성기입니다.

S3-DiT는 **Scalable Single-Stream Diffusion Transformer**의 약자입니다.

한 줄 정의:

> **텍스트 token과 이미지 token을 한 줄로 이어붙여 같은 Transformer block이 같이 처리하는 DiT 구조이다.**

```text
image tokens
text tokens
  -> concat
  -> same Transformer blocks
  -> image tokens만 꺼냄
  -> velocity 예측
```

### 5.3 VAE, Flux VAE

DiT는 RGB 이미지를 직접 만들지 않습니다. 압축된 latent 공간에서 작업합니다.

```text
RGB image 1024 x 1024
  -> VAE encode
  -> latent 16 x 128 x 128

latent 16 x 128 x 128
  -> VAE decode
  -> RGB image 1024 x 1024
```

Z-Image 공개 코드는 Flux VAE 계열 autoencoder를 사용합니다.

### 5.4 Teacher, student, critic

Z-Image-Turbo를 이해하려면 DMD/DMD2에서 쓰던 역할 구분이 필요합니다.

| 역할 | Z-Image에서의 의미 |
|---|---|
| teacher | 느리지만 품질 좋은 50-step 모델 |
| student | 8-step으로 빠르게 이미지를 만들 Turbo 모델 |
| critic 또는 fake score model | student가 만든 분포의 score를 추정하는 보조 모델 |
| reward model | 사람이 선호할 만한 결과인지 점수를 주는 모델 |

DMD/DMD2 문서의 표기로 연결하면:

| 기호 | 이 문서에서의 의미 |
|---|---|
| `G_theta` | 학습되는 student generator |
| `p_real` | teacher 또는 real image 분포 |
| `p_fake_theta` | 현재 student가 만드는 이미지 분포 |
| `s_real` | teacher가 알려주는 real score |
| `s_fake` | critic이 알려주는 fake score |

---

## 6. S3-DiT 구조

### 6.1 핵심 수치

| 항목 | 값 |
|---|---:|
| 전체 파라미터 | 약 6.15B |
| backbone layer | 30 |
| image refiner | 2 layer |
| text refiner | 2 layer |
| hidden dimension | 3840 |
| attention head | 32 |
| head dimension | 128 |
| FFN intermediate dimension | 10240 |
| image latent channel | 16 |
| patch size | spatial 2 x 2, temporal 1 |
| RoPE 축 분배 | t 32, h 48, w 48 |
| adaLN 공유 차원 | 256 |

### 6.2 Single-Stream이란 무엇인가

기존 멀티모달 DiT는 보통 text와 image를 어느 정도 분리해서 처리합니다. SD3, FLUX 계열의 dual-stream 방식은 text용 가중치와 image용 가중치를 따로 둡니다.

```text
Dual-Stream

text tokens  -> text weights  \
                              -> joint attention -> output
image tokens -> image weights /
```

Z-Image는 이와 다릅니다.

```text
Single-Stream

text tokens  \
              -> concat -> same weights -> output
image tokens /
```

즉, 같은 Transformer block이 텍스트와 이미지를 모두 봅니다.

### 6.3 왜 이게 효율적인가

핵심은 "파라미터를 나눠 쓰지 않는다"입니다.

Dual-stream은 같은 크기의 모델이라도 내부적으로 역할이 쪼개집니다.

```text
text 전용 가중치
image 전용 가중치
```

Single-stream은 같은 가중치가 양쪽 학습 신호를 모두 받습니다.

```text
same weights
  <- text 학습 신호
  <- image 학습 신호
  <- text-image alignment 학습 신호
```

그래서 Z-Image의 주장은 다음과 같습니다.

> **6B를 단일 스트림으로 잘 학습하면, 파라미터를 분리한 더 큰 모델과 비슷한 유효 capacity를 낼 수 있다.**

여기서 "유효 capacity"는 실제 파라미터 수가 2배라는 뜻이 아닙니다. 같은 가중치가 더 다양한 학습 신호를 받아 더 효율적으로 쓰인다는 뜻입니다.

### 6.4 Single-Stream의 문제와 Z-Image의 해결

텍스트와 이미지는 성질이 다릅니다. 둘을 그냥 한 줄에 섞으면 학습이 불안정해질 수 있습니다.

Z-Image는 아래 장치들로 안정화합니다.

| 장치 | 하는 일 |
|---|---|
| Noise Refiner | 이미지 token을 backbone에 넣기 전 2 layer로 정리 |
| Context Refiner | text token을 backbone에 넣기 전 2 layer로 정리 |
| 3D Unified RoPE | text 위치와 image 공간 위치를 다른 축으로 표현 |
| QK-Norm | attention의 query/key 크기 폭주를 줄임 |
| Sandwich-Norm | block 입력과 출력 양쪽 정규화로 깊은 학습 안정화 |
| tanh-gated residual | residual branch가 갑자기 커지는 것을 막음 |
| low-rank adaLN | timestep 정보를 저비용으로 모든 layer에 주입 |

### 6.5 forward 흐름

S3-DiT의 forward를 순서대로 쓰면 다음과 같습니다.

```text
1. timestep t
   -> sinusoidal embedding
   -> MLP
   -> adaLN vector

2. image latent
   -> patchify
   -> image tokens
   -> Noise Refiner 2 layers

3. prompt feature
   -> projection
   -> text tokens
   -> Context Refiner 2 layers

4. image tokens + text tokens
   -> concat
   -> Backbone 30 layers

5. output에서 image tokens만 선택
   -> final layer
   -> unpatchify
   -> velocity prediction
```

여기서 output은 clean image가 아니라 **velocity**입니다.

Flow Matching 관점에서 velocity는:

```text
현재 noisy latent가 다음에 어느 방향으로 움직여야 하는가
```

를 뜻합니다.

---

## 7. 데이터 인프라

Z-Image의 중요한 주장 중 하나는 synthetic distillation data를 쓰지 않았다는 점입니다.

즉:

```text
상용 이미지 생성 모델이 만든 이미지를 대량으로 베껴 학습하지 않았다.
```

대신 Alibaba가 보유한 real-world 데이터 풀에서 좋은 데이터를 골라냅니다.

### 7.1 왜 데이터가 중요한가

작은 모델은 큰 모델보다 외우고 일반화할 여유가 적습니다. 그래서 데이터가 더 중요합니다.

```text
큰 모델:
  어느 정도 지저분한 데이터도 흡수 가능

작은 모델:
  잘 고른 데이터가 아니면 capacity를 낭비
```

Z-Image는 이 문제를 데이터 큐레이션으로 풉니다.

### 7.2 4개 모듈

| 모듈 | 쉬운 설명 |
|---|---|
| Data Profiling Engine | 이미지 품질, 중복, 해상도, 압축 흔적, NSFW, AI 생성 여부, 의미 태그를 자동 측정 |
| Cross-modal Vector Engine | 이미지와 텍스트를 같은 embedding 공간에 넣고 중복/유사 샘플을 빠르게 찾음 |
| World Knowledge Topological Graph | 위키피디아 기반 개념 그래프로 드문 개념을 찾아 학습 비중을 조절 |
| Active Curation Engine | 모델이 실패한 prompt를 분석해 부족한 데이터를 다시 보충 |

핵심은 마지막 Active Curation입니다.

```text
모델이 못 그리는 개념 발견
  -> 관련 데이터 검색
  -> 데이터셋 보강
  -> 다시 학습
  -> 실패 감소
```

이런 구조를 closed loop라고 부릅니다.

### 7.3 학습 데이터 갈래

| 갈래 | 내용 |
|---|---|
| T2I single image | 이미지 1장과 여러 길이의 caption/prompt |
| I2I pair | 원본 이미지와 편집된 이미지의 쌍 |
| SFT curated set | 작지만 품질이 높은 후처리용 데이터 |
| RLHF preference pair | chosen/rejected 비교 학습용 데이터 |

Z-Captioner는 이미지 한 장에 대해 여러 수준의 설명을 만듭니다.

```text
tag caption
long caption
medium caption
short caption
simulated user prompt
OCR text
world knowledge
```

이렇게 하면 같은 이미지에서도 여러 종류의 학습 신호를 얻을 수 있습니다.

---

## 8. 학습 커리큘럼

Z-Image 학습은 크게 pre-training과 post-training으로 나눌 수 있습니다.

```text
Pre-training
  -> low-resolution T2I
  -> Omni pre-training

Post-training
  -> SFT
  -> Distillation
  -> RLHF
```

### 8.1 단계별 compute 비중

| 단계 | H800 GPU hour | 비중 | 목적 |
|---|---:|---:|---|
| 256 x 256 pre-training | 147.5K | 약 47% | 기본 시각/언어 지식 학습 |
| Omni pre-training | 142.5K | 약 45% | 고해상도, T2I/I2I joint 학습 |
| post-training | 24K | 약 8% | SFT, distillation, RLHF |
| 합계 | 314K | 100% | 전체 학습 비용 |

주의할 점:

```text
314K는 학습 이미지 장 수가 아니다.
314K H800 GPU hour, 즉 GPU 시간이 314,000시간이라는 뜻이다.
```

### 8.2 왜 저해상도에 compute를 많이 쓰나

처음부터 고해상도만 학습하면 비용이 큽니다.

저해상도에서는 다음을 싸게 배울 수 있습니다.

| 저해상도에서 배우기 좋은 것 | 예시 |
|---|---|
| 일반 물체 개념 | 사람, 동물, 음식, 건물 |
| 언어-이미지 정렬 | prompt의 주체/행동/속성 |
| 중국어/영어 텍스트 렌더링 | 이미지 안 글자 생성 |
| 기본 composition | 어디에 무엇을 배치할지 |

즉, 저해상도 단계는 "기초 체력"을 싸게 쌓는 단계입니다.

### 8.3 Omni pre-training이란 무엇인가

저해상도 256² 단계가 끝난 뒤 들어가는 본격 사전학습 단계입니다. 142.5K H800 hour, 전체 비용의 약 45%가 여기 들어갑니다.

"Omni"는 모든 것을 한꺼번에 학습한다는 뜻입니다. 한 단계 안에서 세 가지를 동시에 학습합니다.

| 무엇을 한꺼번에 | 의미 |
|---|---|
| **Arbitrary-Resolution Training (임의 해상도 학습)** | 원본 해상도를 그대로 두고 다양한 해상도/종횡비 (aspect ratio)로 학습. 1k–1.5k 까지 자연스럽게 커버. |
| **Joint T2I + I2I Training (텍스트→이미지 + 이미지→이미지 통합)** | 텍스트→이미지 (T2I)와 이미지→이미지 (I2I, 자세히 → 8.4) 작업을 비율 4 : 1로 같은 배치에 섞어 학습. |
| **Multi-level + Bilingual Caption** | Z-Captioner의 5종 캡션을 모두 사용 + 원본 alt-text 일부 + 중국어/영어 둘 다. |

이전에 어떤 일을 했는지와 비교하면 출발점이 다릅니다.

```text
기존 방식 (FLUX, SDXL 등):
  256² 학습 → 512² 학습 → 768² 학습 → 1024² 학습
        → T2I SFT → I2I 추가 학습 → bilingual 적응
  단계마다 별도 compute 예산, 합치면 매우 큼

Z-Image의 Omni 방식:
  저해상도 256² (47%) → Omni pre-training 한 단계 (45%) → post-train (8%)
  모든 종류의 학습을 같은 가중치에 한꺼번에 흘림
  → 별도 단계 비용 제거
```

핵심 통찰은 단순합니다.

```text
같은 가중치가 어차피 갱신될 거라면,
여러 단계로 나눠 비싼 학습을 여러 번 돌리지 말고
한 번에 다 보여주자.
```

해상도가 제각각이라 단순 배치는 padding 낭비가 큽니다. 이를 해결하는 시스템 효율 트릭:

| 트릭 | 의미 |
|---|---|
| Sequence-length-aware batching | 비슷한 시퀀스 길이끼리 묶어 padding 낭비를 줄임 |
| Dynamic batch sizing | 긴 시퀀스는 작은 배치 (OOM 방지), 짧은 시퀀스는 큰 배치 (자원 빈자리 방지) |
| FSDP2 (Fully Sharded Data Parallel v2) | 가중치 + optimizer 상태를 GPU들에 쪼개서 보관 |
| Gradient checkpointing | 중간 활성값을 저장 안 하고 역전파 때 다시 계산 |
| torch.compile | DiT 블록을 JIT 컴파일해 CUDA 커널 효율 ↑ |

Omni가 끝나면 하나의 베이스 모델 (Z-Image-Omni-Base)이 나오고, 거기서 두 갈래로 분기합니다.

```text
                 Z-Image-Omni-Base
            (1k–1.5k 임의 해상도, T2I + I2I 모두 가능)
                       │
        ┌──────────────┴───────────────┐
        ↓                                ↓
   T2I 본체 분기                    Edit 분기
   SFT → Distill → RLHF →           Continued PT → SFT →
   Z-Image / Z-Image-Turbo           Z-Image-Edit
```

이게 Z-Image-Edit가 GEdit-CN에서 강한 근본 이유 — 편집을 *나중에 덧붙인 게* 아니라 *처음부터 알고 있던* 모델이라는 점.

### 8.4 I2I training이란 무엇인가

I2I = Image-to-Image, 입력에 이미지가 끼어 있는 모든 생성 작업의 통칭입니다. T2I와 데이터 구조부터 다릅니다.

| | T2I (Text-to-Image) | I2I (Image-to-Image) |
|---|---|---|
| 입력 | 텍스트 prompt 1개 | **이미지 1장 + 선택적 텍스트 (instruction)** |
| 출력 | 이미지 1장 | 이미지 1장 |
| 학습 데이터 단위 | `(prompt, image)` 쌍 | **`(input image, target image, instruction)` 세 개** |
| 학습 신호 | "이 텍스트를 그리면 이런 이미지" | **"이 이미지를 *이렇게 바꾸면* 저 이미지"** |
| 모델이 배우는 것 | 텍스트 → 픽셀 매핑 | 두 이미지 사이의 *변화 패턴* |

#### I2I의 종류

I2I는 큰 범주이고 안에 여러 작업이 있습니다.

| 종류 | 한 줄 |
|---|---|
| 명령어 기반 편집 (instruction-based editing) | "배경을 시드니 오페라하우스로 바꿔라" 식. **Z-Image-Edit의 주된 작업.** |
| 인페인팅 (inpainting) | 일부 영역을 지우고 다시 그리기. mask 사용. |
| 아웃페인팅 (outpainting) | 이미지 바깥쪽으로 확장. |
| 스타일 변환 (style transfer) | "이 사진을 반 고흐 스타일로". |
| 슈퍼 레졸루션 (super-resolution) | 저해상도 → 고해상도. |
| 레퍼런스 기반 생성 (reference-guided) | "이 인물을 기반으로 새 장면을". IP-Adapter, DreamBooth 류. |
| 가상 입어보기 (VTON, Virtual Try-On) | 옷을 다른 사람에게 입혀보기. |
| ControlNet 류 | depth/edge/pose 같은 제어 신호를 받아 그림 생성. |

Z-Image 맥락에서는 주로 **명령어 기반 편집**을 다룹니다.

#### I2I를 시퀀스에 어떻게 끼워넣나

영리한 점 — *I2I용 별도 모델 구조가 없음*. 같은 S3-DiT 시퀀스에 이미지를 하나 더 끼워넣는 방식.

```text
T2I 사전학습 한 샘플의 시퀀스:
  [텍스트 토큰 ... ] + [노이즈 이미지 토큰 ...]
       ↑ Qwen3 hidden states         ↑ 학습 대상

I2I 사전학습 한 샘플의 시퀀스:
  [텍스트 토큰 ...] + [참조 이미지 토큰 ...] + [노이즈 이미지 토큰 ...]
       ↑ instruction         ↑ 깨끗한 참조 (t=1)   ↑ 학습 대상 (t ∈ [0,1])
```

같은 attention 안에서 두 이미지를 구분하는 방법:

| 방법 | 어떻게 작동 |
|---|---|
| 시간 인코딩 (timestep)으로 구분 | 참조는 t=1 (깨끗함), 학습 대상은 t ∈ [0,1] (노이즈 섞임). 모델이 자동 인지. |
| 3D RoPE의 temporal axis로 분리 | 참조와 학습 대상이 같은 공간 좌표 (h, w)지만 시간 축에서 단위 간격 떨어진 위치. |

즉 모델 구조를 *전혀 안 바꾸고* 데이터 형식만 바꿔서 I2I를 학습합니다.

#### I2I 데이터 — 어디서 구하나

자연에서 "편집 전/후 + 명령어"가 다 붙어 있는 경우는 거의 없습니다. Z-Image는 단계별로 다른 출처를 씁니다.

| 단계 | 출처 | 양/질 |
|---|---|---|
| **사전학습용** | 자연 발생 약한 정렬 (weakly aligned) 쌍 — 같은 상품 다양한 각도, 같은 인물 여러 사진 | **양 많음**, 정렬 느슨함 |
| **SFT용 ①** | Mixed Editing + Graphical Representation — 원본 1장 → 전문 모델로 N가지 편집 → N² 페어 폭증 | 질 높음 |
| **SFT용 ②** | Paired Images from Videos — 비디오 프레임 + CN-CLIP 유사도 필터 | 자연스러운 변화 |
| **SFT용 ③** | Rendering for Text Editing — 텍스트만 다른 합성 페어 | 정밀 텍스트 편집용 |

#### 왜 I2I를 사전학습부터 끼워넣었나

T2I 사전학습이 끝난 뒤 I2I를 추가 학습하는 기존 방식과 출발점이 다릅니다.

```text
기존:
  T2I 사전학습 (수개월) → 가중치 동결 → I2I 추가학습 (또 수개월)
  같은 노력을 두 번

Z-Image:
  사전학습 한 단계에 T2I:I2I = 4:1 비율로 같이 학습
  같은 가중치가 두 작업 신호를 동시에 받음
  → 편집 능력의 강한 초기화를 무료로 획득
```

왜 작동하나 — 두 가지 이유:

| 이유 | 설명 |
|---|---|
| 시각 표현 공유 | T2I와 I2I 모두 "텍스트 + 이미지 토큰을 다루는" 작업이라 시각 표현이 공유됨. 한 가중치가 둘 다 배워도 충돌이 적음. 저자가 명시: *"T2I 성능 저하 없음"*. |
| 상호 보완 | T2I는 *어떤 그림이 자연스러운지*를 배우고, I2I는 *변화 패턴*을 배움 → 두 신호가 서로 보완해 일반화 능력 ↑. |

이 덕분에 얻어진 능력:

- Z-Image-Edit가 GEdit-CN에서 Qwen-Edit2509와 공동 2위.
- 여러 이미지 입력 처리가 가능 → ControlNet, IP-Adapter 같은 확장에 유리.
- Z-Image-Omni-Base가 "raw"한 출발점으로 커뮤니티 fine-tuning에 적합.

### 8.5 SFT의 역할

SFT는 모델의 분포를 좁히는 단계입니다.

```text
pre-training:
  넓고 다양한 웹 분포를 배움

SFT:
  더 보기 좋고, 지시를 잘 따르는 분포로 이동
```

Z-Image는 SFT에서 세 가지를 사용합니다.

| 방법 | 의미 |
|---|---|
| Distribution Narrowing | 고품질 결과가 많이 나오는 방향으로 분포를 좁힘 |
| Tagged Resampling | 드문 개념이 잊히지 않도록 더 자주 샘플링 |
| Model Merging | 여러 SFT 모델을 가중치 평균해 장점을 섞음 |

---

## 9. Decoupled DMD와 Z-Image-Turbo

이 절은 DMD, DMD2 문서와 연결해서 읽으면 가장 쉽습니다.

### 9.1 목표

일반 Z-Image는 50 step으로 이미지를 만듭니다.

Turbo는 이것을 8 step으로 줄입니다.

```text
teacher:
  50 step, CFG 사용, 느리지만 품질 좋음

student:
  8 step, CFG 미사용, 빠르지만 처음에는 품질 부족
```

목표는:

```text
student의 8-step 결과 분포 ~= teacher의 고품질 결과 분포
```

입니다.

### 9.2 DMD 복습

DMD의 기본 아이디어는 아래와 같습니다.

```text
student output
  -> 다시 noise를 섞어 x_t 생성
  -> teacher가 s_real 제공
  -> critic이 s_fake 제공
  -> 두 score의 차이로 student 업데이트
```

기호를 일반 텍스트로 쓰면:

```text
p_real       = teacher 또는 real image 분포
p_fake_theta = 현재 student가 만드는 image 분포
s_real       = p_real 쪽으로 가는 score
s_fake       = p_fake_theta 쪽으로 가는 score
```

직관은 이것입니다.

```text
teacher:
  "이 방향이 더 진짜 이미지답다."

critic:
  "현재 student는 이 방향으로 자주 몰린다."

DMD:
  두 방향의 차이를 보고 student를 real 분포 쪽으로 이동시킨다.
```

### 9.3 Z-Image의 관찰: DMD 안에는 두 효과가 섞여 있다

Z-Image 저자는 기존 DMD를 분석하면서 두 가지 효과가 섞여 있다고 봅니다.

| 효과 | 역할 |
|---|---|
| CA, CFG-Augmentation | few-step 능력을 끌어올리는 주 엔진 |
| DM, Distribution Matching | 분포를 안정화하고 아티팩트를 줄이는 정규화 항 |

여기서 중요한 점은 저자의 해석입니다.

> **기존에는 DM이 핵심 엔진처럼 보였지만, 실제 few-step 성능 향상에는 CA가 더 큰 역할을 한다.**

그래서 Z-Image는 둘을 분리합니다.

### 9.4 Decoupled DMD

Decoupled DMD는 CA와 DM을 같은 방식으로 다루지 않습니다.

```text
기존 DMD:
  CA와 DM을 묶어서 같은 re-noising schedule로 처리

Decoupled DMD:
  CA와 DM을 분리하고 서로 다른 re-noising schedule 적용
```

왜 중요할까요?

few-step student는 timestep을 매우 거칠게 밟습니다. 이때 noise를 다시 섞는 방식이 잘못되면:

```text
디테일이 뭉개짐
색이 어긋남
채도가 튐
texture가 불안정
```

같은 문제가 생깁니다.

Decoupled DMD는 CA와 DM의 역할을 나눠서 이 문제를 줄입니다.

### 9.5 DMDR

DMDR은 **DMD meets RL**로 이해하면 됩니다.

일반 텍스트 수식:

```text
L_DMDR = L_DM + lambda * L_RL
```

| 항 | 역할 |
|---|---|
| `L_DM` | teacher/real 분포에서 멀어지지 않게 잡아주는 항 |
| `L_RL` | reward가 좋아지는 방향으로 student를 미는 항 |
| `lambda` | RL 항의 세기 |

핵심은 `L_DM`이 안전벨트 역할을 한다는 점입니다.

일반 RLHF에서는 reward를 너무 세게 밀면 reward hacking이 생길 수 있습니다.

```text
reward model이 좋아하는 꼼수만 배움
  -> 실제 이미지 품질은 나빠짐
```

DMDR에서는 `L_DM`이 계속 teacher/real 분포 쪽으로 붙잡습니다.

```text
RL:
  reward를 올려라

DM:
  그래도 teacher 분포에서 너무 멀어지지 마라
```

그래서 reference KL 같은 제약을 DM 항이 더 강하게 대신하는 구조로 볼 수 있습니다.

---

## 10. RLHF: 증류 후 정렬

Z-Image에서는 RL이 두 번 나옵니다.

1. Distillation 안의 RL: DMDR
2. Distillation 후의 RLHF: DPO와 GRPO

둘은 목적이 다릅니다.

| 항목 | DMDR | 후속 RLHF |
|---|---|---|
| 시점 | 증류 중 | 증류 후 |
| 목적 | 8-step student를 안정적으로 만들기 | 사람 선호도와 지시 준수 개선 |
| 안전장치 | DM 항 | reward model, KL, preference data |
| 성격 | 방어적 RL | 공격적 정렬 |

### 10.1 Reward model

Z-Image의 reward model은 세 축을 봅니다.

| 축 | 의미 |
|---|---|
| instruction-following | prompt를 얼마나 잘 따랐는가 |
| AIGC perception | 생성 이미지로서 자연스럽고 결함이 적은가 |
| aesthetic | 미감이 좋은가 |

instruction-following은 prompt를 더 잘게 쪼개서 봅니다.

```text
subject
attribute
action
space
style
```

즉, prompt 전체를 한 점수로만 보지 않고 어떤 요소를 놓쳤는지 확인합니다.

### 10.2 Stage 1: offline DPO

DPO는 chosen/rejected pair를 사용합니다.

```text
same prompt
  -> better image: chosen
  -> worse image: rejected
```

Z-Image는 VLM이 후보 pair를 만들고 사람이 검증하는 방식으로 preference data를 구성합니다.

Stage 1은 객관적인 항목에 집중합니다.

| 항목 | 예시 |
|---|---|
| 텍스트 렌더링 | 이미지 안 글자를 제대로 썼는가 |
| 객체 수 | "three apples"가 진짜 3개인가 |
| 명확한 속성 | 빨간 옷, 왼쪽 위치, 특정 동작 |

### 10.3 Stage 2: online GRPO

GRPO는 현재 모델이 직접 여러 샘플을 만들고, 그 그룹 안에서 상대적으로 더 나은 샘플을 학습합니다.

```text
prompt 하나
  -> sample 1
  -> sample 2
  -> sample 3
  -> sample 4
  -> 그룹 안에서 상대 점수 계산
```

Stage 2는 주관적인 품질까지 더 밀어 올립니다.

| 신호 | 의미 |
|---|---|
| realism | 실제처럼 보이는가 |
| aesthetic | 보기 좋은가 |
| instruction | prompt를 잘 따르는가 |

---

## 11. 공개 코드 기준 추론 흐름

공개 코드는 학습 코드가 아니라 추론 코드입니다.

따라서 확인할 수 있는 것은:

```text
모델 구조
prompt 처리
sampling loop
VAE decoding
attention backend
model loading
```

확인할 수 없는 것은:

```text
데이터 큐레이션 코드
DMD/DMDR 학습 코드
RLHF 학습 코드
reward model
학습 데이터
```

### 11.1 prompt에서 이미지까지

```text
1. 사용자가 prompt 입력

2. Qwen3-4B가 prompt를 token으로 읽음
   -> hidden_states[-2] 추출
   -> text feature 생성

3. random latent 생성
   -> shape 예: B x 16 x 128 x 128

4. scheduler가 timestep 8개 또는 50개 준비

5. 각 timestep에서 S3-DiT 실행
   -> text feature + current latent + timestep 입력
   -> velocity 예측

6. scheduler가 latent 업데이트
   -> latent = latent + dt * velocity

7. 마지막 latent를 Flux VAE로 decode
   -> RGB image
```

### 11.2 Turbo의 8-step 흐름

Turbo를 단순화하면 다음과 같습니다.

```text
noise latent
  -> step 1: S3-DiT predicts velocity
  -> step 2: S3-DiT predicts velocity
  -> ...
  -> step 8: S3-DiT predicts velocity
  -> VAE decode
  -> image
```

Z-Image-Turbo는 CFG를 쓰지 않습니다.

이 말은:

```text
일반 CFG:
  conditional forward + unconditional forward
  -> 두 결과를 섞음

Turbo:
  conditional forward만 사용
```

따라서 step 수만 줄어든 것이 아니라, step당 모델 호출도 줄어듭니다.

### 11.3 CFG truncation과 normalization

일반 Z-Image는 CFG를 사용합니다. 다만 CFG를 계속 강하게 쓰면 채도 폭주나 색 번짐이 생길 수 있습니다.

그래서 코드에는 두 트릭이 있습니다.

| 트릭 | 의미 |
|---|---|
| CFG truncation | 후반 step에서는 CFG를 끄거나 줄여 모델 호출과 색 문제를 줄임 |
| CFG normalization | CFG 결과의 norm이 너무 커지면 크기를 제한 |

쉽게 말하면:

```text
초반:
  prompt 방향을 강하게 잡는다.

후반:
  너무 세게 밀지 않고 자연스럽게 정리한다.
```

### 11.4 코드에서 논문 기여가 보이는 위치

공개 repo 기준으로는 아래처럼 나눠 볼 수 있습니다.

| 영역 | 보이는 것 | 논문 기여 여부 |
|---|---|---|
| transformer | S3-DiT 구조 | 핵심 기여 |
| pipeline | prompt 처리와 sampling loop | 추론 구현 |
| scheduler | FlowMatch Euler update | 표준적 구성 |
| autoencoder | Flux VAE decode | 외부 VAE 활용 |
| attention | FlashAttention/SDPA backend | 시스템 최적화 |
| loader | HF weight loading | 추론 편의 기능 |

가장 중요한 해석:

> **Z-Image의 학습 알고리즘 기여는 공개 코드보다 공개 가중치 안에 더 많이 들어 있다.**

---

## 12. 실험 결과 요약

### 12.1 학습 비용

| 단계 | H800 GPU hour | 비용 추정 |
|---|---:|---:|
| 256 x 256 pre-training | 147.5K | 약 295K USD |
| Omni pre-training | 142.5K | 약 285K USD |
| post-training | 24K | 약 48K USD |
| 합계 | 314K | 약 628K USD |

동시 GPU 수에 따라 실제 기간은 달라집니다.

| 동시 H800 수 | 314K H800 hour를 쓰는 데 걸리는 시간 |
|---:|---:|
| 128 | 약 102일 |
| 256 | 약 51일 |
| 512 | 약 26일 |
| 1024 | 약 13일 |

### 12.2 인간 선호도

| 평가 | 결과 |
|---|---|
| Artificial Analysis Arena | 전체 8위, 오픈소스 1위, Elo 1161 |
| Alibaba AI Arena | 전체 4위, 오픈소스 1위 |
| FLUX.2 dev 대비 | Good + Same 87.4% |

Z-Image의 주장은 단순히 "최고 성능"이 아닙니다.

더 정확히는:

> **훨씬 작은 파라미터와 낮은 비용으로 상위권 품질에 도달했다.**

### 12.3 자동 벤치마크

| Benchmark | Z-Image | Z-Image-Turbo | 의미 |
|---|---:|---:|---|
| CVTG-2K Word Acc | 0.8671 | 0.8585 | 영어 텍스트 렌더링 |
| LongText-Bench-ZH | 0.936 | 0.926 | 중국어 긴 텍스트 |
| OneIG-EN Overall | 0.546 | 0.528 | 영어 instruction generation |
| GenEval | 0.84 | 0.82 | 객체/속성/관계 |
| DPG-Bench | 88.14 | 84.86 | 복잡한 prompt |
| PRISM-EN | 75.6 | 77.4 | 사람 선호 기반 평가 |
| PRISM-ZH | 75.3 | 75.1 | 중국어 선호 기반 평가 |

Turbo는 8-step 모델인데도 50-step 모델과 가까운 성능을 냅니다. 이것이 Decoupled DMD와 RLHF가 중요한 이유입니다.

### 12.4 편집 모델

| Benchmark | Z-Image-Edit |
|---|---:|
| ImgEdit | 4.30 |
| GEdit-EN | 7.57 |
| GEdit-CN | 7.54 |

---

## 13. Z-Image의 핵심 기여를 다시 정리

Z-Image의 기여는 "새 layer 하나"가 아니라 전체 설계입니다.

| 기여 | 쉬운 설명 |
|---|---|
| S3-DiT | text와 image를 한 줄로 처리하는 6B Single-Stream DiT |
| 데이터 큐레이션 | 모델이 못 하는 개념을 찾아 데이터를 보강하는 closed loop |
| 저해상도 집중 학습 | 비싼 고해상도 전에 기초 지식을 싸게 학습 |
| PE-aware SFT | prompt enhancer와 SFT로 world knowledge와 instruction following 보강 |
| Decoupled DMD | CA와 DM을 분리해 few-step 증류 품질 개선 |
| DMDR | distillation과 RL을 결합해 reward hacking을 줄임 |
| 2-stage RLHF | DPO로 객관 항목, GRPO로 주관 품질을 따로 개선 |
| 시스템 최적화 | FSDP2, torch.compile, dynamic batching, FlashAttention 활용 |

한 줄로 요약하면:

```text
작은 모델을 크게 보이게 만든 것이 아니라,
작은 모델이 배워야 할 것을 아주 낭비 없이 배치했다.
```

---

## 14. DMD/DMD2와의 연결

Z-Image 문서에서 가장 헷갈리기 쉬운 부분은 Decoupled DMD입니다.

DMD/DMD2 문서와 이어 보면 이렇게 정리됩니다.

| 방법 | 핵심 |
|---|---|
| DMD | teacher score와 fake score의 차이로 1-step student를 학습 |
| DMD2 | DMD의 비싼 regression loss를 줄이고 critic 학습, GAN loss, backward simulation 등으로 안정화 |
| Z-Image Decoupled DMD | DMD 안의 CA와 DM 효과를 분리해 8-step Turbo 품질을 개선 |
| Z-Image DMDR | DM loss와 RL loss를 결합해 reward 향상과 분포 안정성을 동시에 추구 |

따라서 Z-Image의 증류는 "DMD를 그대로 썼다"라기보다:

```text
DMD 계열의 분포 매칭 아이디어를 가져오되,
few-step 이미지 품질을 위해 CA와 DM을 분리하고,
RL까지 결합한 변형
```

으로 보는 것이 좋습니다.

---

## 15. 자주 헷갈리는 질문

### Q1. Z-Image는 왜 6B인데 큰 모델과 경쟁할 수 있나?

한 가지 이유 때문이 아닙니다.

```text
Single-Stream 구조
  + Qwen3-4B 텍스트 이해
  + active data curation
  + 저해상도 집중 pre-training
  + SFT
  + Decoupled DMD
  + RLHF
```

가 같이 작동합니다.

가장 중요한 해석은:

> **6B DiT가 모든 것을 혼자 배우는 구조가 아니다. 텍스트 이해, 데이터 품질, 증류, 정렬이 6B의 부담을 나눠 갖는다.**

### Q2. Single-Stream은 무조건 Dual-Stream보다 좋은가?

아닙니다.

Single-Stream은 파라미터 효율이 좋고 text-image alignment에 유리할 수 있습니다. 하지만 학습 안정성이 더 어렵습니다.

그래서 Z-Image는 refiner, QK-Norm, Sandwich-Norm, 3D RoPE 같은 안정화 장치를 많이 넣었습니다.

정리하면:

```text
Dual-Stream:
  안정적이고 모달리티별 전문화가 쉬움

Single-Stream:
  파라미터 효율과 cross-modal 공유가 좋지만 학습 설계가 중요
```

### Q3. Turbo는 왜 LoRA나 SFT에 적합하지 않나?

Turbo는 8-step으로 동작하도록 증류된 모델입니다.

일반 LoRA/SFT는 보통 원래 diffusion training처럼 더 촘촘한 timestep 분포를 가정합니다. 이 가정이 Turbo의 8-step 경로와 맞지 않을 수 있습니다.

```text
Turbo가 배운 것:
  8-step으로 빠르게 가는 짧은 경로

일반 SFT/LoRA가 흔드는 것:
  넓고 촘촘한 noise schedule
```

그래서 Turbo의 빠른 경로가 깨질 수 있습니다.

공식 권장도 비슷합니다.

```text
fine-tuning 또는 LoRA:
  Z-Image 또는 Z-Image-Omni-Base 권장

Z-Image-Turbo:
  빠른 추론용으로 사용
```

### Q4. 그런데 Turbo에 RLHF는 했는데 왜 괜찮나?

핵심은 학습 방식이 다르기 때문입니다.

| 방식 | 데이터 출처 | Turbo schedule과의 관계 |
|---|---|---|
| 일반 SFT/LoRA | 외부 real image dataset | 8-step 경로와 어긋날 수 있음 |
| on-policy RLHF | 현재 Turbo가 직접 만든 sample | 8-step 경로 안에서 학습 |

RLHF는 현재 모델이 실제로 8-step으로 만든 결과를 보고 학습합니다. 그래서 Turbo가 쓰는 경로를 덜 깨뜨립니다.

DMDR에서는 여기에 DM 항까지 붙어 있어 teacher 분포에서 너무 멀어지지 않게 잡아줍니다.

### Q5. 314K는 학습 이미지 수인가?

아닙니다.

```text
314K H800 GPU hour
  = H800 GPU 1장이 314,000시간 일한 양
```

학습 이미지 수, 최종 curated dataset 크기, 전체 training iteration 수는 공개되지 않았습니다.

### Q6. 공개 코드로 논문 전체를 검증할 수 있나?

부분적으로만 가능합니다.

검증 가능한 것:

```text
S3-DiT 구조
추론 pipeline
scheduler
VAE decode
attention backend
가중치 로딩
```

검증하기 어려운 것:

```text
데이터 큐레이션
Decoupled DMD 학습
DMDR
DPO/GRPO RLHF
reward model
학습 데이터 규모
```

따라서 논문의 핵심 학습 기여는 코드 재현보다는 공개 가중치, 논문 ablation, 정성/정량 결과로 판단해야 합니다.

### Q7. Z-Image에서 제일 중요한 아이디어 하나만 고르면?

하나만 고르면 **효율을 모델 크기 밖에서 만든다**입니다.

```text
모델을 키우는 대신:
  텍스트 이해는 LLM에게 맡김
  데이터는 active curation으로 정제
  구조는 single-stream으로 공유
  학습은 curriculum으로 배치
  추론은 distillation으로 압축
```

이 모든 선택이 6B라는 작은 본체를 중심으로 정렬되어 있습니다.

### Q8. DreamLite와 비교하면 모델 차이가 뭔가?

*핵심 질문: 같은 시기에 나온 다른 "효율" 모델 [[paper_dreamlite]] 과 본 논문은 표제만 비슷한데 실제 설계는 어떻게 다른가.*

**한 줄로**: Z-Image는 **"서버사이드 SOTA를 효율적으로 만들자"** (6B로 20–80B 따라잡기), DreamLite는 **"온디바이스에서 처음으로 통합 모델을 가능하게 하자"** (0.39B로 폰에서 1초 미만). 체급이 약 15배 차이라 직접 벤치 비교 의미 없음.

**핵심 대비 한눈에**:

| 측면 | Z-Image | DreamLite |
|---|---|---|
| 타겟 | 서버 GPU (H800) | **스마트폰** |
| 백본 | Single-Stream DiT (6.15B, 30층) | **Pruned SDXL UNet** (0.39B) |
| 텍스트 인코더 | Qwen3-4B + SigLIP 2 (편집용 분리) | **Qwen3-VL-2B 하나로 통합** (텍스트+이미지) |
| VAE | Flux VAE (16ch) | **TinyVAE** (폰 22ms) |
| Conditioning | **시퀀스 concat (1D)** + 3D RoPE | **공간 concat (2D)** + 태스크 토큰 |
| Distillation | **Decoupled DMD** (자체) — 50→8 step | **DMD2** (기존) — 28→4 step |
| RL | **DMDR + 2-stage RLHF** (본격 on-policy) | **ReFL** (가벼움, 베이스라인 위만) |
| 학습 트릭 | **Active Data Curation** (4-모듈 인프라) | **Foreground-Emphasis Mask + 3단계 커리큘럼** |
| 무기 | **알고리즘 노벨티** | **시스템 엔지니어링** |

**공통점**: 단일 모델 통합, Qwen 계열 텍스트 인코더, DMD 계열 distillation, CFG-free 변종, 사람 선호 RL — 표제는 같음.

→ **상세 비교**는 [[paper_dreamlite]] 본체 ([PAPER_DreamLite.md Q5](PAPER_DreamLite.md)) 에 정리. 본 절은 짧은 포인터.

### Q9. Z-Image의 고품질 SFT curated data는 공개됐나?

아닙니다. **Z-Image의 SFT curated data 자체는 공개되지 않았습니다.**

공개된 것은 모델 가중치, 추론 코드, 모델 카드, 리포트/논문 설명 정도입니다.

```text
공개됨:
  모델 가중치
  추론 코드
  모델 카드 / 리포트 / 논문 설명

비공개:
  SFT curated dataset 원본
  이미지-캡션 pair 목록
  최종 SFT 데이터 크기
  source breakdown
  Prompt Enhancer 학습 데이터
  reward model / preference data
  data curation tooling 전체
```

다만 데이터 구성 방식에 대한 설명은 있습니다.

| 항목 | 논문/리포트에서 설명된 수준 |
|---|---|
| 원천 | large-scale internal copyrighted image-text collections |
| 필터링 | 해상도, aspect ratio, pHash, 압축 흔적, 품질 점수, 정보량 기준 |
| 안전/품질 | NSFW 제거, AI-generated content 필터, aesthetic model 사용 |
| 정렬성 | CN-CLIP으로 image-text alignment 낮은 pair 제거 |
| 캡션 | tag, short, medium, long, simulated user prompt 등 5종 caption |
| OCR | long caption에 visible text/OCR 정보 포함 |
| 지식 그래프 | Wikipedia entity + internal tag로 concept graph 구성 |
| 샘플링 | BM25 rarity score와 graph 기반 balanced sampling |
| SFT 목적 | high-fidelity output 쪽으로 distribution narrowing |
| long-tail 보호 | tagged resampling으로 catastrophic forgetting 방지 |
| 마무리 | model merging으로 여러 능력 간 Pareto-optimal 지점 탐색 |

따라서 문서에는 이렇게 쓰는 것이 가장 정확합니다.

```text
Z-Image의 SFT curated data는 비공개다.
논문은 데이터 원본, 샘플 수, 실제 pair는 공개하지 않는다.
대신 고품질 필터링, grounded caption, tagged resampling,
BM25 기반 rarity balancing, model merging 같은 구성 원리는 설명한다.
```

### Q10. 그럼 비슷한 공개 데이터셋은 무엇이 있나?

Z-Image SFT 데이터와 완전히 같은 수준의 내부 curated dataset이 공개된 것은 아니지만, **비슷한 목적의 공개 데이터셋**은 있습니다.

| 데이터셋 | 실제 데이터 링크 | 공개 여부 | 규모 | Z-Image SFT와의 유사도 | 장점 | 주의점 |
|---|---|---:|---:|---|---|---|
| **Fine-T2I** | [HF dataset](https://huggingface.co/datasets/ma-xu/fine-t2i), [arXiv](https://arxiv.org/abs/2602.09439) | 공개 | 6M+ pair, 약 2TB | 매우 높음 | T2I fine-tuning용 대규모 고품질 curated 데이터. Z-Image/FLUX2 등으로 만든 synthetic data + 전문 사진 기반 real data 포함. original/enhanced prompt 제공 | synthetic 비중이 큼. Z-Image가 의도적으로 피한 proprietary/synthetic distillation 계열과 철학이 다를 수 있음 |
| **Alchemist** | [HF dataset](https://huggingface.co/datasets/yandex/alchemist), [paper](https://arxiv.org/abs/2505.19297) | 공개 | 3,350 samples | 매우 높음 | 작지만 SFT 효과를 노린 초고품질 curated 데이터. diffusion model 내부 activation으로 좋은 SFT 샘플을 고르고 VLM recaption 사용 | 규모가 작음. 대규모 general model 학습보다는 SFT 효과 검증/소규모 개선에 가까움 |
| **LAION-Aesthetics** | [dataset page](https://projects.laion.ai/laion-datasets/laion-aesthetic.html), [LAION datasets](https://projects.laion.ai/laion-datasets/) | 공개 | score threshold별 수백만~억 단위 | 중간 | aesthetic score 기반 공개 image-text pool. LAION-Art(score > 8), LAION-Aesthetic(score > 7) 등으로 필터링 가능 | caption 품질과 alignment가 Z-Image식 SFT 데이터보다 거칠다. 추가 filtering/recaptioning 필요 |
| **JourneyDB** | [official repo](https://github.com/JourneyDB/JourneyDB), [HF mirror](https://huggingface.co/datasets/bitmind/JourneyDB), [arXiv](https://arxiv.org/abs/2307.00716) | 공개 | 4M generated image-prompt | 중간 | 고품질 generated image와 prompt 쌍. style/prompt 연구와 generated image understanding에 유용 | 공식 데이터는 Terms of Usage와 신청 폼 기반. HF mirror는 일부 mirror일 수 있음. 생성 모델 편향이 들어 있음 |
| **DiffusionDB** | [HF dataset](https://huggingface.co/datasets/poloclub/diffusiondb), [project page](https://anonacl.github.io/diffusiondb/), [arXiv](https://arxiv.org/abs/2210.14896) | 공개 | 14M image-prompt | 낮음~중간 | Stable Diffusion Discord 기반 대규모 real user prompt-image 로그. CC0로 설명됨 | 고품질 curated SFT 데이터가 아니라 사용 로그에 가깝다. 노이즈, 편향, prompt 품질 차이가 큼 |
| **ShareGPT4V** | [project/data page](https://sharegpt4v.github.io/), [HF org](https://huggingface.co/Lin-Chen) | 공개 | 100K GPT-4V captions / 1.2M generated captions | 보조적으로 높음 | 매우 자세한 image caption 데이터. object property, spatial relation, world knowledge, aesthetic description 포함 | T2I SFT용 image-prompt pair라기보다 captioner/LMM 학습용. 원천 이미지와 라이선스 확인 필요 |
| **SFHQ-T2I** | [GitHub](https://github.com/SelfishGene/SFHQ-T2I-dataset), [Kaggle](https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models) | 공개 | 122K face image-prompt | 도메인 한정 | 고품질 1024x1024 synthetic face image와 prompt. 얼굴 도메인에서는 품질과 다양성이 좋음 | 얼굴 전용이라 general T2I SFT에는 좁다. 여러 생성 모델로 만든 synthetic 데이터 |

가장 가까운 공개 대안만 고르면:

```text
1순위: Fine-T2I
  대규모, T2I fine-tuning 목적, 고품질 filtering, enhanced prompt 포함

2순위: Alchemist
  작지만 SFT용 고품질 curated dataset 철학이 가장 선명함

보조 후보:
  LAION-Aesthetics + ShareGPT4V captioning
  JourneyDB / DiffusionDB를 재필터링해서 사용
```

Z-Image식으로 공개 데이터를 만들려면 단순히 위 데이터셋을 그대로 쓰기보다는 아래 과정이 필요합니다.

```text
1. 원천 후보 수집
2. aesthetic / quality / safety / watermark / alignment 필터링
3. VLM 또는 captioner로 grounded recaption
4. OCR과 visible text를 caption에 포함
5. concept tag 부여
6. rare concept을 tagged resampling으로 보존
7. 여러 SFT checkpoint를 비교하거나 model merging
```

즉:

```text
Fine-T2I, Alchemist는 "이미 SFT용 curated data"에 가깝고,
LAION-Aesthetics, JourneyDB, DiffusionDB는 "SFT 데이터로 만들 수 있는 원천 후보"에 가깝다.
```

---

## 16. 최종 요약

Z-Image는 단순히 "6B짜리 이미지 생성 모델"이 아닙니다.

더 정확한 정의는 다음과 같습니다.

> **Z-Image는 real-world 데이터 큐레이션, Single-Stream DiT, LLM 텍스트 인코더, 단계별 학습 커리큘럼, Decoupled DMD, DMDR, 2-stage RLHF를 하나로 묶어 6B 모델의 효율을 극대화한 이미지 생성 시스템이다.**

가장 기억해야 할 흐름:

```text
1. Qwen3-4B가 prompt를 이해한다.
2. S3-DiT가 text token과 image token을 한 줄로 처리한다.
3. active curation이 작은 모델에 필요한 데이터를 골라준다.
4. SFT가 고품질 분포로 모델을 좁힌다.
5. Decoupled DMD가 50-step teacher를 8-step student로 압축한다.
6. DMDR과 RLHF가 품질과 사람 선호도를 더 맞춘다.
```

그래서 Z-Image의 메시지는 명확합니다.

```text
scale-at-all-costs가 유일한 답은 아니다.
좋은 데이터, 좋은 구조, 좋은 증류가 있으면
6B 모델도 상위권 이미지 생성 모델이 될 수 있다.
```
