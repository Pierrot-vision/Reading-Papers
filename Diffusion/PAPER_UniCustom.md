# UniCustom — Unified Visual Conditioning for Multi-Reference Image Generation

## 메타 정보

| 항목 | 내용 |
|---|---|
| **제목** | UniCustom: Unified Visual Conditioning for Multi-Reference Image Generation |
| **저자** | Yiyan Xu, Qiulin Wang, Wenjie Wang, Yunyao Mao, Xintao Wang, **Pengfei Wan**, **Kun Gai**, Fuli Feng |
| **소속** | Kuaishou Kling 라인업 추정 (Pengfei Wan, Kun Gai 시그니처) |
| **공개일** | 2026-05-12 (arXiv v1, 2026-05-13 수정) |
| **분야** | Multi-Reference Image Generation, VLM-conditioned Diffusion |
| **논문** | [arXiv:2605.12088](https://arxiv.org/abs/2605.12088) · [PDF](https://arxiv.org/pdf/2605.12088) · [HTML](https://arxiv.org/html/2605.12088v1) |
| **코드** | "곧 공개 (will be released soon)" — URL 미공개 |
| **사용 백본** | VLM: **Qwen2.5-VL (얼림)** · DiT: **LongCat-Image-Edit** 초기화 |
| **하드웨어** | H200 GPU 128장, 해상도 512×512 |

---

## 주요 용어 사전 (Glossary)

> 왜 이 절을 두는가? 본문에서 같은 용어가 반복되니, 한 번에 정리해두면 읽기 흐름이 끊기지 않습니다.

### 아키텍처
- **ViT 특성 (F_vit)**: 비전 트랜스포머가 뽑은 의미(semantic) 토큰. 텍스트와 align되어 있어 VLM이 잘 다룸. 차원 ≈ 1280~1536
- **VAE 특성 (F_vae)**: 디퓨전 잠재 공간(latent space) 토큰. 픽셀 복원이 가능할 만큼 외형(appearance)을 보존. 차원 ≈ 16
- **VLM (Vision-Language Model)**: 이미지 토큰과 텍스트 토큰을 함께 처리하는 모델. 이 논문은 Qwen2.5-VL 사용
- **DiT (Diffusion Transformer)**: 잠재 공간 디퓨전을 수행하는 트랜스포머 백본. 이 논문은 LongCat-Image-Edit 사용
- **Slot**: VLM 시퀀스에서 한 참조 이미지가 차지하는 토큰 구간. "Picture k:" 라벨 뒤의 토큰 묶음
- **MRoPE**: Qwen2.5-VL의 multimodal rotary positional embedding

### 핵심 개념
- **Decoupled Visual Conditioning**: 기존 방식 — ViT는 VLM 입력으로, VAE는 DiT 옆구리로 따로 들어가는 두 갈래 구조
- **Unified Visual Conditioning**: 이 논문의 처방 — ViT와 VAE를 VLM에 들어가기 전에 한 토큰으로 합침
- **Grounding**: "1번 사진의 여자" 같은 의미 단계의 짝짓기 (어느 텍스트 단어가 어느 참조 슬롯에 대응)
- **Binding**: 의미 결정 결과를 외형 토큰과 짝짓는 과정 (1번 슬롯 hidden state ↔ 1번 VAE 토큰)
- **Grounding-Binding Gap**: 기존 모델에서 grounding은 VLM에서, binding은 DiT에서 따로 일어나며 발생하는 암묵적 추론 부담

### 실패 모드
- **Attribute Leakage**: 1번 인물의 얼굴에 2번 인물의 옷이 섞이는 식의 속성 유출
- **Cross-Reference Confusion**: 1번 모자가 2번 모자처럼 그려지는 식의 참조 간 혼동

### 평가 / 학습
- **OmniContext**: 멀티 레퍼런스 생성 벤치마크 (Single/Multi × Character/Object 카테고리)
- **MICo-Bench**: Multi-subject Interaction & Composition 벤치마크 (HOI, 분해·재조합 포함)
- **PSNR (Peak Signal-to-Noise Ratio)**: 재구성 품질 지표, dB 단위
- **SFT (Supervised Fine-Tuning)**: 2단계 학습의 미세조정 단계

---

## 논문 요약 (TL;DR)

**한 줄:** ViT(의미)와 VAE(외형) 두 갈래로 따로 다루던 멀티 레퍼런스 컨디셔닝을 VLM 입력 단계에서 선형 결합으로 한 갈래로 묶고, MSE 정규화로 외형 보존을 강제한 작업.

**핵심 문제:** 기존 VLM 기반 디퓨전(OmniGen2, Qwen-Image-Edit, LongCat, FLUX-Kontext, UNO, USO, BAGEL …)은 ViT 특성만 VLM에 넣고 VAE 특성은 DiT 옆구리에 따로 주입. 참조가 둘 이상이면 "1번 의미 ↔ 1번 외형" 짝짓기가 암묵적이라 자주 깨짐 — **속성 유출**, **참조 간 혼동** 발생.

**해결책:** 
1. **Early Fusion** — ViT와 VAE를 VLM 입력 전에 선형 레이어로 합침 (식 1)
2. **Identity-Preserving Init** — 학습 시작 시점에 F_uni ≡ F_vit가 되도록 초기화 (식 2)
3. **Slot-wise Binding Regularization** — 각 슬롯의 VLM hidden state가 원본 VAE를 복원하도록 MSE (식 3)

**검증:** 같은 백본 LongCat 대비 OmniContext +0.58 (7.26 → 7.84), MICo-Bench 거의 2배 (22.78 → 41.71). 단일 이미지 재구성 PSNR ≈ 30dB.

---

## 핵심 기여 (Contributions)

1. **진단**: 기존 decoupled visual conditioning이 만드는 **Grounding-Binding Gap**을 명명하고, 왜 multi-reference에서 암묵적 binding이 깨지는지 분석
2. **아키텍처**: VLM 입력 전 ViT+VAE 조기 융합(early fusion)으로 "의미와 외형이 동시에 살아 있는" hidden state 생성. 단일 선형 레이어 + 항등 초기화로 사전학습 능력 손상 없음
3. **학습 방법론**: 재구성 사전학습 + SFT의 2단계 전략, 그리고 slot-wise binding regularization으로 단일 이미지 PSNR ≈ 30dB 달성 — VAE 디테일이 얼린 VLM을 효과적으로 통과함을 증명

---

## 주요 알고리즘 설명

### 1. 진단 — Grounding-Binding Gap

> 왜 이 절을 두는가? 처방의 의미가 살려면 병명을 정확히 알아야 합니다.

기존 VLM 기반 디퓨전 모델의 데이터 흐름을 시간 순서로 그리면:

```
                  ┌─ F_vit ──[step 1]──> VLM ──[step 2]──> H ──cross-attn──> DiT
Image ─parallel ──┤
                  └─ F_vae ────────────[step 3]──────────────> DiT (side injection)
```

- **step 1**: ViT가 VLM에 들어가 모든 grounding 결정에 참여
- **step 2**: VLM이 H 생성 (이 시점엔 외형 정보 없음)
- **step 3**: VAE가 DiT 옆구리에 도착 — 의미 결정이 이미 끝난 후

이 비대칭이 만든 결과:
- VLM은 "1번 여자가 2번 모자를 쓴다"는 grounding까지만 함
- DiT가 알아서 "이 hidden state는 1번 VAE와 묶인다"는 binding을 **암묵적으로** 추론
- 참조가 둘 이상이면 자주 깨짐 → **속성 유출**, **참조 간 혼동**

---

### 2. ViT와 VAE의 비대칭

> 왜 이 절을 두는가? "그냥 둘 다 VLM에 넣자"가 왜 비자명한지, 식 (1)의 행렬 모양이 왜 그렇게 생겼는지가 여기서 결정됩니다.

| 비교항 | ViT (Qwen2.5-VL 내장) | VAE (LongCat) |
|---|---|---|
| **채널 차원** | d_vit ≈ 1280 ~ 1536 | d_vae ≈ 16 |
| **토큰 수 (512² 입력)** | ≈ 324 (stride 28, 18×18) | ≈ 4096 (stride 8, 64×64) |
| **담는 정보** | 의미(semantic), 텍스트-align | 외형(appearance), 픽셀 복원 가능 |
| **기존 들어가는 곳** | VLM 입력 | DiT 옆구리 |
| **시점** | step 1 | step 3 |

**채널은 ViT가 압도적으로 크고, 토큰 수는 VAE가 압도적으로 많은** 비대칭 구조. 둘을 합치려면 (1) 토큰 수를 맞추고 (2) 채널을 정렬해야 함. 논문은 "512×512 이하로 리사이즈해 시퀀스 길이를 동일 L로 맞춘다"로 (1)을 처리, 식 (1)로 (2)를 처리.

---

### 3. Figure 2 — UniCustom 아키텍처 전체 구조

> 왜 이 절을 두는가? 한 장에 모든 모듈이 어떻게 연결되는지 시각으로 보면 식 (1)~(3)이 어디에 위치하는지 명확해집니다.

![Figure 2: Overview of UniCustom](https://arxiv.org/html/2605.12088v1/x2.png)

> **원문 캡션:** *"Overview of UniCustom. UniCustom fuses ViT and VAE features before VLM encoding, producing semantically addressable and appearance-aware hidden states for DiT generation."*

#### Fig.2 블록별 해설 (좌 → 우)

**① 입력 (Reference Images)**
- N장의 참조 이미지가 좌측에 나란히 들어옴
- 모두 512×512 이하로 리사이즈되어 시퀀스 길이 L 정렬

**② 병렬 인코더 (ViT Encoder · VAE Encoder)**
- **ViT Encoder (위쪽 가지)**: Qwen2.5-VL 내부의 비전 인코더 → `F_vit ∈ ℝ^(L × d_vit)` 출력
- **VAE Encoder (아래쪽 가지)**: LongCat의 VAE → `F_vae ∈ ℝ^(L × d_vae)` 출력
- 두 가지가 **병렬**로 그려지고, 둘 다 동시에 ③번 모듈에 들어감

**③ Linear Fusion Layer (논문의 핵심 모듈)**
- 식 (1)이 일어나는 자리. 채널 방향 concat → 선형 투영 → `F_uni ∈ ℝ^(L × d_vit)`
- 그림에서 가장 강조되는 박스 (저자들이 강조하는 노벨티)
- 출력 F_uni의 모양이 ViT 토큰과 같아서 ④의 입력으로 그대로 쓰임

**④ Slot 시퀀스 조립 ("Picture k:" 라벨 + F_uni_k 반복)**
- 각 참조마다 "Picture 1:", "Picture 2:" … 텍스트 라벨이 토큰화되어 앞에 붙음
- 이어서 그 참조의 F_uni가 슬롯으로 자리잡음
- 마지막에 사용자 지시문(예: "The woman from Picture 1 wears the hat from Picture 2")

**⑤ Qwen2.5-VL (Frozen)**
- 그림에서 보통 자물쇠 아이콘으로 표시됨 — **얼린 상태**
- MRoPE로 위치 인코딩
- 출력: 각 슬롯에 대응하는 hidden state `H_i`

**⑥ DiT (LongCat-Image-Edit)**
- H_i가 cross-attention 조건으로 들어감
- 노이즈 latent에서 출발하여 디노이징 → 최종 생성 이미지
- 그림 우측 끝에 배치

**⑦ Slot-wise Binding Regularization 경로 (보조, 점선)**
- VLM hidden state H_i 에서 projector P를 거쳐 VAE 공간으로 되돌리는 화살표
- 원본 F_vae_i 와 비교하는 MSE 손실 L_bind
- **점선으로 그려진 이유**: 학습 1단계에서만 활성, 추론 시엔 제거됨

#### 데이터 흐름 요약

```
   ┌─────────── ViT ─────► F_vit ──┐
Img│                                ├─► [Concat ; Linear]  ──► F_uni ──► Qwen2.5-VL ──► H_i ──► DiT ──► Output
   └─────────── VAE ─────► F_vae ──┘                                      │
                                                                          ▼
                                                         (학습 시) P(H_i) ≈ F_vae_i   ← L_bind
```

#### 색상 / 선 규약 (관행)
- **실선 화살표**: 추론 + 학습 모두에서 흐르는 메인 경로
- **점선 화살표**: 학습 1단계에서만 활성화되는 보조 경로 (binding reg)
- **자물쇠 아이콘** (Qwen2.5-VL 위): 파라미터 동결
- **불꽃/그라디언트 아이콘** (Linear Fusion, DiT): 학습 가능

---

### 4. 식 (1) — Unified Visual Representation

> 왜 이 절을 두는가? 그림 ③번 블록 안에서 실제로 일어나는 연산. 단순 선형 결합 한 줄이 전부지만 풀어보면 의미가 깊습니다.

```
F_uni = [F_vit ; F_vae] · W_fuse + b_fuse
```

차원 명세:

```
F_vit   ∈ ℝ^(L × d_vit)
F_vae   ∈ ℝ^(L × d_vae)
[F_vit ; F_vae] ∈ ℝ^(L × (d_vit + d_vae))
W_fuse  ∈ ℝ^((d_vit + d_vae) × d_vit)
b_fuse  ∈ ℝ^(d_vit)
─────────────────────────────────────
F_uni   ∈ ℝ^(L × d_vit)
```

**평문 풀이:** 각 공간 위치에서 ViT 채널 벡터(약 1280차원)와 VAE 채널 벡터(약 16차원)를 **옆으로 이어붙여** 약 1296차원짜리 벡터를 만들고, 학습 가능한 행렬 W_fuse를 곱해 **다시 ViT와 같은 약 1280차원으로 압축**. 결과 F_uni 모양은 ViT 토큰과 똑같아서 VLM이 그대로 받습니다. 핵심은 — 이 한 토큰 안에 **의미와 외형이 둘 다 살아 있다**는 점.

---

### 5. 식 (2) — Identity-Preserving Initialization

> 왜 이 절을 두는가? 융합 레이어를 랜덤 초기화하면 사전학습된 VLM 분포가 깨집니다. 항등 초기화는 그 깨짐을 0으로 만드는 트릭.

```
         ⎡ I_(d_vit)         ⎤
W_fuse = ⎢                   ⎥ ,    b_fuse = 0_(d_vit)
         ⎣ 0_(d_vae × d_vit) ⎦
```

학습 시작 시점:

```
                              ⎡ I ⎤
F_uni(init) = [F_vit ; F_vae] · ⎢   ⎥ + 0  =  F_vit
                              ⎣ 0 ⎦
```

학습 초기엔 F_uni ≡ F_vit — VLM이 받는 입력이 기존과 **비트 단위로 동일**. 학습이 진행되며 아래쪽 0 블록이 점점 VAE 신호를 흡수. **사전학습 능력을 한 톨도 잃지 않고** 외형 채널을 슬며시 여는 부트스트랩.

---

### 6. Slot 구조와 멀티 레퍼런스 시퀀스

> 왜 이 절을 두는가? 그림의 ④번 단계. "여러 장의 참조를 줄거리 안으로 어떻게 끼우느냐"가 멀티 레퍼런스의 또 다른 난제.

VLM 입력 시퀀스 구성:

```
"Picture 1:"  <F_uni_1>   "Picture 2:"  <F_uni_2>   ···   "The woman from Picture 1 wears the hat from Picture 2 ..."
 └─ ID tag ┘  └─ slot 1 ┘  └─ ID tag ┘  └─ slot 2 ┘         └─────────────── 텍스트 지시 ───────────────┘
```

"Picture k:" 라는 텍스트 라벨이 **명시적 ID 태그** 역할을 하고, 텍스트의 "from Picture 2"가 어텐션 가중치를 통해 2번 슬롯 토큰들을 정확히 가리키게 됨. 위치 인코딩은 **MRoPE (multimodal rotary positional embedding)** 그대로 사용.

이 슬롯이 VLM을 통과한 후 출력되는 hidden state를 `H_i` (i번째 슬롯) 라 부르고, H_i가 DiT의 조건 신호.

---

### 7. 식 (3) — Slot-wise Binding Regularization

> 왜 이 절을 두는가? Early fusion만으로는 학습 도중 VLM이 ViT 신호에 편향되어 VAE 디테일을 잊을 위험. 강제 보험을 답니다.

```
            1     N
L_bind  =  ─── ·  Σ   ‖ P(H_i) − F_vae_i ‖²₂
            N    i=1
```

- `H_i ∈ ℝ^(L × d_vlm)`: i번째 슬롯의 VLM hidden state
- `P : ℝ^(d_vlm) → ℝ^(d_vae)`: 단일 레이어 projector
- `F_vae_i`: 원본 i번째 참조의 VAE 특성
- `N`: 참조 이미지 개수
- `‖·‖²₂`: L2 노름의 제곱 (= MSE)

**평문 풀이:** "VLM아, 네가 i번 슬롯 자리에 만들어낸 출력 H_i를 작은 디코더 P에 통과시키면 원래 i번째 VAE 토큰이 그대로 복원돼야 한다." → **VLM hidden state 안에 i번째 외형이 살아 있어야 함을 강제**.

일종의 **사이드 채널 오토인코더 제약**. VLM의 forward path 자체는 건드리지 않고, "외형이 살아남으라"는 학습 신호만 옆에서 주입.

**중요한 디테일:** projector P는 **1단계(재구성 사전학습) 동안에만** 사용되고 이후 폐기. 추론 시엔 안 씀.

#### 총 손실 (논문 명시 없음 — 추정)

```
L_total = L_diff + λ · L_bind
```

- `L_diff`: DiT 표준 손실 (flow matching 또는 v-prediction; 본문에 명시 없음)
- `λ`: 가중치 (값 미공개)

---

### 8. Figure 3 — 2단계 학습 전략

![Figure 3: Two-stage training strategy](https://arxiv.org/html/2605.12088v1/x3.png)

> **원문 캡션:** *"Two-stage training strategy. The first stage progressively learns a unified visual representation that supports fine-grained reference encoding, semantic grounding, and reliable textual-to-visual binding through reconstruction-oriented multi-image pretraining. The second stage further adapts the diffusion backbone to reference-based image generation, enabling instruction-following synthesis with single or multiple reference images while preserving the learned grounding and binding abilities."*

#### Stage 1 — Reconstruction-Oriented Pretraining
- **과제:** multi-image reconstruction / localization / tiling (self-supervised)
- **업데이트:** W_fuse, b_fuse + DiT (VLM 얼림)
- **손실:** L_diff + λ · L_bind
- **스텝:** 18K, lr = 5×10⁻⁵
- **목적:** 통합 표현 학습 + slot 단위 grounding-binding 확립

#### Stage 2 — Supervised Fine-Tuning
- **과제 비율:**
  - 단일/멀티 레퍼런스 생성 ≈ 75%
  - 텍스트→이미지 ≈ 5%
  - 이미지 편집 ≈ 10%
  - 1단계 과제 유지 ≈ 10% (망각 방지)
- **업데이트:** DiT만 (**융합 레이어 동결**, L_bind 제거)
- **스텝:** 18K, lr = 1×10⁻⁵
- **목적:** 실제 생성 과제에 백본을 정밀 적응시키되, 1단계에서 배운 grounding-binding 능력 보존

---

### 9. Figure 1 — Decoupled vs Unified

![Figure 1: Decoupled vs Unified visual conditioning](https://arxiv.org/html/2605.12088v1/x1.png)

> **원문 캡션:** *"Illustration of decoupled and unified visual conditioning."*

- **(a) Decoupled** — 기존 방식, 두 갈래
- **(b) Unified** — UniCustom 방식, 한 갈래

이 그림이 논문의 정체성을 한 장으로 압축. (a)와 (b)의 가장 큰 시각적 차이는 **VAE 화살표의 도착지** — (a)에서는 DiT를 향하고, (b)에서는 VLM 입력 합류점을 향함.

---

## 실험 요약

### 데이터
HQ-50K, MultiID-2M, HumanArt, FFHQ, Unsplash, COCO2017, Echo-4o-Image, MICo-150K + 내부 데이터

### Table 1 — OmniContext 벤치마크 (오픈소스 1위)

| 카테고리 | UniCustom | LongCat (백본) | Qwen-IE | OmniGen2 | GPT-Image-2 | Nano Banana 2 |
|---|---|---|---|---|---|---|
| Single Character | 8.06 | — | — | — | — | — |
| Single Object | 7.51 | — | — | — | — | — |
| Multi Character | **7.99** | — | — | — | — | — |
| Multi Object | **7.86** | — | — | — | — | — |
| Char + Obj | **7.86** | — | — | — | — | — |
| **평균** | **7.84** | 7.26 | 7.69 | 7.60 | 9.24 | 8.92 |

→ **같은 백본 LongCat (7.26) 대비 +0.58** 이 사실상 UniCustom 모듈의 순수 기여. 클로즈드 모델과는 약 1.0 ~ 1.4 격차.

### Table 2 — MICo-Bench (격차가 훨씬 큼)

| 카테고리 | UniCustom | LongCat | Qwen |
|---|---|---|---|
| Object | **54.30** | 40.55 | — |
| Person | 18.12 | — | — |
| HOI | 40.51 | — | — |
| Decomp & Recomp | 50.29 | — | — |
| **평균** | **41.71** | 22.78 | 21.67 |

→ MICo-Bench는 멀티 주체 상호작용 / 분해·재조합 특화 벤치. **LongCat 대비 거의 2배.** Early fusion + binding reg가 멀티 레퍼런스 시나리오에서 진짜로 효과적이라는 강한 증거.

### Ablation (Figure 7, 8)

**(a) Binding Reg on/off**

```
ON  :  PSNR_single ≈ 30 dB,  multi-image accuracy: 안정 상승
OFF :  초기 수렴 빠름, 그러나 multi-image accuracy 천장에 막힘
```

**(b) Fusion Strategy 비교**

| 전략 | PSNR | Multi-image accuracy | Binding loss |
|---|---|---|---|
| ViT-only | 최저 | 최저 | — |
| Late fusion (VAE를 VLM 뒤에 주입) | 초기 좋음, 결국 ViT-only로 수렴 | 낮음 | 높음 |
| **Early fusion (제안)** | 일관 최고 | 일관 최고 | ≈ 0 |

핵심 시사점: **"Late fusion은 디테일은 살리지만 binding은 못 살린다"** — UniCustom 주장의 결정타.

### 정성 결과

- 다중 인물 상호작용(포옹·마주보기) 자세 정확
- 의자 배치 등 공간 레이아웃 정확
- 옷·정체성 디테일 보존
- 주체 누락(subject missing), 속성 유출 거의 없음

---

## 💬 Q&A

### Q1. "원래는 ViT > VAE 순서로 적용하는 거 아냐?"

**답:** 맞다. 두 가지 의미 모두에서.

**(a) 시간 순서 관점** — 기존 모델의 처리 순서는 정확히 `ViT(VLM) → VAE(DiT)`:
1. 참조 이미지 → ViT 특성 → VLM 입력으로 들어감
2. VLM이 텍스트와 함께 처리해 hidden state H 생성 (이때 grounding 완료)
3. H가 DiT의 cross-attention 조건으로 전달
4. DiT가 노이즈 latent를 풀면서 **참조의 VAE 토큰**을 옆구리에서 받아 외형 합성

UniCustom의 결정타는 **이 시간 순서를 깨버린 것**. ViT와 VAE를 0단계에서 한 토큰으로 묶어 동시에 VLM에 넣어버림. VLM이 grounding을 결정하는 그 순간에 외형 정보가 이미 옆에 있음.

**(b) 채널 차원 관점** — d_vit ≫ d_vae (1280+ vs 16). 그래서 식 (1)에서 (d_vit + d_vae) 차원을 다시 d_vit 차원으로 압축하는 게 자연스럽고, 식 (2)의 항등 초기화도 "위쪽 d_vit행 = 단위행렬, 아래쪽 d_vae행 = 0" 형태가 됨.

### Q2. 토큰 수가 ViT < VAE인데 어떻게 L을 맞추나?

512×512 입력 기준 실제 토큰 수:
- ViT (Qwen2.5-VL, stride 28): 약 324 토큰
- VAE (stride 8): 약 4096 latent

논문은 "512×512 이하로 리사이즈해 시퀀스 길이를 동일 L로 맞춘다" 한 줄로 넘어가지만, 실제로 어떤 patchify/pooling을 거치는지 본문에 명시 없음. **코드 공개를 기다려야 확인 가능한 디테일.**

### Q3. 왜 융합 레이어를 단순 선형으로 했나?

논문이 명시한 의도는 두 가지 — (1) **사전학습된 VLM 호환성 보존** (항등 초기화 가능), (2) **추가 파라미터 최소화** (Qwen2.5-VL 같은 큰 백본 옆에 무거운 모듈을 붙이면 학습 신호가 흩어짐). Late fusion이 더 강력해 보일 수 있지만, ablation에서 결국 ViT-only 수준으로 수렴 — VLM이 외형을 못 보는 게 문제이지 모듈 표현력이 문제가 아님을 증명.

### Q4. 클로즈드 모델(GPT-Image-2, Nano Banana 2)과의 격차는?

OmniContext 평균에서 UniCustom 7.84 vs GPT-Image-2 9.24, Nano Banana 2 8.92. 약 1.0~1.4 격차 — 주로 **인물 정체성 보존**과 **복잡한 객체 상호작용**에서 뒤짐. 다만 MICo-Bench(멀티 주체 특화)에서는 오픈소스 중 압도적 — 즉 UniCustom의 강점은 **여러 참조의 분리 유지**이고, 격차의 본질은 **단일 정체성의 fidelity** 쪽.

### Q5. Z-Image, FLUX 2와 구조적으로 어떻게 다른가?

> 왜 이 절을 두는가? 비슷한 시기에 나온 동급 오픈/세미오픈 거대 모델과의 자리매김. 같은 "이미지 생성"이지만 셋의 설계 철학이 정반대 방향.

#### 구조 비교 표

| 측면 | **UniCustom** (2026-05) | **Z-Image** (2025-11) | **FLUX 2** (2025-11) |
|---|---|---|---|
| **주체** | Kuaishou Kling 추정 | Alibaba Tongyi MAI | Black Forest Labs |
| **주요 목표** | 멀티 레퍼런스 grounding-binding 해결 | 6B로 20B+ 따라잡기 (효율) | 통합 멀티 레퍼런스 + 편집 |
| **확산 백본** | LongCat-Image-Edit DiT (재사용) | 6B **Single-Stream DiT** (자체) | Rectified Flow Transformer (자체) |
| **파라미터 (확산)** | LongCat 크기 (논문 미명시) | **6B** | **약 32B (dev)** |
| **언어 처리 모듈** | **Qwen2.5-VL (VLM, 얼림)** | **Qwen3-4B (텍스트 only LLM)** | **Mistral-3 24B VLM** |
| **참조 비전 인코더 (ViT)** | Qwen2.5-VL 내장 ViT | 별도 ViT 없음 | VLM 내장 (Mistral-3 vision) |
| **참조 외형 인코더 (VAE)** | LongCat VAE (16ch 추정) | LongCat류 VAE | **새로 학습한 VAE** (compression 최적화) |
| **참조 컨디셔닝 방식** | **Early Fusion** (식 1): ViT+VAE → VLM 전 결합 | 편집은 reference latent를 DiT에 in-context | VLM과 flow transformer 결합 (디테일 미공개) |
| **두 갈래 분리 여부** | **통합 (한 갈래)** ← 노벨티 | 텍스트만 (참조 비전 미사용) | 두 갈래 추정 |
| **멀티 레퍼런스 지원** | **특화** (이론 N장 무제한) | 보조 (편집 위주, 1~소수장) | **최대 10장** (명시) |
| **의미↔외형 binding 강제** | **Slot-wise L_bind MSE** (식 3) | 해당 없음 | 미공개 |
| **항등 초기화 트릭** | W_fuse = [I; 0] (식 2) | 해당 없음 | 미공개 |
| **학습 비용** | H200×128 + 36K steps | **314K H800h** (강조점) | 미공개 |
| **Distillation** | 미공개 | **Decoupled DMD + DMDR** (Turbo 8-step) | 미공개 |
| **RLHF** | 미공개 | **2-stage RLHF** | 미공개 |
| **공개 범위** | "코드 곧 공개" | 추론 코드 + 가중치 공개 | API + 가중치 일부 공개 |

#### 한 줄로 자리매김

- **UniCustom** — "**입력 인터페이스를 손본 모델**". 기존 DiT(LongCat)는 그대로, **VLM에 들어가는 참조 토큰의 모양만** 바꿈. 식 (1) 한 줄이 핵심.
- **Z-Image** — "**전체 파이프라인을 다이어트한 모델**". 참조 이미지 문제는 거의 안 다루고 텍스트→이미지의 효율에만 집중. VLM 없이 Qwen3-4B 텍스트 LLM만, Single-Stream DiT로 파라미터 절약.
- **FLUX 2** — "**큰 VLM과 큰 flow transformer를 결합한 거인**". Mistral-3 24B VLM + 32B rectified flow transformer. 10장까지 받지만 내부 binding 메커니즘 미공개.

#### 자리매김 도식

```
                    멀티 레퍼런스 특화
                          ▲
                          │
                    UniCustom (소형 노벨티)
                          │
                          │
   효율 ◄──── Z-Image ─────┼─────── FLUX 2 ────► 스케일
   (6B)                    │                  (32B + 24B VLM)
                          │
                          │
                    텍스트→이미지 위주
                          ▼
```

#### 흥미로운 비교 포인트

1. **VLM 사용 양상이 셋 다 다름**: Z-Image는 텍스트 LLM만, UniCustom은 Qwen2.5-VL을 얼린 채로 입력만 손봄, FLUX 2는 Mistral-3 24B VLM 전체를 무겁게 운영.
2. **참조 컨디셔닝 철학이 다름**: Z-Image는 "참조는 부차적", UniCustom은 "참조 짝짓기가 모든 것", FLUX 2는 "참조도 많이 받고 잘하자(디테일 비공개)".
3. **UniCustom의 약점이 곧 다른 두 모델의 강점**: 백본 의존성(LongCat 한정), 단일 정체성 fidelity — Z-Image는 자체 효율 백본으로, FLUX 2는 큰 스케일로 각각 해결.

#### 면책

FLUX 2의 내부 ViT/VAE 처리 디테일과 binding 메커니즘은 [공식 announcement](https://bfl.ai/announcements/flux-2)에만 의존. 공식 기술 보고서가 별도로 있다고 함 — 표의 FLUX 2 일부 행은 미공개·추정. 보고서 확인 후 갱신 필요.

---

## 한 줄 요약 (전체)

**"멀티 레퍼런스 디퓨전의 두 갈래 컨디셔닝을 식 (1)의 선형 결합과 식 (3)의 MSE 단 두 줄로 한 갈래로 묶은 작업. 진단(Grounding-Binding Gap)이 처방보다 더 오래 남을 가능성이 큰 논문."**

---

## 강점·약점·자리매김

### 강점
- **진단의 명확성**: Grounding-Binding Gap 이라는 이름이 후속 연구의 공용 어휘가 될 만함
- **모듈의 단순함**: 융합은 선형 한 줄(식 1), reg는 MSE 한 줄(식 3). 그런데 LongCat 대비 멀티 벤치에서 거의 2배
- **항등 초기화**(식 2): 사전학습 능력을 한 비트도 잃지 않는 매끄러운 부트스트랩

### 약점 / 의문
- **클로즈드 격차** (OmniContext 7.84 vs GPT-Image-2 9.24)
- **백본 의존성**: LongCat-Image-Edit이라는 특정 DiT에 강하게 의존. FLUX/SDXL/SD3에 이식 시 같은 효과 미증명
- **λ 값, projector P의 정확한 구조, L_diff 형태 미공개** — 재현은 코드 공개 대기
- **토큰 길이 정렬의 디테일** — 실제 ViT 토큰 수(약 324)와 VAE 토큰 수(약 4096)는 한 자릿수 차이. 어떤 patchify/pooling이 들어가는지 본문에 명시 없음
- **코드/체크포인트 "곧 공개" 상태**

### 자리매김
- 백본 재사용 지형도 관점: **분기 B(디퓨전 재사용) + 분기 C(VLM 재사용)의 인터페이스를 손본 첫 진지한 작업**
- 같은 가족인 [Any2AnyTryon](PAPER_Any2AnyTryon.md)이 **RoPE 채널 재해석**으로 mask-free 통합을 노렸다면, UniCustom은 **VLM 입력 채널 재해석**으로 멀티 레퍼런스 통합을 노린 평행 사례

---

## 관련 메모리 링크

- [[reference_pretrained_backbone_reuse_landscape]] — VLM/Diffusion 백본 재사용 분기 분류
- [[paper_any2anytryon]] — RoPE 채널 재해석으로 mask-free 통합 (UniCustom의 평행 사례)
- [[paper_dreamlite]] — Qwen3-VL + DMD2 4-step 소형 모델, 같은 VLM+DiT 라인의 다른 방향
- [[feedback_paper_summary_format]] — 이 문서가 따른 형식
- [[feedback_beginner_friendly_tone]] — 톤
- [[feedback_chapter_why_intro]] — 장 도입부 "왜?" 한 줄
