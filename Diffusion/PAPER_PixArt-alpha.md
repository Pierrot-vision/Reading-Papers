# PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis

## 메타 정보

- **논문 제목**: PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis
- **저자**: Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, Zhenguo Li
- **소속**: Huawei Noah's Ark Lab, Dalian University of Technology (DUT), HKU, HKUST
- **공개일**: 2023-09-30 (v1) / 2023-12-29 (v3 최종)
- **Venue**: **ICLR 2024 (Spotlight)**
- **분야**: Text-to-Image Generation, Diffusion Transformer
- **논문 링크**: [arXiv abstract](https://arxiv.org/abs/2310.00426) · [PDF](https://arxiv.org/pdf/2310.00426)
- **프로젝트 페이지**: [pixart-alpha.github.io](https://pixart-alpha.github.io/)
- **코드**: [github.com/PixArt-alpha/PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)
- **모델 체크포인트**: HuggingFace [PixArt-alpha](https://huggingface.co/PixArt-alpha)
- **공개 데이터셋**: [SAM-LLaVA-Captions10M](https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M)
- **사용한 외부 모델/데이터**:
  - 텍스트 인코더: **T5-XXL (Flan-T5-XXL, 4.3B)** — 토큰 길이 120, frozen 사용
  - VAE: **sd-vae-ft-ema** (Stable Diffusion 1.x용 VAE, 8배 압축)
  - 백본 초기화: **DiT-XL/2** (ImageNet pretrained) 가중치
  - 캡션 보강 모델: **LLaVA-7B**
  - 학습 데이터셋: ImageNet, SAM (Segment Anything) 1100만 장, LAION-Aesthetic, JourneyDB, 내부 데이터 1400만 장

---

## 주요 용어 사전 (Glossary)

### 아키텍처 관련

- **DiT (Diffusion Transformer)** — 디퓨전 모델의 노이즈 예측기를 U-Net이 아니라 **Transformer 한 덩어리**로 바꾼 구조 (William Peebles, 2022). ChatGPT 같은 언어모델에서 쓰는 Transformer를 이미지 생성에 그대로 가져옴. 단 원본 DiT는 ImageNet 클래스 라벨만 받고 자유 문장은 못 받음. PixArt-α의 출발점.
- **adaLN (adaptive Layer Normalization, 조건 적응형 정규화)** — 조건 정보(c, 예: timestep+클래스)로부터 LayerNorm의 scale/shift 값을 동적으로 만들어 토큰에 주입하는 방식. 원래 DiT는 각 block마다 **6개의 modulation 파라미터** (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)를 각각 학습.
- **adaLN-single** — 본 논문이 제안. **timestep만으로 전역 modulation 1개를 계산**하고, 각 block은 **자기만의 작은 보정값(`scale_shift_table[6, D]`)** 을 더해서 사용. 파라미터 26% 절감.
- **Cross-Attention (교차 주의 메커니즘)** — 이미지 토큰(query)이 텍스트 토큰(key/value)을 참조해 정보를 받아오는 attention. PixArt-α는 self-attention과 MLP 사이에 cross-attention을 삽입. **단방향**: 이미지는 텍스트를 보지만, 텍스트는 이미지를 못 봄.
- **MM-DiT (Multi-Modal DiT, Joint Attention)** — SD3/FLUX의 구조. 이미지 토큰과 텍스트 토큰을 같은 sequence에 **concat 한 뒤 self-attention에 함께 통과**. 양방향 정보 흐름.
- **Reparameterization (재파라미터화)** — ImageNet으로 사전학습된 DiT 가중치를 **그대로 출발점으로 쓰기 위한 초기화 기법**. cross-attention 출력 projection을 0으로, `scale_shift_table`을 0으로 초기화해서 학습 시작 시점에 원본 DiT와 동일하게 동작하도록 만듦.
- **PatchEmbed** — 입력 latent를 patch로 잘라 1D 토큰 시퀀스로 만드는 모듈. 1024px → latent 128×128 → patch=2 → 토큰 64×64 = 4096개.
- **CaptionEmbedder** — T5에서 나온 텍스트 임베딩(4096차원, 120 토큰)을 모델 hidden size(1152)로 투영하는 MLP. (원본 DiT의 `LabelEmbedder`를 대체)
- **T2IFinalLayer** — PixArt 전용 final layer. 원본 DiT의 `FinalLayer`를 adaLN-single 패턴으로 재구성 (2개 보정값: shift, scale).
- **CFG (Classifier-Free Guidance)** — 텍스트 조건이 있을 때와 없을 때(unconditional)의 예측을 보간해 텍스트 정합을 강화하는 디퓨전 샘플링 기법. 학습 중 10% 확률로 캡션을 비워 unconditional 분기 학습.

### 학습 전략

- **Training Strategy Decomposition (학습 전략 분해)** — 한 번에 모든 능력을 학습하지 않고, **픽셀 의존성 → 텍스트-이미지 정합 → 미적 품질** 3단계로 쪼개서 점진 학습.
- **Stage 1: Pixel Dependency Learning** — ImageNet pretrained DiT를 가져와 픽셀 분포를 이미 아는 상태에서 출발.
- **Stage 2: Text-Image Alignment Learning** — SAM 1100만 장 + LLaVA dense caption으로 정합 학습.
- **Stage 3: High-Resolution & Aesthetic** — LAION-Aesthetic, JourneyDB, 내부 데이터로 512→1024px 미적 품질 향상.

### 데이터 관련

- **Concept Density (개념 밀도)** — 캡션 한 줄에 얼마나 많은 명사(객체, 속성, 관계)가 들어있는지. LAION 원본 캡션 평균 noun 6.4개, **SAM-LLaVA dense caption은 30개**.
- **SAM-LLaVA-Captions10M** — PixArt-α가 만든 데이터셋. Segment Anything 1100만 장 이미지에 LLaVA로 dense caption을 자동 생성. HuggingFace 공개.
- **Pseudo-caption (의사 캡션)** — Vision-Language 모델(LLaVA)이 자동 생성한 캡션. 사람이 쓴 캡션과 구분.

### 평가 지표

- **FID-30K (Fréchet Inception Distance)** — MS-COCO에서 30K 샘플 평가. 생성-실제 이미지 분포 거리. 낮을수록 좋음. PixArt-α: **7.32 (zero-shot), 5.51 (COCO fine-tune)**.
- **T2I-CompBench** — 텍스트-이미지 **구성 능력(compositional ability)** 벤치마크. 6개 항목: Color, Shape, Texture (속성-객체 결합), Spatial (공간 관계), Non-spatial (비공간 관계), Complex (복합).
- **User Study (사용자 평가)** — 사람이 두 모델 결과를 직접 비교. PixArt-α vs SDv2: 품질 +7.2%, 정합 +42.4% 우위.

### 비교 모델

- **SD 1.5 / 2 / SDXL** — Stable Diffusion 시리즈. U-Net 기반. SD 1.5: 6,250 A100 days, $320K.
- **Imagen** — Google의 T2I. T5-XXL 텍스트 인코더 + 캐스케이드 디퓨전. 7,132 A100 days.
- **DALL·E 2 / DALL·E 3** — OpenAI의 T2I. DALL·E 2: 41,660 A100 days.
- **RAPHAEL** — 50억+ 이미지로 학습한 거대 모델. 60,000 A100 days (PixArt-α의 80배).

---

## 논문 요약 (TL;DR)

**한 줄 요약**: PixArt-α는 0.6B 파라미터의 Transformer 기반 T2I 디퓨전 모델로, **Stable Diffusion v1.5의 12% 학습 시간 (753 A100 days, $28,000)** 만으로 Imagen·SDXL·Midjourney급 품질을 달성한다.

**핵심 문제**: 기존 SOTA T2I 모델들은 수백만 GPU시간이 필요해 (SD 1.5: 6,250 A100 days, RAPHAEL: 60,000 A100 days) 학계와 스타트업이 진입하기 어렵고 CO2 배출도 크다. 학습 효율을 한 자릿수 % 수준으로 줄이면서도 품질을 유지할 수 있는가?

**해결책 3가지**:
1. **학습 전략 분해 (Training Strategy Decomposition)** — 픽셀/정합/미적 3단계로 쪼개고 ImageNet pretrained DiT를 재활용
2. **효율적 T2I Transformer 아키텍처** — DiT에 cross-attention 추가 + adaLN-single로 파라미터 절약 + reparameterization으로 사전학습 가중치 그대로 활용
3. **고밀도 정보 데이터 (High-informative Data)** — LLaVA로 SAM 이미지에 dense caption을 자동 생성해 개념 밀도를 4~5배 끌어올림

**검증**:
- COCO FID-30K **7.32 (zero-shot), 5.51 (fine-tune)** — Imagen(7.27)과 동급
- T2I-CompBench에서 SDXL 능가 (Texture 0.7044 vs SDXL 0.5637 등)
- User study에서 SDv2 대비 품질 +7.2%, 정합 +42.4%
- 학습 비용 절감: SD 1.5 대비 **CO2 90% 감축, $290K 절약**

---

## 핵심 기여 (Contributions)

1. **adaLN-single + Reparameterization 으로 사전학습 DiT 가중치 재활용** — ImageNet으로 학습된 class-conditional DiT를 초기 가중치로 사용해 학습 비용 대폭 절감. 단순 init이 아니라, 새 구조가 학습 시작 시점에 원본 DiT와 동일한 출력을 내도록 reparameterization 트릭 설계.

2. **3단계 학습 분해 (Training Decomposition)** — "픽셀 분포 학습 → 텍스트-이미지 정합 → 미적 품질"을 순차 학습. 각 단계마다 가장 적합한 데이터를 사용해 단계별 학습량 최소화.

3. **SAM + LLaVA pseudo-caption 데이터셋 (SAM-LLaVA-Captions10M)** — 객체가 빽빽한 SAM 1100만 장 이미지에 LLaVA로 dense caption을 새로 붙여 평균 noun 30개/이미지 (LAION 원본 6.4개) 수준의 고밀도 정합 학습 신호 생성. 공개됨.

4. **DiT에 cross-attention 모듈 통합** — class label 임베딩 분기를 제거하고 self-attention과 MLP 사이에 multi-head cross-attention을 끼워넣어 텍스트 조건 주입.

5. **학습 비용 절감 결과 입증** — SD 1.5의 12% 시간, $28K로 SOTA급 품질 달성. CO2 90% 감축.

---

## 주요 알고리즘 설명

### 1. PixArt block = "기존 DiT block + 4가지 변경"

<p align="center">
  <img src="figures/pixart_alpha_fig4.png" alt="PixArt-α Architecture (Fig. 4)" width="560"/>
</p>

> **Fig. 4** — Model architecture of PixArt-α. A cross-attention module is integrated into each block to inject textual conditions. To optimize efficiency, all blocks share the same adaLN-single parameters for time conditions. (위 그림: 논문 Figure 4)

본질적으로 **DiT를 최대한 보존하면서 텍스트 능력만 추가**한 구조. 하지만 단순한 "cross-attention 추가"가 아니라 **추가 1 + 수정 3** 이 같이 일어남.

```
[기존 DiT block]                       [PixArt block]
                                       
입력 x ─┐                              입력 x ─┐
        ↓                                       ↓
   ┌─────────────┐                        ┌──────────────────┐
   │ adaLN(c)    │  ← c=t+class           │ adaLN-single(t)  │  ← ② t만 (class 제거)
   │ MLP_i 28개  │     block마다 별도      │ scale_shift_table │     글로벌+block 보정
   └─────────────┘                        └──────────────────┘
        ↓                                       ↓
  [Self-Attention]                      [Self-Attention]    ← 그대로 (가중치 재활용)
        ↓                                       ↓
       (없음)                            [Cross-Attention(y)]  ← ① ★ NEW (텍스트 주입)
                                                ↓
   ┌─────────────┐                        ┌──────────────────┐
   │ adaLN(c)    │                        │ adaLN-single(t)  │
   └─────────────┘                        └──────────────────┘
        ↓                                       ↓
      [MLP]                               [MLP]              ← 그대로 (가중치 재활용)
        ↓                                       ↓
출력 x                                  출력 x
```

| # | 변경 종류 | 변경 내용 | 의도 |
|---|---|---|---|
| ① | **추가 (NEW)** | Cross-Attention layer | 텍스트 조건 받기 위한 새 능력 |
| ② | **수정** | adaLN → adaLN-single | 파라미터 26% 절감 |
| ③ | **제거** | class_embedding 분기 (텍스트는 cross-attn으로 가니까) | 입력 조건 단순화 |
| ④ | **수정** | FinalLayer → T2IFinalLayer (adaLN-single 패턴) | 글로벌 modulation 패턴 일관성 |

**Self-Attention과 MLP라는 핵심 연산 블록은 손대지 않음** — 이게 DiT pretrained 가중치를 그대로 재활용할 수 있는 본질적 이유. Cross-attention만 zero-init으로 끼워 넣으면 학습 시작 시점에 DiT와 동일 동작 (= reparameterization 가능).

코드 ([diffusion/model/nets/PixArt.py:25-54](https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt.py#L25-L54)):

```python
class PixArtBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, ...):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, ...)   # ← Self-Attn (DiT 그대로)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, ...)  # ★ ① NEW
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(...)                                                   # ← MLP (DiT 그대로)
        # ② 글로벌 modulation을 받기 위한 block별 작은 보정값
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None):
        # t: 외부에서 미리 계산된 글로벌 modulation (B, 6*D) — 28개 block이 공유
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        # 풀이: 이 block만의 작은 보정 + 모든 block 공유 글로벌 modulation
        
        x = x + gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(x, y, mask)             # ★ ① 텍스트 주입 (zero-init proj)
        x = x + gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

### 2. adaLN-single 의 효과: 파라미터 26% 절감

| 항목 | 원본 DiT | PixArt (adaLN-single) |
|---|---|---|
| timestep → modulation 계산 | block마다 별도 `Linear(D, 6D)` 28개 | 모델 전체에 단 1개 `Linear(D, 6D)` (`t_block`) |
| block당 추가 학습 파라미터 | 6·D² (Linear) | **6·D (scale_shift_table만)** |
| 총 modulation 파라미터 (D=1152, 28 block) | ~28×6·D² ≈ **223M** | 1×6·D² + 28×6·D ≈ **8M** |
| 전체 모델 크기 | 0.68B | **0.61B (≈ −26% 파라미터)** |

코드 ([PixArt.py:85-88](https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt.py#L85-L88)):
```python
# 모델 전체에 한 번만 정의 — 모든 block이 공유
self.t_block = nn.Sequential(
    nn.SiLU(),
    nn.Linear(hidden_size, 6 * hidden_size, bias=True)
)
```

### 3. Reparameterization 트릭

새 구조(adaLN-single + cross-attn 추가)와 사전학습 DiT의 동작이 **학습 시작 시점에 동일**하도록 만드는 초기화:

| 모듈 | 초기화 방법 | 의미 |
|---|---|---|
| `scale_shift_table` | 0으로 설정 | block별 보정 없음 → t의 글로벌 modulation만 적용 |
| `t_block` Linear | 사전학습 DiT의 평균 modulation 출력을 t=500에서 재현하도록 초기화 | 새 구조가 평균적으로 원본과 비슷한 modulation 출력 |
| `cross_attn.proj.weight/bias` | **0으로 초기화** | cross-attn 출력=0 → residual 더해도 변화 없음 → **텍스트 분기가 처음엔 무력화** |
| `final_layer.linear.weight/bias` | 0으로 초기화 | 출력 layer가 0에서 시작 |
| Self-Attn, MLP weight | **DiT pretrained 그대로** | ImageNet에서 배운 픽셀 패턴 보존 |

**비유**: 새 직원(cross-attention)을 회사(DiT)에 데려왔는데, 일단 첫날엔 아무 말도 못 하게(출력 0) 하고 회의만 듣게 한다. 회사는 평소처럼 돌아감. 학습이 진행되면서 그 직원이 점점 자기 의견(텍스트 정보)을 내놓기 시작하고, 회사 운영이 그 의견에 적응해감.

코드 ([PixArt.py:202-210](https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt.py#L202-L210)):
```python
# Zero-out cross-attention output projection
for block in self.blocks:
    nn.init.constant_(block.cross_attn.proj.weight, 0)
    nn.init.constant_(block.cross_attn.proj.bias, 0)

# Zero-out output layers
nn.init.constant_(self.final_layer.linear.weight, 0)
nn.init.constant_(self.final_layer.linear.bias, 0)
```

### 4. Cross-Attention 구현

[diffusion/model/nets/PixArt_blocks.py:29-71](https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py#L29-L71):

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, ...):
        self.q_linear  = nn.Linear(d_model, d_model)       # 이미지 → query 변환
        self.kv_linear = nn.Linear(d_model, d_model * 2)   # 텍스트 → key, value 변환
        self.proj      = nn.Linear(d_model, d_model)       # 출력 projection (init=0)

    def forward(self, x, cond, mask=None):
        # x: 이미지 토큰 (query), cond: T5 텍스트 임베딩 (key/value)
        # padding 토큰은 BlockDiagonalMask로 무시
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask) if mask is not None else None
        x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = self.proj(x.view(B, -1, C))                     # init=0이라 처음엔 출력 0
        return x
```

**핵심**: query는 이미지 토큰(노이즈 낀 latent), key/value는 T5에서 나온 텍스트 토큰. 텍스트 길이는 최대 120 토큰으로 고정, padding은 마스킹. xFormers의 memory-efficient attention으로 메모리 절감.

### 5. 모델 사양

| 항목 | 값 |
|---|---|
| 백본 | PixArt-XL/2 (DiT-XL/2 기반) |
| 파라미터 | **0.6B (610M)** |
| Depth × Hidden size | 28 × 1152 |
| Attention heads | 16 (head dim 72) |
| Patch size | 2 |
| Input latent (1024px용) | 32×32 (또는 multi-scale) |
| 텍스트 인코더 | T5-XXL (Flan-T5-XXL, 4.3B), frozen |
| 텍스트 토큰 길이 | 120 |
| 텍스트 임베딩 차원 | 4096 → CaptionEmbedder로 1152 투영 |
| VAE | sd-vae-ft-ema (latent 채널 4, 압축비 8) |
| Predict variance | ✅ (`pred_sigma=True`, 출력 채널 8) |
| Position embedding | 2D sin-cos (fixed buffer, `lewei_scale`로 보간) |

---

## 학습 전략: 3단계 분해

| Stage | 목표 | 데이터 | 이미지 수 | GPU days (V100) |
|---|---|---|---|---|
| **Stage 1: Pixel Dependency** | 픽셀 분포 학습 | ImageNet (class-conditional DiT pretrain) | 1M | 88 |
| **Stage 2: Text-Image Alignment** | 텍스트-이미지 정합 | SAM-LLaVA (개념 밀도 높은 dense caption) | 10M | 672 |
| **Stage 3: High-Res & Aesthetic** | 미적 품질 + 512→1024px | LAION-Aesthetic + JourneyDB + 내부 데이터 | 14M | 640 |
| **합계** | | | 25M | **1400 V100 days ≈ 753 A100 days** |

### 학습 단계별 핵심

**Stage 1**: 처음부터 학습하지 않고 ImageNet으로 학습된 DiT를 reparameterization으로 그대로 가져옴 → 픽셀 단계 학습 비용이 거의 0.

**Stage 2**: 캡션 품질이 가장 중요한 단계. LAION 원본(noun 6.4개/이미지)으론 정합 학습이 비효율 → SAM 이미지에 LLaVA로 dense caption (noun 30개/이미지)을 만들어 학습 효율을 끌어올림.

**Stage 3**: 512px → 1024px 해상도 확장 + 미적 데이터 추가. 학습은 짧지만 품질 향상의 핵심 단계.

### 캡션 품질 비교 (concept density 분석)

| 데이터셋 | 캡션 출처 | 평균 noun/이미지 | 총 noun 수 |
|---|---|---|---|
| LAION (원본) | 웹 alt-text | 6.4 | - |
| LAION-LLaVA | LLaVA 재캡션 | ~13 | - |
| **SAM-LLaVA** | LLaVA로 SAM에 캡션 | **30** | **328M** |

→ SAM이 객체 밀도가 높아서 LAION에 LLaVA를 돌린 것보다도 2배 이상 dense.

### 학습 가속 트릭

- **T5 + VAE feature 사전 추출**: 학습 전 모든 이미지/캡션을 인코딩해 디스크 저장 → 학습 중 transformer만 forward
- **Mixed precision (fp16)** + **gradient checkpointing**: VRAM 절약
- **xFormers memory-efficient attention**: cross-attention에 사용
- **Multi-scale 학습 (1024px)**: aspect ratio bucket으로 다양한 종횡비 한 배치 처리
- **CFG**: 학습 시 10% 확률로 캡션을 비움 (`class_dropout_prob=0.1`)

코드 ([configs/pixart_config/PixArt_xl2_img256_SAM.py](https://github.com/PixArt-alpha/PixArt-alpha/blob/master/configs/pixart_config/PixArt_xl2_img256_SAM.py)):
```python
train_batch_size = 176         # GPU당
num_epochs = 200
gradient_clip = 0.01            # 매우 작음 (DiT 학습 안정화)
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)
```

### 학습률 (Learning Rate) 디테일

3단계 학습 시 lr이 어떻게 다뤄지는지가 헷갈리기 쉬운 부분. 정리하면 **"base_lr은 모든 stage에서 동일(2e-5), 단 effective_lr은 batch size에 따라 자동으로 달라진다"**.

#### Stage별 LR 정보 (config 파일 추출)

| Stage | 해상도 | **base_lr** | warmup | LR schedule | grad clip | batch (GPU당) |
|---|---|---|---|---|---|---|
| Stage 2 (Align) | 256 | **2e-5** | 1000 step | constant | 0.01 | 176 |
| Stage 3a (512) | 512 | **2e-5** | 1000 step | constant | 0.01 | 38~40 |
| Stage 3b (1024) | 1024 | **2e-5** | 1000 step | constant | 0.01 | 2~12 |
| PixArt-δ (LCM) | 1024 | **2e-5** | 100 step | constant | 0.01 | 16 |

→ **모든 stage에서 base_lr=2e-5, constant schedule (decay 없음), warmup 1000 step**으로 동일.

#### base_lr vs effective_lr — 가장 중요한 구분

config에 적힌 `lr=2e-5`는 **표기값(base_lr)** 일 뿐, 실제 옵티마이저가 weight에 적용하는 값은 다름.

```
실제 적용되는 lr = effective_lr
                = base_lr × auto_lr_ratio × warmup_factor
                = 2e-5    × √(BS_현재stage / base_BS) × min(step/1000, 1.0)
```

베이스 config에 명시된 **`auto_lr=dict(rule='sqrt')`** 규칙 때문에 batch size가 작아지면 effective_lr도 자동으로 작아짐. 해상도가 커지면 GPU 메모리 한계로 batch가 줄어드니, **고해상도 stage일수록 effective_lr이 자연스럽게 감소**.

#### Stage별 Effective LR (warmup 후 도달값, BS 11392 기준 추정)

| Stage | BS (global, 추정) | √(BS/base_BS) | **Effective LR** |
|---|---|---|---|
| Stage 2 (256) | 11,392 | √(11392/2048) ≈ 2.36× | **~4.7e-5** |
| Stage 3a (512) | ~2,400 | √(2400/2048) ≈ 1.08× | **~2.2e-5** |
| Stage 3b (1024) | ~768 | √(768/2048) ≈ 0.61× | **~1.2e-5** |

→ **암묵적 LR scheduling**: 명시적 decay는 없지만 batch size 감소가 자동으로 lr을 줄여줌.

#### Stage 전환 시 무엇을 이어받는가?

각 stage는 별도 학습 job이고, 전환 시 **weight만 이어받고 학습 dynamics는 새로 시작**:

| 항목 | Stage 전환 시 | 근거 |
|---|---|---|
| **모델 가중치 (weight)** | ✅ **이어받음** | `load_from = "이전 stage checkpoint"` |
| EMA weight | ✅ 선택적 이어받음 | config의 `load_ema` 옵션 |
| Optimizer state (Adam m, v) | ❌ **새로 초기화** | `resume_from = None` (기본값) |
| LR scheduler state | ❌ **새로 시작** | warmup 1000 step 다시 |
| Step counter | ❌ **0부터 다시** | epoch reset |

→ **"몸은 가져가되 페이스는 다시 정한다"** — weight(능력)는 인계, optimizer/lr(학습 dynamics)는 reset.

`load_from` (stage 전환용, weight만) vs `resume_from` (같은 stage 학습 재개용, 전부 복원)이 분리되어 있음.

#### 실제 LR 흐름 — 그림으로

```
모든 Stage 공통 출발점: step 0 → effective_lr = 0  (warmup_factor=0)
                            │
                            │  1000 step 동안 선형 ramp-up
                            ▼
                       각 stage 도달값 (BS에 따라)

Stage 2 (BS 큼):   step 0 → 0 ──warmup──→ 4.7e-5 ──→ constant 4.7e-5
Stage 3a (BS ↓):  step 0 → 0 ──warmup──→ 2.2e-5 ──→ constant 2.2e-5
Stage 3b (BS ↓↓): step 0 → 0 ──warmup──→ 1.2e-5 ──→ constant 1.2e-5
```

세 stage 모두 **출발선(step 0)에선 effective_lr=0으로 동일**하지만, 워밍업 후 **순항 lr이 stage마다 다름**.

#### 왜 작은 lr (2e-5) + constant schedule인가? — 학계 표준과 다른 비전형적 디자인

**먼저 비교** — 학계 표준 vs PixArt-α:

| 학습 종류 | LR 크기 | LR Schedule | Stage별 LR 변화 |
|---|---|---|---|
| 처음부터 학습 (from scratch) | 큼 (1e-4) | cosine decay or step | - |
| 표준 fine-tuning | 작음 (1e-5~5e-5) | cosine decay or constant | 보통 단계별 ↓ |
| Multi-stage fine-tuning (보통) | 작음 | 단계별 step decay | 1e-4 → 1e-5 → 1e-6 식 |
| **PixArt-α (예외)** | **작음 (2e-5)** | **constant (decay 없음)** | **모든 stage 동일** |

→ **PixArt-α는 multi-stage fine-tuning인데도 lr decay를 명시적으로 안 씀**. 학계 표준과 다른 선택. 어떻게 가능했나?

#### Constant LR이 가능한 5가지 안전 장치

명시적 decay 없이도 학습이 안정적인 이유 — 다섯 가지 메커니즘이 decay 역할을 대신함:

| # | 안전 장치 | 효과 |
|---|---|---|
| ① | **lr 자체가 작음 (2e-5)** | SD/Imagen의 1/5. decay 안 해도 큰 변동 불가능한 수준 |
| ② | **Reparameterization** | 학습 시작점에 DiT와 동일 출력. 초반 큰 gradient 위험 원천 차단 |
| ③ | **gradient_clip 0.01** | 일반 (1.0)의 1/100. 한 step 변동을 강제 클램핑 |
| ④ | **BS 변화의 implicit decay** | `auto_lr='sqrt'` rule로 BS 감소 → effective_lr 자동 감소 (Stage 2 ~4.7e-5 → Stage 3b ~1.2e-5) |
| ⑤ | **Stage 진입 warmup의 reset 효과** | 매 stage 시작 시 effective_lr이 0으로 reset → 1000 step ramp-up → mini-cyclic lr 효과 |

→ **"lr decay 대신 다른 메커니즘들로 같은 효과를 달성"** 한 단순함 우선 디자인.

#### 트레이드오프

| 항목 | 표준 (lr decay) | PixArt (constant + 작은 lr) |
|---|---|---|
| 학습 초반 빠른 수렴 | ✅ 큰 lr로 빠르게 | ❌ 작은 lr이라 느림 |
| 학습 후반 안정성 | ✅ decay로 정밀 조정 | ✅ 이미 작아서 안정 |
| 사전학습 손상 위험 | 🟡 초반 큰 lr이 위험 | ✅ 작은 lr로 보존 |
| 구현 복잡도 | 🟡 schedule 설계 필요 | ✅ 매우 단순 |
| 하이퍼파라미터 수 | 많음 (peak_lr, min_lr, decay_steps) | **적음 (lr 하나)** |

→ "**빠른 초기 수렴**"을 포기하고 "**보존 + 단순함**"을 가져간 트레이드오프. PixArt-α의 핵심 목표인 **사전학습 효과 보존**과 잘 맞음.

#### 비유

- **일반 fine-tuning** = 자동변속기 (lr scheduler가 알아서 변속)
- **PixArt-α** = 수동변속기 (constant lr) + 자동 페이스 조절 (BS 기반 auto_lr) + 안전벨트 (grad_clip + reparam)

> 💡 **핵심 통찰**: stage 간 effective_lr이 달라 보이는 건 의도된 LR scheduling이 아니라 **batch size 변화의 부산물**. **BS가 같다면 stage 간 effective_lr도 완전히 동일**. → 자세히는 [Q17](#q17-stage-전환-시-lr을-이어받나-bs가-같으면-stage-간-lr이-동일한가) 참조.

---

## 실험 요약

### 1. 학습 비용 비교 (FID-30K MS-COCO)

| 모델 | 타입 | 파라미터 | 학습 이미지 | FID-30K ↓ | A100 GPU days |
|---|---|---|---|---|---|
| DALL·E | Diff | 12.0B | 250M | 27.50 | - |
| GLIDE | Diff | 5.0B | 250M | 12.24 | - |
| LDM | Diff | 1.4B | 400M | 12.64 | - |
| DALL·E 2 | Diff | 6.5B | 650M | 10.39 | 41,660 |
| **SDv1.5** | Diff | 0.9B | 2,000M | 9.62 | **6,250** |
| GigaGAN | GAN | 0.9B | 2,700M | 9.09 | 4,783 |
| Imagen | Diff | 3.0B | 860M | 7.27 | 7,132 |
| RAPHAEL | Diff | 3.0B | 5,000M+ | 6.61 | 60,000 |
| **PixArt-α (zero-shot)** | Diff | **0.6B** | **25M** | **7.32** | **753** |
| **PixArt-α (COCO FT)** | Diff | 0.6B | 25M | **5.51** | 753 |

**핵심 수치**: SD 1.5의 1/80 학습 이미지(25M vs 2000M), 12% 학습 비용으로 FID 7.32 (Imagen 7.27과 동급). COCO fine-tune 시 RAPHAEL(6.61)도 넘어 5.51.

### 2. T2I-CompBench (구성 능력 평가)

| 모델 | Color ↑ | Shape ↑ | Texture ↑ | Spatial ↑ | Non-spatial ↑ | Complex ↑ |
|---|---|---|---|---|---|---|
| SD v1.4 | 0.3765 | 0.3576 | 0.4156 | 0.1246 | 0.3079 | 0.3080 |
| SD v2 | 0.5065 | 0.4221 | 0.4922 | 0.1342 | 0.3096 | 0.3386 |
| SDXL | 0.6369 | 0.5408 | 0.5637 | 0.2032 | 0.3110 | 0.4091 |
| **PixArt-α** | **0.6886** | **0.5582** | **0.7044** | **0.2082** | **0.3179** | **0.4117** |

→ 거의 모든 항목에서 SDXL 능가. 특히 **Texture에서 압도적 (0.7044 vs SDXL 0.5637)**. dense caption 학습 효과로 보임.

### 3. User Study (vs SDv2)

| 비교 항목 | PixArt-α 우위 |
|---|---|
| 이미지 품질 (Quality) | **+7.2%** |
| 텍스트 정합 (Alignment) | **+42.4%** |

특히 정합에서 큰 우위 — SAM-LLaVA 학습의 효과.

### 4. 추론 속도 (1024×1024)

| 하드웨어 | PixArt-δ LCM (4 step) | PixArt-α (14 step) | SDXL standard (25 step) |
|---|---|---|---|
| T4 (Colab 무료) | 3.3s | 16.0s | 26.5s |
| V100 (32GB) | 0.8s | 5.5s | 7.7s |
| A100 (80GB) | **0.51s** | 2.2s | 3.8s |

PixArt-α는 step 수가 14로 SDXL의 25보다 적어 추론도 빠름. LCM distillation한 PixArt-δ는 A100 0.5초.

---

## 코드 구조

```
PixArt-alpha/
├── diffusion/
│   ├── model/
│   │   ├── nets/
│   │   │   ├── PixArt.py             ← 메인 모델 (PixArtBlock, PixArt class)
│   │   │   ├── PixArt_blocks.py      ← MultiHeadCrossAttention, T2IFinalLayer, CaptionEmbedder
│   │   │   ├── PixArtMS.py           ← 1024 multi-scale 버전 (SizeEmbedder 추가)
│   │   │   └── pixart_controlnet.py  ← PixArt-δ ControlNet
│   │   ├── t5.py                     ← T5-XXL wrapper
│   │   ├── llava/                    ← LLaVA captioning
│   │   └── builder.py                ← @MODELS.register_module() 등록 시스템
│   ├── data/                          ← Dataset (SAM, InternalData)
│   ├── dpm_solver.py / sa_sampler.py / iddpm.py  ← 샘플러
│   └── lcm_scheduler.py               ← LCM scheduler (PixArt-δ)
├── configs/pixart_config/
│   ├── PixArt_xl2_img256_SAM.py       ← Stage 2 config (lr=2e-5, bs=176, 200 epoch)
│   ├── PixArt_xl2_img1024_internalms.py  ← Stage 3 multi-scale
│   └── PixArt_xl2_img1024_lcm.py      ← PixArt-δ
├── train_scripts/
│   ├── train.py                       ← 메인 학습
│   ├── train_diffusers.py             ← Diffusers 통합
│   ├── train_pixart_lcm.py            ← LCM distillation
│   └── train_pixart_lora_hf.py        ← LoRA
├── tools/
│   ├── extract_features.py            ← T5/VAE feature 사전 추출
│   ├── VLM_caption_lightning.py       ← LLaVA dense captioning
│   └── convert_pixart_alpha_to_diffusers.py
└── app/app.py                         ← Gradio 데모
```

### 학습 워크플로우 (사용자 직접 학습)

**Step 1**: 데이터 준비 (SAM 예)
```
data/SA1B/
├── images/sa_xxxxx.jpg                            (원본 이미지)
├── captions/sa_xxxxx.txt                          (LLaVA dense caption)
├── partition/part0.txt                            (이미지 이름 리스트)
├── caption_feature_wmask/sa_xxxxx.npz             (T5 추출 결과)
└── img_vae_feature/train_vae_256/noflip/sa_xxxxx.npy  (VAE 추출 결과)
```

**Step 2**: T5/VAE feature 사전 추출
```bash
python tools/extract_features.py --img_size=256 --json_path "data/data_info.json" ...
```

**Step 3**: 분산 학습 시작
```bash
python -m torch.distributed.launch --nproc_per_node=2 train_scripts/train.py \
  configs/pixart_config/PixArt_xl2_img256_SAM.py --work-dir output/train_SAM_256
```

### 추론 (diffusers)
```python
from diffusers import PixArtAlphaPipeline
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16
).to("cuda")
image = pipe("A small cactus with a happy face in the Sahara desert.").images[0]
```

**VRAM 요구**: 자체 repo 23GB, Diffusers 11GB, full offloading **<8GB**.

---

## 💬 Q&A 섹션

### Q1. PixArt-α는 기존 DiT에 그냥 cross-attention만 추가한 구조인가?

**본질적으론 맞지만, 정확히는 "추가 1 + 수정 3"이 같이 일어남.**

| # | 변경 종류 | 변경 내용 | 의도 |
|---|---|---|---|
| ① | **추가** | Cross-Attention layer (NEW) | 텍스트 조건 받기 |
| ② | **수정** | adaLN → adaLN-single | 파라미터 26% 절감 |
| ③ | **제거** | class_embedding 분기 | 텍스트는 cross-attn으로 가니까 |
| ④ | **수정** | FinalLayer → T2IFinalLayer | adaLN-single 패턴 일관성 |

**핵심**: **Self-Attention과 MLP는 손대지 않음**. 이게 DiT pretrained 가중치를 그대로 재활용할 수 있는 이유. Cross-attention만 zero-init으로 끼워 넣으면 학습 시작 시점에 DiT와 동일하게 동작 (= reparameterization 가능). 이 설계 철학("기존 사전학습 모델은 최대한 건드리지 말고 새 능력만 끼워 넣어라")은 이후 IP-Adapter, ControlNet, LoRA 등 거의 모든 fine-tuning 기법의 표준이 됨.

### Q2. adaLN-single이 정확히 어떻게 파라미터를 절약하나?

**원본 DiT의 adaLN**: 각 block마다 `Linear(D, 6D)` MLP를 따로 둠. 입력은 `c = t + class`. 28개 block × 6·D² ≈ 223M.

**PixArt의 adaLN-single**: 모델 전체에 `Linear(D, 6D)` 단 1개(`t_block`). 각 block은 자기만의 `scale_shift_table[6, D]` 작은 보정값만 추가. 1×6·D² + 28×6·D ≈ 8M.

→ modulation 파라미터 28배 절감, 전체 모델 0.68B → 0.61B (≈ -26%).

### Q3. Reparameterization은 왜 필요한가? 그냥 weight를 복사하면 안 되나?

**문제**: 원본 DiT는 `condition = t + class`로 modulation을 만든다. PixArt는 class를 없애고 `t`만 쓴다 + adaLN-single로 구조도 다르다. 단순 weight 복사 시 새 구조의 forward 출력이 원본과 달라 학습이 망가짐.

**해결책**: 새 구조가 **학습 시작 시점에 원본과 정확히 동일한 출력을 내도록** 초기화한다. 자세히는 알고리즘 섹션 3번 참조.

학습이 진행되면서 0이었던 부분들이 텍스트 정보를 받아들이도록 점차 활성화됨 → **사전학습 효과를 최대한 보존하면서 새 능력만 점진적으로 추가**.

### Q4. adaLN-single을 씀으로써 단점은 없나?

**있음**. 7가지로 정리:

1. **표현력(capacity) 손실** — block당 학습 가능한 modulation 파라미터가 약 1000배 작음 (8M → 7K). 원리상 표현력 손실. 다만 T2I 디퓨전에서는 block별 modulation이 거의 비슷한 패턴이라 실제로는 문제 안 됨 (저자 실험).

2. **Block 간 동작 다양성 제약** — 초기 block(전체 구도)과 후기 block(디테일)이 timestep을 다르게 해석해야 할 수도 있는데, 작은 `scale_shift_table` 보정만으론 함수 형태 자체를 못 바꿈. **공장 표준 가구를 받고 위치만 살짝 옮기는 정도**의 자유도.

3. **추가 조건 주입 어려움** — `t_block` 입력이 timestep만이라, 다른 글로벌 조건(해상도, 종횡비)을 modulation에 넣으려면 입력을 확장해야 함. 실제로 PixArtMS는 SizeEmbedder를 추가해야 했음.

4. **표준 DiT 호환성 문제** — PixArt 가중치는 adaLN-single 구조 전용. 표준 DiT 체크포인트와 직접 호환 X. Reparameterization은 ImageNet DiT → PixArt 일회용 변환.

5. **Reparameterization 부수 비용** — 학습 초반엔 cross-attn proj=0이라 텍스트 영향이 0에서 점진적으로 커짐. 학습 효율이 살짝 떨어지는 구간 존재 (단 수천 step 안에 해소).

6. **매우 깊은 모델 확장 한계 가능성** — depth가 매우 깊으면 모든 block이 같은 글로벌 modulation 공유하는 부담. [[paper-mv-split-dit]] 같은 1000층 모델은 별도 안정화 기법(MV-Split residual) 필요.

7. **Modulation 해석 가능성 손실** — 글로벌 함수 + 작은 offset으로 분리돼 어디서 어떤 동작이 나오는지 추적하기 어려움.

**후속 모델들의 보완**:
- **FLUX**: timestep + **text pooled embedding** 을 합쳐 글로벌 modulation 입력 확장
- **SD3 (MMDiT)**: modality별 modulation MLP 2개 (image/text 각각)
- **Z-Image, HiDream-O1**: modulation 입력을 더 풍부하게 확장하는 방향

→ 후속 모델들은 **adaLN-single의 효율성은 유지하되 글로벌 modulation 입력을 풍부하게** 만드는 방향으로 진화. PixArt-α의 단순화가 약간 극단적이었음을 후속작들이 살짝 되돌린 셈.

### Q5. MM-DiT(SD3/FLUX)의 concat 방식 대신 cross-attention으로 회귀하면 멀티모달에 문제 없나?

**핵심**: T2I 단일 태스크엔 cross-attn이 효율적이지만, **멀티모달 확장 시 분명한 문제들이 있다**.

#### Cross-Attention vs MM-DiT 구조 차이

```
[PixArt-α (Cross-Attn)]                      [SD3/FLUX (MM-DiT, Joint Attn)]

이미지 ──Self-Attn──→ Cross-Attn ──→ MLP     이미지 ─┐
                       ↑                              concat → Self-Attn → split → 각자 MLP
                       텍스트 (frozen, k/v만)  텍스트 ─┘     ↑
                                                              둘 다 업데이트됨

→ 단방향 (이미지가 텍스트 봄)                  → 양방향 (서로 봄)
```

#### MM-DiT가 concat을 쓰는 5가지 이유

1. **양방향 정보 흐름**: 텍스트도 layer마다 이미지를 보고 자기 표현을 업데이트. Cross-attn은 텍스트가 frozen이라 "이미지가 만들어지는 단계에 따라 텍스트 강조가 달라지는" 효과 X.

2. **깊은 결합 (deep fusion)**: 이미지와 텍스트가 같은 query/key/value 공간에 매핑되어 joint representation 형성. 편집·IP-preservation·인페인팅 같은 멀티모달 태스크에서 자연스러움.

3. **새 modality 추가가 쉬움**: `[image, text, reference, depth, ...]` concat 한 번이면 끝. Cross-attn은 modality마다 별도 cross-attn layer 추가 필요 → 구조 복잡도 증가.

4. **텍스트 길이 유연성**: MM-DiT는 sequence가 동적이라 가변 텍스트 길이 자연스럽게 지원. Cross-attn은 토큰 길이 고정 (PixArt 120).

5. **모든 layer에서 텍스트-이미지 매칭 신호 학습**: self-attn과 cross-attn이 분리되지 않음.

#### Cross-Attention의 멀티모달 확장 시 5가지 문제

1. **새 modality마다 별도 cross-attn layer 필요 → 구조 폭발**: modality N개면 cross-attn N개. IP-Adapter는 PixArt에 새 cross-attn 별도 학습 필요.

2. **텍스트가 frozen이라 상호 참조 어려움**: "이 사람을 빨간 옷으로" + 참조 이미지일 때, "이 사람" 토큰이 참조 이미지를 가리켜야 하는데 cross-attn 방식에선 어려움.

3. **편집·인페인팅 시 일관성 부족**: 원본 이미지 표현과 새 텍스트 지시가 별도 공간에 살아 융합이 표면적. 편집 부분과 안 된 부분 경계가 부자연스러울 수 있음.

4. **긴 텍스트·복잡 reasoning 한계**: 멀티모달 reasoning task에서 텍스트가 매우 길어지면 토큰 길이 고정 한계.

5. **비디오·3D 등 차원 확장**: 시간-공간-텍스트 attention을 따로 두는 복잡 구조 필요 (SVD, AnimateDiff). MM-DiT는 sequence에 concat으로 자연스럽게 처리.

#### PixArt-α가 cross-attention을 선택한 5가지 이유 (2023년 시점)

1. **T5-XXL이 너무 강력**: 이미 강력한 텍스트 표현이라 model 안에서 다시 업데이트할 필요 적음.
2. **계산 효율**: self-attn sequence가 이미지 토큰만이라 짧음.
3. **사전학습 DiT 호환성**: cross-attn을 새 layer로 끼워 넣는 게 reparameterization과 가장 잘 맞음.
4. **학습 안정성**: 단방향이라 학습이 안정적.
5. **단일 modality 조건이면 충분**: PixArt 목표는 T2I 한 가지였음.

#### 결론

**T2I 단일 태스크엔 cross-attn이 효율적·단순. 멀티모달(편집·IP·ControlNet·비디오·3D)로 가면 MM-DiT 본질적으로 유리.** PixArt-α의 선택은 2023년 T2I 단일 태스크에 맞는 영리한 trade-off였지만, 그 trade-off가 멀티모달 시대의 후속 모델들(SD3, FLUX, Z-Image, HiDream-O1)이 MM-DiT 또는 그 변형으로 옮겨간 이유.

흥미롭게도 **PixArt-Σ는 여전히 cross-attention 유지** — 사전학습 가중치 재활용 철학과 단일 T2I 태스크 가정을 그대로 계승.

### Q6. 왜 T5-XXL을 쓰나? CLIP은 안 되나?

- **T5-XXL (Flan-T5-XXL, 4.3B)**: 큰 텍스트 인코더로 장문 캡션의 디테일까지 인코딩. Imagen이 처음 입증.
- **CLIP**: 짧은 캡션 최적화, 인코더가 상대적으로 작음 (~63M). 디테일 손실.

PixArt-α는 dense caption (noun 30개/이미지)을 다뤄서 T5 같은 큰 텍스트 인코더가 필수. 토큰 길이도 120 (CLIP은 보통 77). 추론 시 T5 weight 안 올리는 8GB VRAM 모드 지원 (T5 feature 미리 캐싱).

### Q7. SAM 데이터셋을 왜 썼나? LAION이 훨씬 큰데?

**LAION의 문제**: 웹 alt-text 기반이라 캡션이 짧고 노이즈 많음 (noun 6.4개). 한 이미지에 한 객체만 언급되는 경우 다수 → 개념 밀도 낮음 → 정합 학습 비효율.

**SAM의 강점**: Segment Anything 데이터셋은 **객체가 빽빽하게 들어찬 이미지** 1100만 장 (한 이미지에 객체 100여 개 평균). 단 캡션이 없음 → LLaVA로 dense caption 자동 생성 → noun 30개/이미지. 같은 GPU 시간에 정합을 4~5배 빨리 학습.

이 데이터셋이 **SAM-LLaVA-Captions10M**으로 공개됨. 이후 DALL·E 3, SD3, FLUX의 표준 전략(합성 캡션 활용)으로 정착.

### Q8. PixArt-α의 학습 비용이 진짜 그렇게 적은가? 숨겨진 비용은 없나?

**명시된 비용**: 753 A100 days, $28K, CO2 90% 감축.

**유의할 점**:
1. ImageNet pretrained DiT 가중치를 출발점으로 씀 — DiT 자체 학습 비용 별도 (~150 V100 days). 포함해도 SD 1.5의 13%.
2. LLaVA로 SAM 1100만 장 캡션 생성 — GPU 시간 필요하지만 일회성, SAM-LLaVA로 공개돼 재사용 가능.
3. T5+VAE feature 사전 추출 — 학습 가속용, 추가 비용 작음.

→ "재현"하려는 입장에선 753 + α 정도지만, **공개된 자원(SAM-LLaVA, DiT pretrained)을 그대로 활용하면 정확히 753 days로 재현 가능**한 게 핵심 의의.

### Q9. PixArt-α의 한계 (Limitations)는?

논문이 명시한 한계:

1. **객체 수 세기 (counting)** — "사과 3개"를 정확히 3개 그리는 능력 약함. dense caption이 개수보다 종류·속성에 집중하기 때문.
2. **사람 손/팔 (human limbs)** — 손가락 수, 팔 방향 디테일 깨짐 (SD/Imagen도 공통).
3. **텍스트 렌더링** — 이미지 안에 글자 정확히 그리는 능력 약함. 폰트 관련 데이터 부족 (DALL·E 3, Imagen은 별도 강화).
4. **모델 크기 (0.6B)** — SDXL(2.6B), Imagen(3B), DALL·E 2(6.5B) 대비 작아서 디테일 표현 상대적 한계. 후속작 PixArt-Σ는 0.6B 유지하지만 데이터 33M 확장으로 일부 보완.
5. **학습 데이터 25M의 한계** — 매우 희귀한 개념(특수 직업, 지역 문화 등) 표현 약함.

### Q10. 후속 연구 (PixArt-δ, PixArt-Σ) 와의 차이는?

| 모델 | 공개일 | 핵심 변경 |
|---|---|---|
| **PixArt-α** | 2023-09 (ICLR 2024) | 본 논문. 753 A100 days로 SOTA급 T2I |
| **PixArt-δ** | 2024-01 | α 위에 **LCM (Latent Consistency Model)** + ControlNet 추가. 1024px 0.5초 (4 step, A100) |
| **PixArt-Σ** | 2024-03 | 0.6B 유지 + 4K 해상도 지원. **KV compression**으로 길이 확장. 데이터 33M로 확장 |

PixArt 시리즈 공통 철학: "**작은 모델 + 좋은 데이터 + 점진적 학습 + 사전학습 재활용**". cross-attention 방식도 전 시리즈 일관 유지 — 멀티모달이 핵심 use case가 아니라서.

### Q11. DiT vs U-Net 디퓨전, 어떤 게 더 나은가?

| 항목 | U-Net (SD 계열) | DiT (PixArt 계열) |
|---|---|---|
| 구조 | Conv + Attention (멀티스케일 다운/업샘플) | 순수 Transformer (단일 해상도 토큰) |
| 스케일링 | 어려움 (Conv 채널 증가 → 메모리 폭증) | **잘됨** (depth/width 증가에 선형) |
| 텍스트 주입 | cross-attention | cross-attention (PixArt) 또는 joint attention (SD3) |
| 학습 효율 | 낮음 (큰 데이터 필요) | **높음** (PixArt 검증) |
| 1024px 추론 속도 | ~7s (SDXL A100) | **~2s** (PixArt-α A100) |

**결론**: PixArt-α가 입증한 건 "**Transformer 백본 + 좋은 학습 전략으로 U-Net보다 훨씬 효율적**". 이 흐름이 SD3, FLUX, Z-Image, HiDream-O1로 이어짐.

### Q12. PixArt-α가 이후 모델들(SD3, FLUX, Z-Image 등)에 끼친 영향은?

| 기법 | 영향받은 모델 |
|---|---|
| **adaLN-single 패턴** | SD3, FLUX, Z-Image 의 modulation 설계 (단 글로벌 입력은 확장) |
| **DiT + cross-attention** | PixArt-Σ 그대로 유지. SD3/FLUX는 MMDiT로 대체. |
| **합성 캡션 (synthetic caption)** | DALL·E 3, SD3, FLUX 표준 |
| **사전학습 가중치 재활용** | HiDream-O1 (Qwen3-VL backbone), Z-Image |
| **소규모 데이터 + 효율적 학습** | Z-Image (6B, 314K H800h) 직접적 영감 |

### Q13. 사전학습 DiT를 그대로 쓰면 성능이 좋아지는 게 당연한 거 아닌가? FAIR한 비교 맞나?

**비판은 정당함**. 사전학습 모델을 출발점으로 쓰면 학습 효율이 좋아지는 건 당연. 그러나 PixArt-α의 진짜 기여는 "유리해진다"가 아니라 **"다른 구조(DiT) 모델 가중치를 텍스트 조건부 T2I에 어떻게 깔끔히 옮기느냐"의 방법론(reparameterization)**.

**핵심 반론 4가지**:

1. **다른 T2I 모델들도 사전학습 자원 활용** — SD 1.5는 CLIP frozen + LDM 사전학습, Imagen은 T5-XXL frozen. PixArt-α만 사전학습 쓴 게 아님. 단지 백본(DiT) 자체를 옮긴 게 새로움.

2. **DiT pretrain 비용 포함해도 SD 1.5의 13%**:
   - PixArt-α 직접 학습: 753 A100 days
   - + DiT ImageNet pretrain: 약 80 A100 days
   - **합계 ~833 days vs SD 1.5의 6,250 days → 13% 수준**

3. **데이터 효율은 사전학습으로 다 설명 안 됨** — PixArt-α는 25M, SD 1.5는 2,000M. **1/80 데이터**. 이 격차는 사전학습 활용만으론 설명 못 함. SAM-LLaVA dense caption + 3단계 학습 분해의 효과.

4. **이전 시도들은 잘 안 됐다** — Masked-DiT, U-ViT 등 PixArt 이전 DiT 기반 T2I 시도들은 학습 효율이 SD보다 별로. PixArt-α 이전엔 **DiT pretrained → T2I 변환 표준이 없었음**. PixArt-α가 처음 깔끔히 풀어내서 SD3, FLUX, Z-Image 모두 이 패턴 차용.

**논문이 더 강조했어야 할 부분**: DiT pretrain 비용을 학습 비용 비교에 명시적으로 포함했어야 함 (753 days만이 아니라 "DiT pretrain 포함 ~833 days"). 후속작들(PixArt-Σ, Z-Image)은 이 비판을 의식해 사전학습 비용을 더 명시적으로 다룸.

### Q14. Cross-attention을 추가했는데 왜 모델이 더 작아? DiT보다 커져야 하지 않나?

**실제로는 정반대로 DiT보다 65M 작음** (DiT-XL/2 0.68B → PixArt-XL/2 0.61B). 영리한 "추가 + 절감" 동시 진행 결과.

**파라미터 변화 분해** (D=1152, depth=28):

| 변경 | 파라미터 변화 | 세부 |
|---|---|---|
| ② adaLN → adaLN-single | **-215M** | 28×6·D² (223M) → 1×6·D² + 28×6·D (8M) |
| ① Cross-Attention 추가 | **+148M** | block당 q+kv+proj ≈ 5.3M × 28 block |
| ③ class_embed → Caption Embedder | +5M | nn.Embedding(1000,D) 제거, Caption MLP 추가 |
| ④ FinalLayer → T2IFinalLayer | -2M | adaLN-single 패턴으로 재구성 |
| **순합계** | **-64M** | 675M → 611M |

**핵심 통찰**: PixArt-α는 **"새 기능 추가" + "기존 비효율 제거"를 동시에** 했음. 일반적인 설계는 "기능 추가 시 파라미터 증가"가 자연스러운데, PixArt-α는 modulation MLP 절감(-215M)이 cross-attn 추가(+148M)보다 더 커서 **net으로는 작아짐**.

**가정**: adaLN-single 없이 cross-attn만 추가했다면 0.82B가 되어 SD 1.5(0.9B)에 근접 → "**작은 모델로 SOTA**" 메시지가 약해졌을 것. adaLN-single이 cross-attn 추가 비용을 상쇄했기에 PixArt-α의 핵심 인상이 가능.

DiT의 modulation이 전체 파라미터의 **33%(223M)** 였고, 저자들이 "**block 28개가 사실상 비슷한 modulation 동작**"임을 실험으로 발견 → 글로벌 1개 + 작은 보정으로 충분하다는 통찰.

### Q15. 본질 한 문장으로 정리하면?

**"기존 모델(DiT) 이용 + 파인튜닝 + 구조 변화로 더 큰 모델 따라잡기"** 가 정확한 본질. 다만 세 가지 정밀화:

1. **"파인튜닝"이라기보단 "능력 추가형 학습"** — 일반 fine-tuning은 기존 능력 유지 + 약간 조정. PixArt-α는 완전히 다른 modality(텍스트) 처리 능력을 cross-attn으로 새로 학습 (init=0에서 출발). LoRA와 다른 패턴.

2. **"구조 변화"는 양방향** — 새 기능 추가(cross-attn +148M) + 기존 비효율 제거(adaLN-single -215M)의 동시 진행. → Q14 참조.

3. **"따라잡았다"의 정밀한 의미**:

| 비교 차원 | PixArt-α 0.6B | 따라잡은 대상 |
|---|---|---|
| 이미지 품질 (FID) | 7.32 | Imagen 3B (7.27) — **동급** |
| 텍스트 정합 | 0.69 (Color) | SDXL 2.6B (0.64) — **능가** |
| 학습 비용 | 753 days | SD 1.5 6,250 days — **12%** |
| 추론 속도 | 2.2s | SDXL 3.8s — **빠름** |

→ "**완벽한 SOTA 추월**"이 아니라 **"동급/일부 능가 + 압도적 효율"**. "적은 비용으로 큰 모델 성능을 따라잡았다"가 가장 정확.

**최종 한 문장**: 이미지 패턴은 이미 아는 사전학습 DiT(0.68B)를 출발점으로, 텍스트 능력은 cross-attention 한 layer만 새로 끼워 넣고(+148M), 동시에 modulation MLP의 비효율을 제거(-215M)해서, 결과적으로 더 작은 0.61B 모델로 3단계 점진 학습을 거쳐 자기보다 4~5배 큰 SDXL·Imagen·Midjourney급 품질을 1/10 비용에 달성한 논문.

### Q16. 이어서 학습하면 더 향상되는 거 아닌가?

**맞다. 그리고 후속작들이 정확히 그 방향으로 진화**. 다만 한계 3가지가 있어 무한정 좋아지진 않음.

**후속작 진화 경로**:

| 모델 | 변경 방향 | 결과 |
|---|---|---|
| **PixArt-Σ** (2024-03) | 모델 0.6B 유지 + 데이터 25M → 33M + 4K 해상도 | SDXL 능가 |
| **Z-Image** (2025-11) | 모델 6B로 10배 키움 + 더 많은 학습 | SOTA |
| **HiDream-O1** (2026-05) | LLM backbone (Qwen3-VL 8B)으로 modality 확장 | 멀티모달 통합 |

→ 사용자 직관 그대로 "**기존 모델 재활용 + 능력 추가 + 이어서 학습**"이 현대 생성 모델의 표준 설계 철학.

**일반화된 패턴** ([[reference-pretrained-backbone-reuse-landscape]] 참조):

| 모델 | 재활용 백본 | 추가 능력 |
|---|---|---|
| PixArt-α | ImageNet DiT | 자유 문장 → 이미지 |
| IP-Adapter | 기존 T2I | 참조 이미지 (스타일/인물) |
| ControlNet | SD 1.5 | 외부 조건 (엣지/깊이/포즈) |
| LoRA | SD/PixArt | 특정 스타일/캐릭터 |
| PixArt-δ | PixArt-α | LCM (4-step 추론) |
| HiDream-O1 | Qwen3-VL LLM | 이미지 생성 |

**무한 향상의 3가지 한계**:

1. **Catastrophic Forgetting (재앙적 망각)** — 텍스트로 너무 오래 학습 시 원래 픽셀 능력 손상 가능. PixArt-α의 reparameterization(cross-attn zero-init)이 이 문제 완화책.

2. **모델 크기 천장** — 0.6B는 표현력 본질적 한계. 데이터 늘려도 SDXL/FLUX의 디테일을 완전히 따라잡기 어려움. → 그래서 FLUX(12B), Z-Image(6B), HiDream-O1(8B)는 모델 크기를 키우는 방향 선택.

3. **데이터 다양성 한계** — SAM 1100만 장은 일상 사진 편향. 추상화/일러스트/만화 등에 약함.

→ 어느 지점부터는 **모델 크기 확장 / 새 학습 패러다임(RLHF, DMD distillation) / 멀티모달 확장** 이 필요. PixArt 시리즈는 "0.6B 유지"를 정체성으로 가져가지만, 시장은 두 방향(작고 효율적 vs 크고 품질 좋음)으로 갈라짐.

### Q17. Stage 전환 시 LR을 이어받나? BS가 같으면 stage 간 LR이 동일한가?

**답을 두 부분으로 나누면 명확함**:

#### (1) Stage 전환 시 무엇을 이어받나?

| 항목 | 이어받음? | 비유 (자동차) |
|---|---|---|
| **모델 weight** | ✅ 이어받음 (`load_from`) | 차의 상태(엔진/연료) 그대로 |
| Optimizer state (Adam m, v) | ❌ 새로 초기화 | 새 운전자 |
| LR scheduler state | ❌ 새로 시작 (warmup 1000 step 다시) | 페이스 다시 정함 |
| Step counter | ❌ 0부터 | - |

→ **"몸(weight)은 가져가되 페이스(lr/optimizer)는 다시 정함"**. effective_lr 자체는 인계 개념이 아니라 매 step `base_lr × √(BS/base_BS) × warmup_factor`로 새로 계산되는 함수값.

#### (2) BS가 같으면 effective_lr도 동일한가?

**✅ 맞다**. 같은 BS면 같은 auto_lr_ratio, 같은 base_lr(2e-5), 같은 warmup → **warmup 후 정상 학습 구간에서 stage 간 effective_lr 완전히 동일**.

```
effective_lr = base_lr(=2e-5) × √(BS/base_BS) × warmup_factor
                ↑ 모든 stage 동일   ↑ BS 같으면 동일      ↑ warmup 후 1.0
              → BS만 같으면 effective_lr도 stage 간 동일
```

#### (3) 그러면 왜 PixArt는 stage마다 effective_lr이 달라 보이나?

**저자가 lr을 일부러 줄인 게 아니라**, 해상도 증가로 GPU 메모리 한계에 부딪혀 **BS를 강제로 줄였고**, `auto_lr='sqrt'` rule이 그 효과를 흡수했기 때문:

| Stage | 해상도 | 토큰 수 | 메모리 부담 | 강제 BS | → effective_lr |
|---|---|---|---|---|---|
| 2 | 256 | 256 | 적음 | 큼 | 큼 |
| 3a | 512 | 1024 (4×) | 4× | 중간 | 중간 |
| 3b | 1024 | 4096 (16×) | 16× | 작음 | 작음 |

→ **"고해상도 = BS 강제 감소 = effective_lr 자동 감소"** 연쇄. PixArt-α의 stage별 lr 변화는 **의도된 LR scheduling이 아니라 메모리 제약의 부산물**. `auto_lr='sqrt'` rule이 우연히 좋은 LR scheduling 역할.

#### (4) 핵심 시사점

> **"3단계 학습"의 본질은 lr 변화가 아니라 데이터/해상도 변화**. lr decay는 부산물. 만약 메모리 무한 가정 하에 모든 stage에서 BS를 동일하게 했다면 stage 간 effective_lr도 동일했을 것이고, 그래도 학습 효과는 유사했을 것.

→ Stage별 LR 표·수식·effective_lr 추정값은 [학습 전략 섹션](#학습률-learning-rate-디테일) 참조.

---

## 핵심 인사이트 (일반화 가능한 교훈)

### 1. 새 모듈 추가 시 zero-init이 강력하다
Cross-attention 출력 projection을 0으로 초기화해서 학습 시작점에 영향을 0으로 만드는 트릭. **새 능력을 기존 모델에 끼워 넣을 때 사전학습 효과를 잃지 않는 표준 기법**. LoRA의 B 행렬 zero-init, ControlNet의 zero convolution, DiT의 final layer zero-init 모두 같은 원리.

### 2. 데이터 양보다 데이터 정보 밀도가 중요할 수 있다
SD 1.5가 20억 장 학습한 결과를 PixArt-α는 2500만 장(1.25%)으로 따라잡음. **고밀도 캡션을 가진 양질의 데이터 한 장이 노이즈 캡션 100장보다 효율적**임을 입증.

### 3. 학습은 단계로 쪼개면 효율적이다
모든 능력을 한 번에 학습하지 않고 픽셀 → 정합 → 미학으로 분리. 단계마다 최적 데이터를 쓰는 게 같은 GPU 시간으로 더 좋은 결과.

### 4. 큰 텍스트 인코더는 비용 대비 효과가 크다
T5-XXL(4.3B)을 frozen으로 쓰는 비용은 그리 크지 않지만(feature 미리 추출 가능), 짧고 정확한 텍스트 표현을 위해선 큰 인코더가 필수.

### 5. 파라미터의 33%가 같은 일을 반복하고 있을 수 있다
원본 DiT는 28개 block × 6 modulation MLP = 223M 파라미터를 modulation에만 썼는데, PixArt가 8M로 줄여도 성능에 거의 영향이 없었음. **"꼭 필요한 파라미터가 어딨는가"를 다시 물어보면 큰 절감이 가능**.

---

## 한 줄 요약 (전체)

**PixArt-α는 ImageNet pretrained DiT 가중치를 reparameterization으로 보존하면서, adaLN-single로 modulation 파라미터를 26% 절감하고, cross-attention 한 layer만 새로 끼워 넣어 T5-XXL 텍스트를 주입하며, SAM 이미지에 LLaVA로 noun 30개짜리 dense caption을 자동 생성해 3단계로 점진 학습시킨 결과, 0.6B 모델을 753 A100 days($28K)에 학습해 Imagen·SDXL·Midjourney급 품질을 달성한 ICLR 2024 Spotlight 논문이다. "기존 사전학습 모델은 최대한 건드리지 말고 새 능력만 끼워 넣어라"는 설계 철학이 이후 IP-Adapter, ControlNet, LoRA의 표준이 되었으며, adaLN-single·합성 캡션·DiT 백본 패턴은 SD3·FLUX·Z-Image·HiDream-O1 등 거의 모든 후속 T2I 모델의 표준으로 정착했다. 단 cross-attention 방식은 멀티모달 확장(편집·IP·비디오·3D) 시 단점이 분명해 후속 멀티모달 모델들은 MM-DiT joint attention으로 옮겨갔다.**

---

## 관련 메모리 링크

- [[paper-z-image]] — Z-Image (6B, 314K H800h, 2025-11). "작은 모델 + 좋은 데이터 + 효율적 학습" 철학 계승. adaLN-single 변형 사용.
- [[paper-hidream-o1-image]] — HiDream-O1 (8B, 픽셀 직접 처리). PixArt 이후의 T2I 통합 모델 흐름.
- [[paper-mv-split-dit]] — DiT 깊이 확장 (1000-layer). PixArt가 시작한 DiT 백본 트렌드의 연장선. 매우 깊은 모델에선 adaLN-single만으론 부족.
- [[feedback-paper-summary-format]] — 본 문서의 작성 형식
- [[feedback-beginner-friendly-tone]] — 본 문서의 표현 톤
- [[feedback-revert-and-clarify]] — "리뷰"와 "정리해서 저장" 구분 규칙 (이 문서 작성 전 채팅 리뷰 단계 거침)
