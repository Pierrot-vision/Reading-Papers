# PAPER: MV-Split DiT — 평균이 비명을 지르는 현상(Mean Mode Screaming) 진단과 1000층짜리 이미지 생성 트랜스포머

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | Mean Mode Screaming: Mean–Variance Split Residuals for 1000-Layer Diffusion Transformers |
| **저자** | Pengqi Lu |
| **공개일** | 2026-05-07 |
| **분량** | 43쪽 (본문 9쪽 + 부록) |
| **분야** | cs.LG, cs.CV |
| **논문 링크** | https://arxiv.org/abs/2605.06169 |
| **PDF** | https://arxiv.org/pdf/2605.06169 |
| **HTML** | https://arxiv.org/html/2605.06169v1 |
| **공식 코드** | https://github.com/erwold/mv-split |
| **추론 가중치** | `model.pt` (코드 저장소 제공) |
| **사용 외부 모델** | FLUX.2 VAE, Qwen3-0.6B 텍스트 인코더 |

---

## 📖 주요 용어 사전 (Glossary)

### 아키텍처/구조
- **노이즈에서 이미지를 만드는 모델을 트랜스포머로 구현한 구조 (DiT, Diffusion Transformer)**: 옛날엔 U자형 합성곱 신경망(U-Net, CNN 기반)을 썼지만 요즘은 트랜스포머(Transformer)로 바꿈. Stable Diffusion 3, FLUX 계열이 사용.
- **이미지와 텍스트 토큰을 한 시퀀스에 합쳐 같이 attention하는 DiT 변형 (MMDiT, MultiModal DiT)**: FLUX.2의 표준 구조.
- **층 출력에 이전 입력을 그대로 더해 학습 신호가 멀리 전달되게 하는 구조 (Residual Connection, 잔차 연결)**: `X_{l+1} = X_l + F(X_l)` 형태.
- **정규화를 잔차 더하기 전에 두는 방식 (Pre-LN, Pre-LayerNorm) / 더한 후에 두는 방식 (Post-LN)**.
- **벡터의 평균은 그대로 두고 크기만 일정 범위로 조정하는 정규화 (RMSNorm, Root-Mean-Square Normalization)**: `x / sqrt(mean(x²))`. **LayerNorm과 달리 평균을 빼지 않음**. LLaMA, FLUX, Qwen에서 표준.
- **벡터의 평균을 빼고 표준편차로 나눠 정규화하는 방식 (LayerNorm)**: 평균 제거 효과가 있음.
- **어텐션의 Q와 K를 따로 정규화해 학습을 안정화하는 기법 (QK-Norm)**.
- **위치 정보를 회전 행렬로 인코딩하는 방식 (RoPE, Rotary Position Embedding)**.
- **두 갈래 게이트로 결합한 FFN (SwiGLU)**: `Swish(xW₁) ⊙ (xW₂)` 형태.
- **여러 어텐션 헤드가 K/V를 공유해 메모리를 아끼는 attention 변형 (GQA, Grouped-Query Attention)**.

### 핵심 개념
- **깊은 DiT 학습 중 잔차의 평균 성분이 갑자기 폭주해 모델이 무너지는 현상 (MMS, Mean Mode Screaming)**: 이 논문이 처음 명명·진단.
- **시퀀스 평균에 해당하는 성분 (mean component) / 각 토큰에서 시퀀스 평균을 뺀 편차 부분 (centered component, zero-mean variation)**: 출발은 같은 잔차에서 분리되지만, **다루는 방식이 달라야 한다**가 본 논문의 핵심 주장.
- **잔차를 평균/편차로 나눠 각각 다른 게이트로 더하는 본 논문 기법 (MV-Split, Mean-Variance Split Residual)**.
- **잔차 흐름의 평균을 한 번에 갈지 않고 조금씩 흘려보내는 방식 (Leaky Trunk-Mean Replacement, 새는 평균 교체)**: `J(Z) = (1−α)·J(X) + α·J(F)` 형태로 지수이동평균(EMA, Exponential Moving Average)처럼 작동.
- **모든 토큰 벡터가 거의 같은 값으로 수렴하는 표현 붕괴 (Token Homogenization, representation collapse)**: 토큰 간 코사인 유사도(cosine similarity)가 1로 수렴.
- **하나의 벡터 방향으로만 가중치를 미는 외적 형태의 학습 업데이트 (rank-1 update, outer product)**: `ΔW = u·vᵀ`. 가중치 다양성을 죽임.
- **Softmax가 입력 전체에 같은 상수를 더해도 출력이 변하지 않는 성질 (Softmax null space, 영공간)**: `softmax(z + c·𝟙) = softmax(z)`.
- **시퀀스 토큰들의 평균을 뽑아내는 선형 사상 (J, sequence-mean projection)**.
- **평균을 뺀 편차만 남기는 사상 (P = I − J, centered projection)**.
- **잔차 경로에 쓰는 가중치의 그래디언트 (writer gradient)**: 학습 신호.

### 비교 기법 (이전에 등장한 잔차 안정화 방법들)
- **각 잔차 출력에 채널별 학습 가능한 작은 스칼라(예: 1e-4)를 곱해 안정화하는 기법 (LayerScale)**: ViT 계열 CaiT 논문에서 도입.
- **단 하나의 스칼라 게이트(처음 0으로 시작)로 잔차를 통제하는 기법 (ReZero)**.
- **Microsoft의 1000층 Transformer 안정화 기법 (DeepNorm)**: `α·x + Sublayer(x)` 형태.

### 평가 지표
- **생성 이미지 품질 지표, 낮을수록 좋음 (FID-50K, Fréchet Inception Distance)**.
- **이미지 다양성·품질 종합 지표, 높을수록 좋음 (IS, Inception Score)**.
- **각 층에서 평균 성분과 편차 성분 에너지의 비율 (ρ_T, per-layer mean/centered energy ratio)**: MMS 진단용.
- **writer 그래디언트의 평균 정렬 성분과 편차 성분 크기 (G_mean / G_ctr)**.

### 학습/추론
- **노이즈에서 이미지로 가는 경로를 미분방정식으로 따라가는 학습/추론 방식 (Flow Matching)**: 일반 디퓨전(DDPM/DDIM)의 변종.
- **미분방정식을 한 스텝씩 풀어가는 알고리즘 (ODE solver, 본 논문은 Euler 35-step)**.
- **조건 출력과 무조건 출력을 섞어 텍스트 따름을 강화하는 기법 (CFG, Classifier-Free Guidance)**.
- **이미지 ↔ 잠재공간(latent) 변환 신경망 (VAE, Variational Auto-Encoder)**.
- **GPU에서 융합 연산을 짜는 도메인 특화 언어 (Triton kernel)**.

---

## 1️⃣ 논문 한눈에 보기 (TL;DR)

> 깊이 1000층짜리 이미지 생성 트랜스포머(DiT, Diffusion Transformer)를 안정적으로 학습하려면, 잔차 연결(residual connection)에서 **시퀀스 평균 성분(mean)**과 **편차 성분(centered)**을 따로 다뤄야 한다.

**문제 발견**: 깊은 DiT를 학습하다 보면 어느 순간 **모든 토큰이 평균 방향으로 무너지는 붕괴(collapse) 현상**이 일어난다. 저자는 이걸 **평균이 비명을 지른다는 비유로 "Mean Mode Screaming(MMS)"**이라 이름 붙이고 수학적으로 진단했다.

**해결책**: 잔차에 더해지는 업데이트를 **평균 추출 사상(J, mean projection)**과 **편차 추출 사상(P = I − J, centered projection)**으로 쪼개고, 각각에 **채널별 학습 게이트(α, β ∈ ℝ^D)**를 따로 붙임. 평균 경로는 한 번에 교체하지 않고 새는 방식(leaky) — 지수이동평균(EMA) 같은 부드러운 교체.

**검증**: 이미지 생성 트랜스포머로는 사상 처음 **1000층까지 안정 학습 가능**함을 시연.

---

## 2️⃣ 핵심 기여 (Contributions)

1. **MMS 현상 발견·명명**: 깊은 DiT가 무너지는 메커니즘을 처음으로 체계화.
2. **그래디언트의 정확한 분해**: 잔차 경로 가중치의 그래디언트(writer gradient)를 **평균 정렬 성분(ΔW_μ, mean-coherent)**과 **편차 성분(ΔW_c, centered)**으로 정확히 나눔. 아래 수식이 이걸 보여주는 핵심.
   ```
   ∇_W ℒ = T·δ̄·ȳᵀ + Σ_t δ̃_t·ỹ_tᵀ
           (ΔW_μ)    (ΔW_c)
   ```
   (즉 가중치 업데이트는 두 조각으로 정확히 나뉘는데, 앞쪽은 모든 토큰을 같은 방향으로 밀고, 뒤쪽은 토큰마다 다르게 민다.)
3. **MV-Split 잔차 제안**: 평균과 편차를 분리해 각각 게이트.
4. **새는 평균 교체 메커니즘 (Leaky Trunk-Mean Replacement)**: 평균 폭주를 댐퍼처럼 완충.
5. **1000층 안정 학습 최초 시연**.

---

## 3️⃣ 주요 알고리즘: MV-Split 잔차 (Mean-Variance Split Residual)

### 3-1. 수식

**기존 방식 (Pre-LN baseline)** — 잔차에 업데이트를 통째로 더한 뒤 정규화:
```
X_{l+1} = RMSNorm(X_l + F_l)
```
(즉 이전 층 출력 `X_l`에 새 업데이트 `F_l`을 그냥 더하고 RMSNorm으로 크기만 정리.)

**MV-Split 방식 (논문 식 7)** — 평균과 편차를 분리해 별도 게이트로 더함:
```
Z_l     = X_l + β ⊙ (P · F_l) + α ⊙ J · (F_l − X_l)
X_{l+1} = RMSNorm(Z_l)
```
(즉 업데이트 `F_l`을 **편차 부분 `P·F_l`**과 **평균 변화량 `J·(F_l − X_l)`**으로 쪼개고, 각각에 채널별 가중치 `β`와 `α`를 곱해서 더한 뒤 RMSNorm.)
- `J`: 시퀀스 평균을 뽑는 사상 (sequence-mean projection)
- `P = I − J`: 평균을 뺀 편차만 남기는 사상 (centered projection)
- `α, β ∈ ℝ^D`: 채널마다 따로 학습되는 게이트 (per-channel learnable gates)

**평균 항만 따로 풀어쓰면 (새는 평균 교체, leaky replacement)**:
```
J(Z_l) = (1 − α) ⊙ J(X_l) + α ⊙ J(F_l)
```
(즉 잔차 흐름의 평균은 **이전 평균을 `1−α`만큼 유지하고, 새 평균을 `α`만큼만 흡수** — 지수이동평균(EMA)와 같은 형태.)
- α = 0 → 이전 평균 유지 (새 평균 무시)
- α = 1 → 평균 완전 교체
- 0 < α < 1 → **댐퍼처럼 새는 평균** (EMA식 부드러운 교체)

### 3-2. 코드 매핑 (`kernels/fused_mvsplit_rmsnorm.py`)

(아래 세 줄이 위 수식 그대로 구현됨. `x`는 잔차 흐름(trunk), `u`는 업데이트(update).)
```python
# x: 잔차 흐름 (trunk),  u: 업데이트 (Attn / FFN 출력)
mu_x = mean(x, dim=-2)
mu_u = mean(u, dim=-2)
y    = x + beta * (u - mu_u) + alpha * (mu_u - mu_x)
out  = RMSNorm(y)   # RMSNorm 가중치는 1로 고정, 학습 안 됨
```

### 3-3. 초기화 — 안정 학습의 비밀

(α와 β의 출발점을 다르게 둔다 — α는 닫고, β는 작게.)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `init_alpha` | **0.0** | 평균 경로(mean path)를 처음엔 완전히 닫음 |
| `init_beta` (DiTBlock 기본) | **0.03** | 편차 경로(centered path)를 매우 작게 시작 (warm-up) |
| RMSNorm 가중치 | 1.0 (학습 X, buffer) | 정규화 가중치 고정 |

### 3-4. DiTBlock 구조 (`dit.py`)
```
Attention (RoPE → QK-Norm) → FusedMVSplitNorm1 → SwiGLU FFN → FusedMVSplitNorm1
```
(즉 한 블록 안에서 어텐션과 FFN 둘 다 MV-Split 잔차를 거침.)

### 3-5. 모델 스케일
- **1000층 구성**: 50 stages × 10 blocks × 2 = 1000
- 히든 차원(hidden_size) 1024, 어텐션 헤드 8개 × 헤드당 128차원
- 코드 기본 config: 깊이 100, 너비 1280, 헤드 10 (소형 실험용)

### 3-6. 코드 전체 흐름 (Prompt → 이미지)

**Level 0 — 파이프라인 한눈에**
```
"a cat..."  ──▶  Qwen3 텍스트 인코더        (text_encoder.py)
                       │  text_emb [B, L_t, D_t]
노이즈 z_T   ──┐        │
              │        ▼
              └──▶  DiT 1000 blocks         (dit.py)   ⏎ 35회 반복
                       │  → 속도 v_θ                   (Flow Matching Euler)
                       ▼
              CFG: v = v_unc + s·(v_cond − v_unc)
              z ← z + Δt · v
                       │
                       ▼ (35 step 후)
                  FLUX.2 VAE decode          (vae.py)
                       │
                       ▼
                  RGB 이미지 (PNG)
```
(즉 텍스트 임베딩과 노이즈 latent를 DiT에 넣어 속도장(velocity field) `v_θ`를 예측하고, Euler 적분으로 35번 업데이트한 뒤 VAE로 이미지 복원.)

**Level 1 — `sample.py` 진입 순서** ([sample.py](https://github.com/erwold/mv-split/blob/main/sample.py))
1. `load_flux2_ae()` → `Qwen3TextEncoder()` → `DiT()` + checkpoint(학습 가중치 파일) 로드 (L232–243)
2. `select_jobs()` 프롬프트 선택 (L280)
3. `text_encoder.encode()` 조건/무조건 임베딩 둘 다 만듦 (L296)
4. `randn(bs, 128, H/16, W/16)` 노이즈 초기화 (L298)
5. `perform_multi_step_sampling()` 35-step Euler + CFG (L301–322)
6. `vae.decode()` → PNG 저장 (L323–329)

**Level 2 — `DiT.forward` 시퀀스** ([dit.py:451–505](https://github.com/erwold/mv-split/blob/main/dit.py))
```
z [B, 128, h/16, w/16]
  → patch_embed + norm_img_input        → x_img [B, L_img, D]
text_emb → context_proj                 → text_ctx [B, L_t, D]
concat([x_img, text_ctx], dim=1)        → x [B, L_img+L_t, D]
RoPE 생성 (이미지는 2D, 텍스트는 identity)
for block in self.blocks:               (1000회)
    x = block(x, rope, L_img)           ← DiTBlock (구조는 → 3-4)
x[:, :L_img] → final_proj → unpatchify  → v_θ [B, 128, h/16, w/16]
```
(즉 이미지를 패치 벡터로 잘라(patch embedding) 텍스트 임베딩과 합친 뒤 1000개 블록을 통과시키고, 마지막에 다시 이미지 격자로 되돌리는 단계(unpatchify) 거침.)

**Level 3 — Triton 커널의 2-pass 역전파(backward)** ([kernels/fused_mvsplit_rmsnorm.py](https://github.com/erwold/mv-split/blob/main/kernels/fused_mvsplit_rmsnorm.py))
- Forward: 각 토큰별로 segment(`l < L_img`이면 이미지, 아니면 텍스트) 판별 → 해당 segment의 `mu_x, mu_u` 로드 → 수식 (→ 3-2) 적용 → RMSNorm
- Pass A: 시퀀스를 CHUNK_L(16~128) 단위로 청크 분할, segment별 `Σ dY`와 `Σ dY·u` 누적 → 부분합 텐서 저장
- Pass B: 부분합을 합산해 segment별 `mean_dY`를 얻고, 토큰별로 `dX = dY − α·mean_dY`, `dU = β·dY + (α−β)·mean_dY` 기록
- **2-pass인 이유**: 평균(mean)은 시퀀스 전체를 봐야 계산 가능하므로 청크 간 동기화가 필요. Pass A에서 부분합 누적 → Pass B에서 일괄 적용.

### 3-7. 코드에서 논문 기여의 위치 (⭐ vs ⚪)

전체 코드 중 **논문 고유 기여는 약 5%만 차지**. 나머지는 표준 부품(ViT 패치화, Attention+RoPE+QK-Norm, SwiGLU, Qwen3, FLUX.2 VAE, Flow Matching).

**⭐ 논문 기여 (단 두 곳)**
- `DiTBlock`의 잔차 더하기 자리 ([dit.py:407–424](https://github.com/erwold/mv-split/blob/main/dit.py)) — 표준 `RMSNorm(x + op_out)` 대신 `FusedMVSplitNorm1(R, op_out, L_img)` 호출 (구조는 → 3-4 참조)
- `FusedMVSplitNorm1` 모듈 ([kernels/fused_mvsplit_rmsnorm.py](https://github.com/erwold/mv-split/blob/main/kernels/fused_mvsplit_rmsnorm.py)) — MV-Split + RMSNorm Triton 커널, **모달리티별 별도 평균 계산 (segment-aware mean)**, 2-pass 역전파 (수식 → 3-1, 3-2, 초기화 → 3-3 참조)

**⚪ 표준 부품 (논문 외 인프라)**
patch_embed/unpatchify, RoPE(이미지 2D / 텍스트 identity), QK-Norm, GQA Attention, SwiGLU FFN, Qwen3-0.6B 텍스트 인코더, FLUX.2 VAE, sample.py의 Flow Matching Euler 35-step + CFG.

**대략 라인 수 비율**
| 영역 | 라인 | 분류 |
|---|---|---|
| FusedMVSplitNorm1 (Triton + module) | ~400 | ⭐ |
| DiTBlock 잔차 교체 부분 | ~6 | ⭐ |
| Attention, RoPE, SwiGLU, QK-Norm | ~500 | ⚪ |
| patch embed, unpatchify, final proj | ~50 | ⚪ |
| sample.py (CFG, Euler, loop) | ~330 | ⚪ |
| text_encoder, vae 래퍼 | ~200 | ⚪ |

### 3-8. MMS 폭주 메커니즘 ↔ 코드 요소 매핑

Q1의 5가지 폭주 원인을 코드의 어떤 부분이 차단하는지.

| 폭주 원인 (→ Q1 참조) | 차단 코드 요소 | 차단 메커니즘 |
|---|---|---|
| ① 잔차에 평균 누적 (1000배) | `init_beta = 0.03` | 처음부터 더하는 양 자체를 작게 |
| ② RMSNorm이 평균을 안 닦음 | `α·(mu_u − mu_x)` 항 분리 | 평균을 수동으로 따로 제어 |
| ③ Softmax 영공간(null space) | `β·(u − mu_u)` 항 분리 | 편차 학습 신호를 따로 보존 |
| ④ 시퀀스 길이 T-배율로 ΔW_μ 폭주 | `init_alpha = 0.0` | 평균 그래디언트 통로를 처음엔 닫음 |
| ⑤ 자기강화 루프 | leaky `(1−α)·J(X) + α·J(F)` | 평균을 EMA처럼 천천히 흐르게 |

→ **단일 모듈(FusedMVSplitNorm1) 하나가 5개 폭주를 모두 차단**.

---

## 4️⃣ 실험 요약

(400층은 메소드 비교용, 1000층은 안정성 시연용 — **출발점이 다르다**.)

| 깊이 | 비교 메소드 | 역할 |
|---|---|---|
| **400층** | Post-Norm baseline / LayerScale / MV-Split | **메소드 비교** (80k step) |
| **1000층** | MV-Split 단독 | **안정성 시연** (50k step까지) |

**데이터**: ImageNet-2012를 FLUX.2 VAE로 잠재공간(latent)으로 인코딩한 것 + Qwen3-0.6B 텍스트 임베딩. 1000층 모델은 약 50k 큐레이션 이미지로 사전학습 후 추가 학습 단계(post-training) 진행.

**핵심 결과**:
- 기준 모델(baseline)은 400층에서 발산(diverge).
- MV-Split은 LayerScale 대비 **20k–30k step 빠르게 수렴**.
- 1000층에서 MV-Split만으로 안정 학습 입증 (완전 수렴이 아닌 가능성 시연, feasibility).

---

# 💬 Q&A — 자주 묻는 질문 정리

## Q1. 왜 1000층에서 평균 쪽으로 와르르 무너지는가?

다섯 가지 메커니즘이 동시에 맞물려 폭주합니다.

### ① 잔차 자체가 1000번 누적
```
X_L = X_0 + Σ F_l   (l = 0..999)
```
(즉 최종 출력은 초기 입력에 모든 층의 업데이트를 다 더한 값.)

각 `F_l`의 평균 편향이 작은 값 `ε`라도 → 1000번 더하면 `1000ε`로 **선형 누적**. 반면 편차(centered)는 √1000 ≈ 31배만 늘어 **평균/편차 비율(ρ_T)이 폭증**.

### ② RMSNorm은 평균을 안 닦아낸다
| | 평균 제거 | 분산 정규화 |
|---|---|---|
| LayerNorm | ✅ | ✅ |
| **RMSNorm** | ❌ | ✅ |

LLaMA, FLUX, Qwen이 모두 RMSNorm을 사용 → 평균이 시스템의 "자유 방향"이 됨 (아무도 청소 안 하는 영역).

### ③ Softmax가 평균 이동을 못 본다
```
softmax(z + c·𝟙) = softmax(z)
```
(즉 입력 전체에 같은 상수 `c`를 더해도 Softmax 출력이 동일.)

Q/K가 평균 방향으로 쏠려도 어텐션 출력 변화 없음 → 역전파(backprop)에서 **"평균을 줄여라"라는 신호가 안 옴**. 일방통행 통로(Softmax null space, 영공간).

### ④ 그래디언트의 시퀀스 길이 T-배율
```
ΔW_μ = T · δ̄ · ȳᵀ
```
(즉 가중치 업데이트의 평균 정렬 성분에는 토큰 개수 `T`가 그대로 곱해진다.)

고해상도 이미지(긴 시퀀스) × 1000층 → 양쪽으로 증폭됨.

### ⑤ 자기강화 루프
평균이 커짐 → 토큰들이 균질화(token homogenization) → 편차 신호 소멸(`ỹ_t → 0`) → 학습 신호가 전부 평균으로 몰림 → 평균이 더 커짐 → ... = **Screaming**.

> **비유**: 평균은 "브레이크 없는 가속 페달". 쌓이고(①), 안 닦이고(②), 줄이는 법을 모르고(③), 시퀀스가 길수록 커지고(④), 한번 시작되면 멈출 수 없다(⑤).

---

## Q2. "옵티마이저가 평균 방향만 키운다"가 실제로 어떻게 망가지는가?

`ΔW_μ = T·δ̄·ȳᵀ`는 외적 형태의 한 방향 업데이트 (rank-1 update). 즉 가중치가 **단 하나의 벡터 방향으로만 학습**됨.

### 결과 1: 가중치가 "획일화 머신"이 됨
어떤 토큰 `x`를 넣어도 `W·x ≈ c·ȳ` (거의 같은 출력). 1024차원짜리 가중치가 1차원처럼 움직임.

### 결과 2: 표현 붕괴 (Representation Collapse)
```
초기:   → ↗ ← ↘ ↑ ↖   (다양함)
폭주 후: → → → → → →   (다 같아짐, 코사인 유사도 → 1)
```
(즉 토큰들의 방향이 처음엔 다양했다가 모두 한 방향으로 쏠림.)

DiT에선 **모든 이미지 패치가 같은 값** → 결과 이미지가 **회색/단색 노이즈**.

### 결과 3: Attention이 무력화
Q, K가 비슷해지면 `softmax(Q·Kᵀ)`는 **모든 위치가 동일한 1/T 분포** → attention이 **그냥 평균 내기**로 퇴화. 게다가 Softmax 영공간(null space) 때문에 **회복 불가능**.

### 결과 4: 편차 그래디언트가 자기 자신을 죽인다
토큰들이 같아지면 `ỹ_t = y_t − ȳ ≈ 0` → `ΔW_c ≈ 0`. **편차 학습 신호 소멸** → 남은 신호는 평균뿐 → 더 폭주.

### 결과 5: 실제 학습 로그
| 시점 | 증상 |
|---|---|
| 정상 | 손실(loss)이 천천히 감소 |
| MMS 직전 | 잔차 크기(norm) 폭증, ρ_T 가파른 상승 |
| **MMS 발생** | 그래디언트 크기 spike → NaN |
| 이후 | 샘플 이미지가 회색 노이즈, attention map이 균일 |

### 결과 6: Adam 옵티마이저의 자원 분배
```
정상:  ΔW_μ 30% + ΔW_c 70%
폭주:  ΔW_μ 99% + ΔW_c  1%
```
한번 모멘텀(momentum)이 평균 방향으로 쌓이면 **돌이킬 수 없음**.

---

## Q3. 1000층짜리 공개 모델이 있나?

세 개만 존재합니다. 모두 **학술적 시연용**.

| 모델 | 연도 | 분야 | 비고 |
|---|---|---|---|
| **ResNet-1001** | 2016 | CNN | He et al., CIFAR. 잔차 학습(residual learning)의 시조 |
| **DeepNet** | 2022 | Transformer (NMT/LM) | Microsoft, DeepNorm. [코드](https://github.com/microsoft/unilm/tree/master/deepnet) |
| **MVSplit-DiT** | 2026 | Diffusion Transformer | 본 논문, [코드+가중치](https://github.com/erwold/mv-split) |

### 상용 모델은 모두 얕다
| 모델 | 깊이 |
|---|---|
| GPT-3 175B | 96 |
| LLaMA-3 70B | 80 |
| DeepSeek V3 | 61 |
| Qwen3 235B | ~96 |
| FLUX.1 | ~57 |
| SD3 | ~24 |

### 왜 상용은 1000층을 안 쓰나
1. 추론 지연(inference latency)이 깊이에 **선형 비례**.
2. 학습 안정성 위험 (MV-Split 같은 기법이 필수가 됨).
3. 같은 연산량(FLOPs)이면 **전문가 혼합(MoE, Mixture-of-Experts) / 너비(width) 확장**이 성능이 더 좋음.
4. 활성값 메모리(activation memory) 부담.

---

## Q4. 본 연구는 실제로 1000층을 쌓아 학습했나?

**둘 다 했지만 출발점이 다르다.** 400층은 비교용, 1000층은 시연용.

| 항목 | 400층 | 1000층 |
|---|---|---|
| 목적 | **메소드 비교** | **안정성 시연** |
| 비교 대상 | 기준 모델 / LayerScale / MV-Split | MV-Split 단독 |
| 학습 step | 80k | 50k까지 |
| 결과 측정 | 정량 (FID 등) | 정성 시연(demonstration) |
| 완전 수렴? | — | ❌ (가능성, feasibility만) |

> 즉 "**1000층이 진짜 죽지 않는다**"만 보여주고, **메소드 우열은 400층에서 가렸다**. DeepNet(2022)과 동일한 패턴 — 200층에서 SOTA 비교, 1000층은 시연.

---

## Q5. FLUX.2 [klein] 4B는 몇 층인가?

[config.json](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/transformer/config.json) 기준:

(아래는 트랜스포머 설정 파일에서 가져온 핵심 하이퍼파라미터.)
```json
{
  "num_layers": 5,         ← 이미지·텍스트 합동 어텐션 블록 (MMDiT)
  "num_single_layers": 20, ← 이미지만 처리하는 단일 스트림 블록 (Single-stream DiT)
  "num_attention_heads": 24,
  "attention_head_dim": 128,
  "mlp_ratio": 3.0
}
```

- **총 깊이 = 5 + 20 = 25층**
- 히든 차원(hidden_size) = 24 × 128 = **3072**
- MLP 차원 = 3072 × 3.0 = **9216**

### MMDiT 구조 (SD3/FLUX 표준)
- **MM-DiT 블록 (5)**: 이미지+텍스트 토큰을 같이 어텐션 (joint attention)
- **단일 스트림 블록 (20)**: 이미지만 처리 (텍스트는 통과)

### 깊이 비교
| 모델 | 깊이 | 파라미터 |
|---|---|---|
| FLUX.1-dev | 57 (19+38) | 12B |
| **FLUX.2 klein 4B** | **25 (5+20)** | 4B |
| MVSplit-DiT | 1000 | 미공개 |

klein 4B는 "**얕고 넓고 빠른**" 디자인 — FLUX.1의 절반 이하 깊이.

---

## Q6. 이런 작은 레이어를 지닌 모델에도 MV-Split이 효과가 있나?

**거의 효과 없습니다.** MV-Split의 효과는 **깊이에 따라 달라진다 (depth-dependent)**.

### 왜 효과가 없나
| 폭주 원인 | 25층에서 |
|---|---|
| 잔차 평균 누적 | 25배만 누적 → 무시 가능 |
| RMSNorm의 평균 보존 | 동일하지만 누적량이 적음 |
| Softmax 영공간(null space) | 동일하나 임계점 미도달 |
| 자기강화 루프 | 25번으론 폭주 시작 못 함 |

> **비유**: 1층 단독주택에 1000층 빌딩급 내진설계는 불필요.

### 깊이별 효과 곡선 (개념)
```
효과 │                            ●●●
     │                       ●●●
     │                  ●●●
     │             ●●●
     │ ━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━
     │ ✕  ✕
     └────────────────────────────────
       25  50 100 200 400  600 1000  층
```
논문이 **400층부터 비교**한 이유 — 그 이하에선 기준 모델(baseline)도 안 터지므로 차이가 안 보임.

### 깊이별 권장 안정화 기법
| 깊이 | 권장 |
|---|---|
| ~30층 (klein 4B 등) | RMSNorm + QK-Norm만으로 충분 |
| 30~100층 | + LayerScale |
| 100~400층 | + DeepNorm / ReZero |
| 400~1000층 | + **MV-Split** |

### 25층에 적용하면?
- **해는 없음**: `α=0`으로 시작해서 처음엔 항등 함수(identity)에 가까움.
- **이득도 없음**: FID 개선 측정 불가, 약간의 연산 오버헤드만 발생.
- "**무해하지만 무효한 영양제**".

---

## 🎯 한 줄 요약 (전체)

> **MV-Split은 1000층 DiT만 걸리는 "평균이 비명을 지르는 병(Mean Mode Screaming, MMS)"의 처방약**: 잔차의 평균과 편차를 분리해 각각 게이트(α, β)로 다뤄 평균의 폭주를 차단한다. **400층에서 메소드 우열을 검증**하고 **1000층은 안정성을 시연**했다. **FLUX.2 klein 4B(25층) 같은 얕은 모델엔 처방할 병이 없으므로 효과 없음**. 상용 영역(30~120층)에서는 RMSNorm + QK-Norm + LayerScale 정도로 충분하고, MV-Split은 깊이 400을 넘는 학술적 영역의 무기다.

---

## 📂 관련 메모리 링크

- 메모리: `paper_mv_split_dit.md` — 논문/코드 상세 분석
- 메모리: `very_deep_models_landscape.md` — 1000+층 공개 모델 풍경
