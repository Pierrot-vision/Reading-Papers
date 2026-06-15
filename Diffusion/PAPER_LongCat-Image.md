# PAPER: LongCat-Image - 6B 이중언어 효율 이미지 생성 모델 + MPO RLHF

## 0. 이 문서를 읽는 법

이 문서는 LongCat-Image 기술 보고서(arXiv 2512.07584)와 공개 코드(meituan-longcat/LongCat-Image)를 처음 읽는 사람이 흐름을 놓치지 않도록 정리한 리뷰입니다.

핵심 목표는 하나입니다.

> **LongCat-Image 는 80B 까지 거대화한 동시대 모델들 한복판에서, 6B 단일 모델로 GenEval 0.87 / 중국어 한자 정확도 90.7 / 편집 SOTA 를 동시에 잡은 "효율 중심" 이미지 생성 기반 모델 (foundation model)이다.**

이 문서는 [PAPER_Z-Image.md](PAPER_Z-Image.md) 와 같은 구성으로 읽기 쉽게 만들었습니다.

1. **메타 정보·용어 사전**: 누가/언제/무엇을, 그리고 알아둘 단어들
2. **큰 그림**: 왜 6B 인가, 왜 중국어인가
3. **모델 구조**: dual+single stream 하이브리드와 코드 검증
4. **데이터 파이프라인**: 진짜 차별점이 이 안에 있음
5. **학습 단계**: pre → mid → SFT → RLHF
6. **RLHF**: DPO + GRPO + MPO (자체 제안 알고리즘)
7. **편집 모델 확장**
8. **실험 결과**
9. **글자 렌더링 9 메커니즘 심층 분석**: 토대 vs 증폭기, FLUX.2 klein 비교, 반사실 ablation
10. **공개 코드 / Q&A / 한계**

GitHub 렌더링 호환을 위해 수식은 LaTeX 보다 평문 표기를 우선합니다.

---

## 1. 메타 정보

| 항목 | 내용 |
|---|---|
| 논문 | LongCat-Image Technical Report |
| 저자 | Meituan LongCat Team (Hanghang Ma 외 12 인) |
| 소속 | 메이투안(Meituan, 중국) |
| 공개일 | 2025-12-08 |
| arXiv abstract | https://arxiv.org/abs/2512.07584 |
| arXiv PDF | https://arxiv.org/pdf/2512.07584 |
| arXiv HTML | https://arxiv.org/html/2512.07584v1 |
| 공식 코드 | https://github.com/meituan-longcat/LongCat-Image |
| 가중치 | HuggingFace: `meituan-longcat/LongCat-Image`, `LongCat-Image-Dev`, `LongCat-Image-Edit`, `LongCat-Image-Edit-Turbo` |
| 라이선스 | Apache 2.0 (상용 가능) |
| 분야 | Text-to-Image, Image Editing, Diffusion Transformer, RLHF |
| 외부 의존 모델 | Qwen2.5-VL-7B (텍스트 인코더), Flux1.dev VAE, InternVL2.5 (캡션 보조) |
| VRAM 요구 | 약 17–18GB (CPU offload 권장) |

---

## 2. 주요 용어 사전 (Glossary)

*다른 절에서 처음 등장하는 용어가 헷갈리지 않게 한곳에 모아둠. 풀어쓴 한국어 + 학술 원어 괄호 매칭.*

### 아키텍처

| 용어 | 풀이 |
|---|---|
| DiT (Diffusion Transformer) | 이미지의 압축 표현(latent)을 Transformer 로 디노이징하는 구조. UNet 대체로 2023 년 이후 표준이 됨 |
| 이중 스트림 (Dual-Stream) | 텍스트 토큰과 이미지 토큰을 **서로 다른 가중치 경로** 로 처리하면서 매 블록 안에서 교차 어텐션으로 상호작용. Flux 가 도입한 디자인 |
| 단일 스트림 (Single-Stream) | 텍스트와 이미지 토큰을 **하나의 시퀀스** 로 이어 붙여 같은 가중치로 처리. dual 보다 단순하고 빠름 |
| MM-DiT (Multimodal DiT) | dual-stream 의 다른 이름. Stable Diffusion 3, Flux 가 채택 |
| MRoPE (Multimodal Rotary Position Embedding) | 텍스트(1D)·이미지(2D) 좌표를 회전 변환으로 위치 정보 부여. 3 축 (모달리티, 높이, 너비) 으로 확장한 형태 |
| AdaLN (Adaptive Layer Normalization) | timestep 같은 조건 벡터로 정규화 파라미터(스케일·시프트) 를 동적으로 만들어 주입하는 기법 |
| VAE (Variational Autoencoder) | 이미지를 작은 잠재(latent) 로 압축하고 다시 복원하는 모델. LongCat 은 Flux1.dev VAE 를 그대로 씀 (8 ×8 공간 압축) |

### 텍스트 인코더 / 캡셔닝

| 용어 | 풀이 |
|---|---|
| Qwen2.5-VL-7B | 알리바바의 70 억 파라미터 비전-언어 모델 (Vision-Language Model). LongCat 은 이 모델을 **텍스트 인코더로 그대로 가져옴**. 출력 차원 3584 |
| 다중 입도 캡셔닝 (Multi-Granularity Captioning, MGC) | 하나의 이미지에 대해 **서로 다른 길이/추상도의 캡션** 을 여러 개 만들어 학습 시 비율로 섞는 기법 |
| Entity / Phrase / Composition / Photographic | MGC 의 4 단계 — 객체만 / 짧은 구 / 전체 구조 / 미세 묘사 |
| 프롬프트 재작성 (Prompt Rewriting) | 사용자 짧은 입력을 모델이 잘 받아들이는 풍부한 프롬프트로 다시 쓰는 단계. LongCat 은 외부 API 없이 Qwen2.5-VL 내부에서 처리 |

### 학습 / 평가

| 용어 | 풀이 |
|---|---|
| Pre-training | 가장 긴 첫 학습 단계. 의미 학습이 목표 |
| Mid-training | Pre-training 직후 미학·사실성 강화 단계. **이 시점 체크포인트(Dev) 를 별도 공개** — 가소성 유지 때문 |
| SFT (Supervised Fine-Tuning) | 사람 검증을 거친 고품질 데이터로 모델을 정렬하는 지도 학습 단계 |
| RLHF (Reinforcement Learning from Human Feedback) | 사람 선호 점수를 보상으로 삼아 강화학습으로 모델을 정렬 |
| DPO (Direct Preference Optimization) | 명시적 보상 모델 없이 선호 쌍(win/lose) 만으로 정렬하는 RL 변형 |
| GRPO (Group Relative Policy Optimization) | 한 프롬프트에 여러 샘플 묶음(G 개) 생성 → 묶음 내 상대 점수로 advantage 계산 |
| MPO (Monolithic Policy Optimization) | **본 논문 제안**. GRPO 의 묶음 동기화 병목을 없앤 단일 궤적 온폴리시 RL |
| Flow Matching | 노이즈에서 이미지로 가는 속도장(velocity field) 를 학습. diffusion 의 한 변형 |
| AIGC (AI-Generated Content) | AI 로 만들어진 콘텐츠. **학습 데이터로 섞이면 모델이 플라스틱·유분진 텍스처로 조기 수렴** 한다는 게 본 논문의 핵심 관찰 |
| CFG (Classifier-Free Guidance) | 조건/무조건 예측을 섞어 조건 충실도를 높이는 추론 트릭. LongCat 은 추가로 CFG renorm 옵션 제공 |
| Logit-Normal / Uniform timestep sampling | 학습 시 timestep 을 뽑는 분포. Logit-Normal 은 중간 t 에 가중(전역 구조), Uniform 은 골고루(고주파 디테일) |

### 벤치마크

| 용어 | 풀이 |
|---|---|
| GenEval | 짧은 프롬프트로 속성 결합·개수·공간 관계 같은 세밀 제어 가능성 평가 |
| DPG-Bench | 1,065 개 밀집 구조 복잡 프롬프트로 명령 따르기 능력 평가 |
| WISE | 1,000 개 프롬프트로 세계 지식 추론 평가 |
| ChineseWord | 중국어 한자 렌더링 정확도. Level 1(흔한 자) ~ Level 3(희귀자) |
| GlyphDraw2 | 그래픽·포스터 등 복잡 텍스트 렌더링 종합 평가 |
| CEdit/GEdit/ImgEdit-Bench | 이미지 편집 능력 벤치마크. 의미 일관성과 지각 품질을 모두 봄 |

---

## 3. TL;DR (한 줄 요약)

> **LongCat-Image 는 "거대화 = 좋은 모델" 이라는 흐름에 반대로, dual+single stream Flux 계열 6B DiT 를 엄격한 AIGC 필터링·다중 입도 캡셔닝·동적 한자 샘플링·자체 RLHF (MPO) 로 학습해, 20B+ 경쟁 모델과 동급이면서 중국어 한자 렌더링은 압도하는 결과를 만든 효율 중심 모델이다.**

조금 더 풀어쓰면:

> **사이즈(6B) 는 작게 묶어두고, 데이터 큐레이션·학습 커리큘럼·RLHF 알고리즘에 디테일을 몰아넣어 "거대 모델 동급 성능 + 중국어 텍스트는 SOTA + 편집 SOTA + Diffusers 공식 통합" 까지 한 번에 잡은 보고서.**

핵심 네 가지:

```text
데이터 큐레이션 (AIGC 오염 차단 + MGC 4단계)
  + Flux 계열 6B 하이브리드 DiT (dual 19 + single 38)
  + 4단계 학습 (pre → mid → SFT → RLHF)
  + 자체 RLHF 알고리즘 MPO
  = 6B 로 20B+ 모델 동급 + 중국어 한자 SOTA
```

---

## 4. 핵심 기여 (Contributions)

1. **6B 단일 모델로 거대 모델 동급 달성** — GenEval 0.87 (Qwen-Image 20B 와 동률), WISE 0.65 (모든 경쟁 모델 능가). "모델을 키워야 성능이 오른다" 라는 통념을 데이터·RLHF 디테일로 반박.
2. **중국어 한자 렌더링 SOTA** — ChineseWord 종합 90.7 (Qwen-Image 56.6, Seedream 4.0 58.5 대비 약 +30 점). 특히 희귀자(Level 3) 70.3 대 6.1/2.3.
3. **AIGC 오염 차단 데이터 파이프라인** — "적은 비율의 AI 합성 데이터도 학습 중 플라스틱·유분진 텍스처로 조기 수렴시킨다" 는 정량 관찰과 그 차단 방법.
4. **다중 입도 캡셔닝 (MGC)** — 4 단계 캡션을 0.05 / 0.1 / 0.2 / 0.65 비율로 섞어 짧은 프롬프트와 디테일 두 목표를 동시에 잡음.
5. **MPO RL 알고리즘 제안** — GRPO 의 묶음 동기화 병목을 없앤 단일 궤적 온폴리시 RL. 칼만 풍 reward 추적 + UCB 풍 커리큘럼.
6. **편집 모델의 가소성 보존 디자인** — Mid-training T2I 체크포인트에서 init (post-train 후 X). 편집 + T2I SFT 공동 학습.
7. **오픈 생태계 완비** — T2I / Dev (중간 체크포인트) / Edit / Edit-Turbo 4 모델 + 전체 학습 코드 + Diffusers 공식 통합 + Apache-2.0.

---

## 5. 큰 그림 — 왜 이 연구를 했나

*"왜 또 다른 이미지 생성 모델인가" 에 답하는 절. 두 가지 진짜 동기를 짚어두면 이후 결정들(6B 사이즈, 한자 집중, 자체 RLHF) 이 자연스럽게 읽힘.*

### 5.1 거대화의 수확체감

최근 이미지 생성 모델은 빠르게 커지고 있습니다.

```text
PixArt-α         0.6B
Stable Diffusion 3 (large)  ~8B
Flux1.dev       12B
Qwen-Image      ~20B
Flux2.dev       32B
HunyuanImage    ~80B
Hunyuan-3.0     80B
```

저자들의 진단: "모델 파라미터의 무분별한 확대가 **예상된 질적 도약을 제공하지 못했으며, 대신 계산 비용 급증과 배포 장벽 증가**를 초래했다."

LongCat 팀의 결론은 **6B 가 성능과 효율성의 최적 균형점** — 이건 같은 2025-말 동시대 모델 [Z-Image (6B)](PAPER_Z-Image.md) 와 동일한 직관이고, [DreamLite (0.39B)](PAPER_DreamLite.md) 와는 다른 트레이드오프 (DreamLite 는 모바일 우선, LongCat 은 서버 우선).

### 5.2 중국어 렌더링의 만성적 약점

기존 오픈/상용 모델이 흔한 한자는 어느 정도 그려도, **희귀자(Level 3) 에선 정확도가 2.3 ~ 6%** 로 무너지는 상황이었습니다. 일상 사용에서는 큰 문제가 없지만 포스터·간판·고전 인용 같은 상업적 응용에서는 치명적.

LongCat 은 이 한 점을 70.3% 까지 끌어올린 것이 가장 큰 단일 기여입니다 (§9 표 참조).

---

## 6. 모델 구조 — 코드 검증 포함

*"6B 안에 무엇이 들었나" 를 코드 라인까지 확인한 절. 본문 주장이 코드와 일치하는지 직접 검증해야 신뢰가 생김.*

### 6.1 전체 흐름

```text
prompt
  → Qwen2.5-VL-7B (텍스트 인코더, 동결)
  → text feature [B, L, 3584]
  → LongCatImageTransformer2DModel (6B DiT)
     - context_embedder: Linear(3584, 3072)
     - x_embedder: Linear(in_channels=64, 3072)
     - time_embed: Timesteps(256) → TimestepEmbedding(256, 3072)
     - pos_embed: FluxPosEmbed(theta=10000, axes_dim=[16,56,56])  # MRoPE 3D
     - dual-stream blocks × 19  (FluxTransformerBlock)
     - single-stream blocks × 38  (FluxSingleTransformerBlock)
     - norm_out: AdaLayerNormContinuous
     - proj_out: Linear(3072, patch_size² × out_channels)
  → latent [B, C, H, W]
  → Flux1.dev VAE decoder (동결)
  → RGB image
```

### 6.2 코드 검증 (`longcat_image/models/longcat_image_dit.py`)

확인한 핵심 클래스 시그니처:

```python
class LongCatImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
```

- `ModelMixin` — diffusers 표준 저장/로드
- `ConfigMixin` — config JSON 직렬화
- `PeftAdapterMixin` — LoRA 부착 지원 → SFT/LoRA 학습에 직접 활용

핵심 하이퍼파라미터:

| 파라미터 | 값 | 역할 |
|---|---|---|
| `num_layers` | **19** | dual-stream 블록 수 |
| `num_single_layers` | **38** | single-stream 블록 수 |
| `num_attention_heads` | 24 | 어텐션 헤드 수 |
| `attention_head_dim` | 128 | 헤드당 차원 |
| `inner_dim` | 3072 (= 24 × 128) | 모델 hidden 차원 |
| `joint_attention_dim` | 3584 | 텍스트 입력 차원 (= Qwen2.5-VL-7B hidden) |
| `in_channels` | 64 | VAE latent 채널 |
| `axes_dims_rope` | [16, 56, 56] | MRoPE 3축 차원 |

### 6.3 디자인 결정 4 가지

#### (1) Dual 19 : Single 38 = 1 : 2 비율

본문은 "구현 효율성" 으로 정당화. 실제로 single 이 dual 보다 가볍지만 (텍스트·이미지 별도 가중치 불필요), **19 + 38 = 57 층이 정확히 Flux1.dev 와 같음** → Flux1.dev 가중치를 출발점으로 워밍업했을 가능성이 매우 높음. VAE 도 Flux1.dev 것 그대로.

⇒ [reference_pretrained_backbone_reuse_landscape](memory/reference_pretrained_backbone_reuse_landscape.md) 의 분기 B "Diffusion 가중치 재사용" 에 해당.

#### (2) Timestep embedding 을 텍스트 인코더에 주입하지 않음

본문: "일반적 timestep embedding 주입 제거 — 실험적으로 성능 이득 미미."

코드 확인:

```python
# 텍스트 처리 부분이 단순한 한 줄 Linear
self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
```

즉 Qwen2.5-VL 출력에 t 를 더하지 않음. Stable Diffusion 3 / Flux 의 표준 디자인과 다른 단순화.

#### (3) MRoPE 3 축 — 모달리티 / 높이 / 너비

```python
self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])
ids = torch.cat((txt_ids, img_ids), dim=0)
image_rotary_emb = self.pos_embed(ids)
```

- 텍스트 토큰은 (modality=0, y=k, x=0) 형태
- 이미지 토큰은 (modality=1, y, x) 형태
- 첫 16 차원은 모달리티/순서 구분, 56+56 = 112 차원은 2D 공간 좌표

복잡한 기하학적 제약 없이 **임의 해상도로 일반화** 됨.

#### (4) AdaLN 은 출력 한 곳만

```python
self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim,
                                       elementwise_affine=False, eps=1e-6)
```

블록 내부 정규화는 Flux 표준을 따르고, 마지막 출력 정규화에만 timestep 조건 AdaLN. timestep 은 1000 배 스케일 (`timestep * 1000`) 적용 — diffusion 표준.

### 6.4 다른 6B 모델들과의 비교

| 구성 | LongCat-Image | [Z-Image](PAPER_Z-Image.md) | [DreamLite](PAPER_DreamLite.md) |
|---|---|---|---|
| 총 파라미터 | **6B** | 6B | 0.39B |
| Stream 디자인 | dual 19 + single 38 (Flux 유전자) | single-only (S3-DiT) | UNet (SDXL-pruned) |
| 텍스트 인코더 | Qwen2.5-VL-7B (3584) | Qwen3-VL-4B | Qwen3-VL-2B |
| RoPE | 3D MRoPE [16,56,56] | rope+ | 2D |
| VAE | Flux1.dev | Flux | SDXL |
| 학습 비용 | 미공개 | 314K H800h | 미공개 |
| 추론 step | 50 (Turbo 8) | 50 (Turbo 8) | 4 (DMD2) |
| 타깃 환경 | 서버 | 서버/엣지 | 모바일 |

세 모델 모두 **Qwen-VL + Flux/SDXL VAE 재사용** 이라는 같은 레시피를 따름.

---

## 7. 데이터 파이프라인 — 진짜 차별점이 여기 있음

*아키텍처는 Flux 계승이고, RLHF 도 기존 알고리즘 조합. 다른 팀이 LongCat 결과를 재현하려면 이 절에 들인 디테일을 따라가야 한다.*

### 7.1 4 단계 필터링

#### (1) 중복 / 저품질 제거
- **MD5 해시** → 정확 중복 제거
- **SigLIP 임베딩 유사도** → 근사 중복 제거
- 해상도 ≥ 384px, 종횡비 0.25 ~ 4.0
- 워터마크 탐지 후 제거
- **LAION-Aesthetics ≥ 4.5** 만 유지

#### (2) AIGC 탐지 후 제거 ⭐
*왜 이게 중요한가: 인터넷에 AI 생성 이미지가 차고 넘치는 2025-26 시점에, 무필터로 크롤링하면 학습 데이터 자체가 오염됨.*

본문 관찰: **"작은 비율의 AI 생성 콘텐츠도 학습 중 조기 수렴으로 플라스틱 또는 유분진 텍스처를 야기"**.

LongCat 의 답: 별도 AIGC 분류기로 탐지 → 제거. (분류기 구체 구조는 비공개)

이 한 줄이 **포토리얼리즘 차별점의 진짜 원천**. 다른 팀이 같은 모델을 만들어도 데이터를 안 거르면 같은 결과 안 나옴.

### 7.2 메타 정보 추출

각 이미지에 다음 라벨을 자동 부착:

| 라벨 종류 | 도구 | 예시 |
|---|---|---|
| 카테고리 | VLM 분류기 | 초상화 / 스포츠 / 식물 / UI / 차트 / 포스터 / 합성 텍스트 등 16종 |
| 스타일 | VLM (phrase) | "oil painting", "anime", "cyberpunk" |
| 명명된 개체 | VLM | "Eiffel Tower", "Taj Mahal" |
| OCR 텍스트 | OCR 모델 | 이미지 안 모든 텍스트 |
| 미학 점수 | 종합 미학 평가기 | 0~10 |

이 라벨들이 학습 시 계층화 샘플링과 SFT 데이터 큐레이션에 쓰임.

### 7.3 다중 입도 캡셔닝 (MGC) ⭐

*왜 4 단계 캡션이 필요한가: 매우 상세한 캡션만 쓰면 짧은 프롬프트(GenEval 같은) 에서 망가지고, 짧은 캡션만 쓰면 사실성·디테일이 무너진다. 비율로 푼다.*

| 입도 | 사용 모델 | 길이 | 샘플링 확률 |
|---|---|---|---|
| Entity (객체만) | Qwen2.5-VL | 1~3 단어 | 0.05 |
| Phrase (짧은 구) | Qwen2.5-VL | 1 문장 | 0.10 |
| Composition (전체 구조) | InternVL2.5 | 2~3 문장 | 0.20 |
| Photographic (미세 묘사) | LoRA-tuned Qwen2.5-VL | 매우 상세 | **0.65** |

65% 가 가장 상세한 photographic 캡션 — 이것이 본문이 강조하는 "포토리얼리즘" 의 기반.

### 7.4 계층화 (Stratification)

학습 단계별로 데이터 비율을 다르게 가져감:

| 단계 | 예술 데이터 비율 | 핵심 차이 |
|---|---|---|
| Pre-training | **0.5%** | 포토리얼리즘 우선, 사진 압도적 |
| Mid-training | 0.5 → **2.5%** + 고해상도(>1024) 강화 | 미학 다양성 확장 |
| SFT | 수동 큐레이션 실제 + 엄격 필터 합성 | 정밀 정렬 |
| RL | 선택적 고품질 | 보상 모델로 평가 가능한 데이터만 |

### 7.5 동적 합성 텍스트 샘플링 ⭐

*왜 따로 다뤄야 하나: 자연 이미지에서 한자가 충분히 나오지 않아 합성 데이터로 보충해야 함. 그런데 무차별 합성은 합성-실사 갭을 만들기 때문에 영리하게 빼야 함. 이게 LongCat 의 글자 우위(특히 희귀자 70.3%)를 만든 진짜 단일 메커니즘.*

#### 작업 흐름

```text
1. SynthDoG 도구로 한자 1000만+ 합성 샘플 생성
   - 약 3,000 흔한 자 + 5,000+ 희귀자
   - 다양한 텍스처 / 색상 팔레트 / 폰트
   - 고전문학 텍스트 활용

2. 학습 중 실시간 문자별 정확도 모니터링
   - OCR 모델로 검증 이미지에서 출력 글자를 읽음
   - 한자별 "현재 모델이 이 글자를 얼마나 잘 그리나" 추적
   - 어려운 문자 → 샘플링 가중치 ↑ (UCB 풍 exploration)
   - 학습된 문자 → 합성 샘플 비율 ↓ (실제 이미지로 점진 전환)

3. 최종 pre-training 단계
   - 합성 텍스트 데이터 비율 → 0
   - 실제 이미지 속 텍스트로만 학습 마무리
   - 합성-실사 갭(synthetic-real domain gap) 메꿈
```

#### "합성 텍스트 제거" 의 정확한 의미 ⚠️

오해 주의: "최종 단계에서 합성 텍스트 제거" 는 **글자 학습 중단을 뜻하지 않음**. 빠지는 건 데이터의 **출처** 일 뿐, 글자 학습은 pre-training 끝까지 계속됨.

```text
합성 텍스트 이미지(SynthDoG 인공 제작)  →  실제 이미지 속 텍스트(자연 사진/포스터)
```

둘 다 "글자가 들어 있는 학습 데이터" 이고, 단지 만들어진 방식이 다를 뿐.

#### 왜 마지막에 합성을 빼나 — 도메인 갭 문제

합성 텍스트 이미지의 한계:
- 글자 **모양(글리프)** 배우기엔 좋음 — 모든 한자가 명확하게 보임
- 그러나 **텍스처는 "가짜"** — 폰트 깔끔, 배경 인위적, 조명·원근·그림자·노이즈 없음

합성만으로 학습을 끝내면 모델이 잘못 배움:

> "글자는 깔끔한 합성 배경 위에 떠 있어야 한다"

→ 실제 사진에 글자를 넣으면 **스티커를 붙인 듯 어색** 해짐. 간판이 벽돌 질감 무시, 종이 글자가 구겨짐·그림자 무시.

마지막에 실제 이미지로 갈아타는 효과:
- 이미 익힌 **"글자 모양 지식"** 유지하면서
- 그 글자를 **"실제 텍스처·조명·원근"** 위에 자연스럽게 올리는 법 학습
- 학습이 끝나는 시점의 데이터가 실사 → 모델의 **최종 출력 분포가 실사 쪽에 정렬**

#### 비유로 정리

> **연습장(합성)에서 글씨체를 익히고 → 실전(실제 사진)에서 자연스럽게 쓰는 연습으로 마무리.**

연습장만 하다 끝내면 글씨는 예쁜데 실제 종이·간판·화면에서 어색. 마지막에 실전으로 옮겨야 진짜 환경에 정착.

학술 용어로:
- **커리큘럼 학습 (curriculum learning)** — 쉬운 것(합성) 먼저, 어려운 것(실사) 마무리
- **도메인 갭 (domain gap)** — 합성 분포와 실사 분포의 차이를 학습 종료 시점에 닫음

#### 격차 증거

희귀자(Level 3) **70.3% 는 이 메커니즘 없이는 불가능**. 다른 팀이 같은 SynthDoG 합성 데이터만 써도 6.1% 수준에 그치는 게 증거 — **데이터만 있고 동적 루프가 없으면 long-tail 분포에 묻혀 그래디언트가 안 흐름**.

---

## 8. 학습 단계

*"한 번에 다 학습" 이 아닌 4 단을 두는 이유: 의미 학습 / 사실성 / 정밀 정렬 / 인간 선호 — 목표가 단계마다 다르고, 한 손실로 동시에 잡으면 충돌함.*

### 8.1 Pre-training (의미 학습)

점진적 해상도 (progressive resolution):

| 서브 단계 | 해상도 | 학습률 | 스텝 | Global BS | 비고 |
|---|---|---|---|---|---|
| Pre-256 | 256 | 1e-4 | 900K | 4608 | 의미 |
| Pre-512 | 512 | 5e-5 | 300K | — | 안정화 (생략하면 1024 발산) |
| Pre-mix | 512–1024 | 2e-5 | 200K | 3072 | 디테일 |

- 옵티마이저: AdamW (β₁=0.9, β₂=0.95)
- 버킷 샘플링 (bucket sampling) 으로 가변 종횡비 처리
- 실시간 평가: 검증 손실 + 이미지-텍스트 정렬 + 미학 + OCR 정확도
- Timestep 샘플링: **Logit-Normal** (중간 t 가중 → 전역 구조 형성)

### 8.2 Mid-training (미학·사실성)

*왜 별도 단계로 두나: pre-training 의 의미 학습 직후 바로 RLHF 로 가면 모드 붕괴(mode collapse) 위험. 사실성을 먼저 안정화 후 정렬해야 함.*

- 계층적 평가: 고급 미학 모델 + 이미지 품질 추정기 + 도메인별 분류기
- 인간 개입 검증 (Human-in-the-loop)
- 수백만 샘플 고충실도 코퍼스

**⭐ Mid-training 체크포인트 = "Dev" 버전 공개**

본문 표현: "높은 가소성과 적응성 유지, 공격적 정렬에 의한 모드 붕괴 방지".

이 체크포인트가 두 가지 가치를 가짐:
- 다운스트림 파인튜닝 init 으로 좋음 (post-train 모델은 너무 굳어서 안 됨)
- 편집 모델의 init 포인트가 됨 (§10.1 참조)

### 8.3 SFT (미학 정밀 정렬)

- 수십만 샘플: 실제 + 고품질 합성 (전문가 검증)
- 검증 차원: 구성, 조명, 색조, 감정 표현
- 결함 합성 데이터 엄격 필터링

**⭐ 모델 가중치 평균화 (Model Soup)**

조명·초상화·예술 스타일 등 **전문화된 모델 여러 개를 따로 미세조정 후 가중 병합** → 다중 속성 균형. (어떤 가중치로 병합했는지는 비공개)

**⭐ Timestep 샘플링 전환**

- Pre-training: Logit-Normal (전역 구조 형성)
- SFT: **Uniform** (모든 t 균등 → 고주파 세부 정제)

이건 [PAPER_Min-SNR.md](PAPER_Min-SNR.md) 의 w(t) 가중치 논의와 같은 결의 통찰.

### 8.4 RLHF (인간 선호 정렬)

별도 절(§9) 에서 상세 설명.

학습 설정 공통:
- 옵티마이저: AdamW
- 학습률: 5×10⁻⁶
- Global batch size: 64
- 샘플러: 12-step SDE, Euler-Maruyama 이산화

---

## 9. RLHF — DPO · GRPO · MPO 세 가지를 다 쓰는 야심작

*"왜 세 가지를 다 쓰나" — 각각의 강점이 다름. DPO 는 안정적 기본기, GRPO 는 탐험, MPO 는 그 둘의 병목을 푸는 자체 제안. 순차 결합.*

### 9.1 Diffusion-DPO (기본기)

명시적 보상 모델 없이 **승리 샘플 / 패배 샘플** 쌍만으로 정렬.

손실 함수 (평문 표기):

```text
L(θ) = - E[ log σ( -β · T · ω(λ_t) · ΔΔ ) ]

ΔΔ = ( ||v^w - v_θ(x_t^w, t)||² - ||v^w - v_ref(x_t^w, t)||² )
   - ( ||v^l - v_θ(x_t^l, t)||² - ||v^l - v_ref(x_t^l, t)||² )

v^w / v^l   : 승리/패배 샘플의 정답 velocity
v_θ         : 현재 모델 예측
v_ref       : 동결된 reference 모델 예측
β           : 정렬 강도 하이퍼파라미터
ω(λ_t)      : timestep 별 가중 (Flow Matching 표준)
σ           : 시그모이드
```

직관: 승리 샘플에서는 ref 보다 더 잘 맞히고, 패배 샘플에서는 ref 보다 덜 맞히도록 유도.

데이터 구축:
- **PromptSet** — 실제 사용자 쿼리 클러스터링 + 데이터 정제
- **ImagePair** — 6 후보 이미지 생성 → 주관적 품질 1~5 점수
  - 점수 3 (중립) 제외
  - 4~5: 긍정, 1~2: 부정
  - 4~5 vs 1~2 페어링

### 9.2 GRPO (그룹 상대 정책)

한 프롬프트에 G 개 샘플 묶음 → 묶음 내 평균/표준편차로 advantage 정규화 → 표준편차가 자연 베이스라인 역할.

```text
A_i = ( R(x_0^i, h) - mean(R) ) / std(R)
```

목적 함수:

```text
L_GRPO(θ) = E[ (1/G) Σ_i (1/T) Σ_t  min( r_t^i · A_i, clip(r_t^i, 1-ε, 1+ε) · A_i ) ]

r_t^i : importance ratio = π_θ / π_old (PPO 의 ratio)
clip  : PPO 의 신뢰 영역 클리핑
ε     : 클립 폭
```

SDE 샘플링으로 탐험 가능:

```text
dx_t = ( v_t + (σ_t² / (2t)) · (x_t + (1-t) · v_t) ) dt + σ_t · dw
```

### 9.3 MPO (Monolithic Policy Optimization) — 자체 제안 ⭐

*왜 새 알고리즘이 필요했나: GRPO 는 묶음 G 개를 다 만들 때까지 기다려야 update 가능 → 묶음 동기화 병목. 단일 궤적만으로 안정적 학습 가능하면 throughput 이 G 배 됨.*

#### 9.3.1 단일 궤적 생성

```text
dz_t = v_θ(z_t, c, t) dt + g(t) dw_t

z_t : noisy latent at timestep t
c   : 조건 (텍스트)
g(t): 노이즈 강도
dw  : 위너 과정
```

#### 9.3.2 안정화 트릭 3 가지

**(a) 칼만 필터 풍 가우시안 값 추적**

*왜 필요한가: 단일 궤적은 분산이 큼. 프롬프트별로 reward 의 평균과 분산을 온라인 추정해서 advantage 계산에 활용.*

```text
K_t   = σ²_{o,t-1} / (σ²_{o,t-1} + σ²_obs)         # 칼만 게인
μ_{o,t} ← μ_{o,t-1} + K_t · (r - μ_{o,t-1})         # 평균 업데이트
σ²_{o,t} ← (1 - K_t) · σ²_{o,t-1} + Q_t              # 분산 업데이트
Q_t   = α · D_KL(π_θ' || π_θ)                       # KL 적응 망각

μ_o, σ_o : 프롬프트별 추적된 평균/분산
Q_t      : KL 발산에 비례한 분산 증가 → 정책이 많이 바뀐
           프롬프트는 더 빨리 잊음 ("KL-adaptive forgetting")
α        : 망각 강도
```

직관: 새 정책이 옛 정책에서 멀어진 만큼 옛 통계를 빨리 잊어 안정성 확보.

**(b) 글로벌 advantage 정규화**

```text
Ã = (A - μ_A) / sqrt(σ²_A + ε)
```

배치 전역 평균·표준편차로 추가 정규화 → scale 안정.

**(c) 불확실성 기반 커리큘럼**

UCB (Upper Confidence Bound) 풍으로 **모르는 프롬프트를 더 자주 보여줌**:

```text
p(c) ∝ σ_o + η / sqrt(n_o + 1)

σ_o : 현재 분산 (불확실성)
n_o : 이 프롬프트가 등장한 횟수
η   : 탐험 강도
```

#### 9.3.3 정책 업데이트

```text
L_MPO(θ) = E[ stop_grad(w_o · Ã) · ||v_θ(z_t, c, t) - u(z_t, z_0)||² ]

w_o = 1 + γ · |r - μ_o| / (σ_o + ε)
```

특이점: **RL 을 supervised flow matching MSE 처럼 표현**. stop-grad 된 advantage 가 가중치로만 작용. 부호와 크기로 학습 방향 조절.

이 표현 방식은 [PAPER_UniRef-Image-Edit.md](PAPER_UniRef-Image-Edit.md) 의 MSGRPO 와 같은 패러다임 (flow matching 위 closed-form 가중치).

#### 9.3.4 MPO 하이퍼파라미터

| 기호 | 값 | 의미 |
|---|---|---|
| λ | 0.99 | EMA 감쇠 |
| η | 1.0 | UCB 탐험 강도 |
| α | 1.0 | KL 망각 강도 |
| γ | 0.5 | 가중치 강도 |

---

## 10. 이미지 편집 (LongCat-Image-Edit)

*왜 별도 모델이 필요한가: T2I 후처리로 편집을 강제하면 (1) catastrophic forgetting 으로 T2I 능력 손상, (2) 레퍼런스 그라운딩이 잘 안 됨. 별도 fine-tune.*

### 10.1 가장 중요한 결정: Init 선택

**Mid-training T2I 체크포인트에서 init** (post-train 후가 X).

이유: post-train 모델은 "이미 너무 굳었다" — 가소성이 죽어 편집 같은 새 task 로 갈아끼우면 catastrophic forgetting. Mid-training 시점이 **가소성과 미학이 균형 잡힌 지점**.

이건 같은 LongCat 팀의 [UniCustom](PAPER_UniCustom.md) 디자인 철학과 일관. ([paper_unicustom](memory/paper_unicustom.md) 참고)

### 10.2 아키텍처 수정

```text
참조 이미지 I_ref
  → Flux VAE → latent
  → 3D RoPE ID 부여 (modality=2, y, x)  # 노이즈 잠재(modality=1)와 구분
  → 노이즈 잠재와 시퀀스 차원으로 concat
  → DiT 입력

Qwen2.5-VL 입력
  → (참조 이미지 + 명령) 함께 투입
  → 편집 전용 system prompt 사용
```

핵심: **레퍼런스 그라운딩을 시퀀스 concat + RoPE ID 구분** 으로 처리. 추가 모듈 거의 없음. ([PAPER_Any2AnyTryon.md](PAPER_Any2AnyTryon.md) 의 RoPE 재해석과 동일 계열)

### 10.3 데이터 5 종

| 소스 | 설명 |
|---|---|
| 오픈 데이터셋 | OmniEdit, OmniGen2, NHREdit |
| 합성 데이터 | 객체 조작, 스타일 전이, 배경 변경, 참조 기반 생성 |
| 비디오 프레임 | 광학 흐름 (optical flow) 으로 자동 주석 |
| 인터리빙 코퍼스 | 웹 규모 이미지-텍스트 시퀀스 마이닝 |
| 명령 재작성 | GPT-4o 로 다양성 향상, 다중 변형 생성 |

### 10.4 학습 단계

#### 10.4.1 Pre-training
- 512×512 → 1024×1024 다중 규모
- **편집 + T2I mid-training 데이터 균형 혼합** ← T2I 능력 잃지 않게
- 3 ~ 5 개 후보 프롬프트 사용

#### 10.4.2 SFT
- 수십만 샘플 고충실도 코퍼스
- 구조적 정렬에 대한 엄격한 인간 필터
- 고품질 T2I SFT 데이터와 **공동 학습**

#### 10.4.3 DPO (편집용 변형)

T2I DPO 와 같은 구조이되 입력에 참조 이미지 I_src 가 추가:

```text
L(θ) = - E[ log σ( -β · T · ω(λ_t) · ΔΔ ) ]

ΔΔ = ( ||v^w - v_θ(x_t^w, I_src^w, P^w, t)||² - ||v^w - v_ref(x_t^w, I_src^w, P^w, t)||² )
   - ( ||v^l - v_θ(x_t^l, I_src^l, P^l, t)||² - ||v^l - v_ref(x_t^l, I_src^l, P^l, t)||² )

I_src^w / I_src^l : 승리/패배 케이스의 참조 이미지
P^w / P^l          : 승리/패배 케이스의 편집 명령
```

### 10.5 Edit-Turbo 변형
- guidance 1.0 / **8 step** 추론 (표준 4.5 / 50 step)
- **10× 가속**
- 별도 distilled 체크포인트로 공개

---

## 11. 실험 결과

*"진짜 강한 곳이 어디인지" 를 솔직하게 보여주는 절. 전반적으로는 SOTA "동급", 한 점(중국어) 만 SOTA "압도".*

### 11.1 T2I 정렬 벤치마크

| 벤치 | LongCat-Image (6B) | Qwen-Image (20B) | Seedream 4.0 | HunyuanImage-3.0 (80B) |
|---|---:|---:|---:|---:|
| GenEval | **0.87** | 0.87 | — | 0.72 |
| DPG-Bench | 86.80 | **88.32** | — | 86.10 |
| WISE | **0.65** | 0.62 | — | 0.57 |

읽는 법:
- GenEval: 동률
- DPG-Bench: 살짝 짐 (~ -1.5)
- WISE: 모든 경쟁 모델 능가

### 11.2 텍스트 렌더링 벤치마크 ⭐

| 벤치 | LongCat-Image | Qwen-Image | Seedream 4.0 |
|---|---:|---:|---:|
| GlyphDraw2 Complex-zh | **0.92** | 0.87 | 0.91 |
| GlyphDraw2 Avg | 0.95 | 0.93 | **0.97** |
| CVTG-2K (영어) Word Accuracy | 0.8658 | — | **0.8917** |

**ChineseWord (종합 중국 한자)** — 핵심 차별점:

| 모델 | Level 1 (흔한 자) | Level 2 | **Level 3 (희귀자)** | 종합 |
|---|---:|---:|---:|---:|
| **LongCat-Image** | **98.7** | **90.8** | **70.3** | **90.7** |
| Seedream 4.0 | 94.8 | 41.2 | 2.3 | 58.5 |
| Qwen-Image | 92.5 | 37.1 | 6.1 | 56.6 |

**Level 3 의 70.3 대 2.3 / 6.1** 가 본 논문의 가장 충격적인 숫자. 약 +30 점 종합.

### 11.3 인간 평가

- 400 개 프롬프트
- 4 차원: 텍스트-이미지 정렬 / 시각적 그럴듯함 / 사실성 / 미학
- 결과: HunyuanImage-3.0 능가, Qwen-Image 동등, **시각적 사실성에서 Seedream 4.0 미세 우위**

### 11.4 이미지 편집

| 벤치 | LongCat-Image-Edit | Qwen-Image-Edit | Seedream 4.0 |
|---|---:|---:|---:|
| **CEdit-Bench** (1,464 쌍 15 카테고리) | **7.67** | 7.52 | 7.58 |
| CEdit 의미 일관성 | **8.27** | 8.07 | 8.12 |
| CEdit 지각 품질 | 7.88 | 7.84 | **7.95** |
| **GEdit-Bench** | 7.64 | 7.56 | **7.68** |
| **ImgEdit-Bench** (9 편집 유형) | **4.50** | 4.27 | 4.18 |

→ **편집은 오픈 SOTA** 라 부를 만함 (CEdit, ImgEdit 1 위; GEdit 만 미세 차).

### 11.5 T2I-CoreBench (종합)

본문 진술: "LongCat-Image 는 모든 오픈소스 모델 중 **종합 2 위**, 32B Flux2.dev 에만 미세하게 짐."

---

## 12. 글자 렌더링이 왜 그렇게 강한가 — 9 메커니즘 심층 분석

*왜 이 절을 두는가: ChineseWord 종합 90.7 / 희귀자(Level 3) 70.3 이라는 +30 ~ +68 점 격차는 단일 트릭이 아니라 9 개 메커니즘의 곱셈으로 만들어진다. 어떤 게 진짜 1등인지, 다른 팀이 흉내 내려면 무엇이 필요한지 따로 정리.*

### 12.1 9 가지 메커니즘 한눈에

| # | 메커니즘 | 어디서 다뤘나 | 다른 팀도 하나? |
|---|---|---|---|
| 1 | Qwen2.5-VL-7B 텍스트 인코더 (한자 토큰화 강함) | §6, §12.2 | ◯ 거의 모든 최신 모델 |
| 2 | 따옴표 트리거 + character-level encoding | §12.2 | ◯ Qwen-Image, LongCat 등 |
| 3 | SynthDoG 1000만+ 한자 합성 데이터 (5000+ 희귀자 포함) | §7.5 | △ 유사 합성은 함 |
| 4 | OCR 정확도 기반 **동적 합성 샘플링** ⭐ | §7.5 | ✗ LongCat 특유 |
| 5 | Pre-training 끝에 합성 텍스트 → 실사 전환 (도메인 갭 메꿈) | §7.5 | ✗ 잘 안 함 |
| 6 | MGC photographic 캡션(65%)에 OCR 텍스트 명시 포함 | §7.3 | △ 일부 함 |
| 7 | AIGC 오염 차단 (흐릿한 글자 누적 방지) | §7.1 | △ 일부 함 |
| 8 | Mid-training 고해상도(>1024) 강화 (작은 글자 capacity) | §8.2 | △ 일부 함 |
| 9 | RLHF 4 차원 평가에 텍스트 정렬 포함 (DPO/MPO 텍스트 정렬) | §9 | △ 일부 함 |

### 12.2 텍스트 인코더 — 다른 최신 모델과의 비교

*"강한 LLM/VLM 인코더" 가 글자 능력의 토대지만, 이건 이미 최신 모델 공통 전제. LongCat 만의 차별점이 아님.*

| 모델 | 텍스트 인코더 | 종류 | 세대 |
|---|---|---|---|
| **LongCat-Image** | Qwen2.5-**VL**-7B | **비전-언어 (이미지+텍스트)** | Qwen2.5 |
| **FLUX.2 dev** | Mistral Small 3 (24B) | 비전-언어 (VLM) | Mistral-3 |
| **FLUX.2 klein** (4B) | Qwen**3** 4B | **텍스트 전용** | Qwen3 (신세대) |
| **FLUX.2 klein** (9B) | Qwen**3** 8B | 텍스트 전용 | Qwen3 |
| [Z-Image](PAPER_Z-Image.md) | Qwen3-4B | 텍스트 전용 | Qwen3 |
| FLUX.1 dev (구세대) | T5-XXL + CLIP | 텍스트 전용 | T5/CLIP |

핵심 관찰:
- **klein·Z-Image 도 강한 Qwen 인코더를 씀** — 그런데도 ChineseWord 70 을 못 따라옴
- 따라서 "강한 인코더 = 강한 글자" 인과는 성립 X. **인코더는 필요조건이지 충분조건이 아님**
- 격차는 §12.1 의 #3~#5(데이터·동적 샘플링·도메인 전환)에서 나옴

#### VL 인코더 vs 텍스트 전용 — 어느 게 더 좋은가?

흔한 직관 ("VL이 더 풍부하니 더 좋다") 의 함정:

| 축 | VL (LongCat, Qwen2.5-VL-7B) | 텍스트 전용 (klein, Qwen3) |
|---|---|---|
| 이미지 입력 받기 | **가능** (편집·레퍼런스에 유리) | 불가 |
| 한자 토큰화·이해 | 강함 | 강함 (차이 거의 없음) |
| 같은 파라미터당 텍스트 깊이 | 비전 타워에 capacity 분산 | **텍스트에 전부 투입** |
| 세대 | Qwen2.5 (구세대) | **Qwen3 (신세대)** |
| 무게 / 추론 비용 | 무거움 (T2I 때 비전 타워는 죽은 무게) | **가벼움** |

→ VL 이 절대적으로 우월하지 않음. LongCat 이 VL 을 고른 건 **글자 때문이 아니라 편집 모델까지 한 인코더로 통합** 하려는 설계 판단. T2I 단일 성능만 보면 텍스트 전용도 충분하고 더 가벼움.

#### 따옴표 + character-level encoding 의 무게

추론 코드 안내: "렌더링할 텍스트는 반드시 단일/이중 따옴표로 감싸야 함" — 이게 character-level 인코딩 트리거.

추정 동작:
- 따옴표 밖: 단어 단위 BPE (의미 인코딩)
- 따옴표 안: **한 글자씩 분리** → 각 한자의 글리프(자형) 정보가 독립 임베딩으로 유지

영향 강도: 매우 큼. 일반 BPE 로 한자를 묶거나 부분바이트로 자르면 글리프 정보가 손실 → 한 자씩 정확히 그릴 수 없음. **이게 없으면 ChineseWord 종합이 20~30% 대로 추락할 것으로 추정**.

### 12.3 토대(Enabler) vs 증폭기(Amplifier) — 두 관점의 진짜 1등

*"가장 효과적인 메커니즘" 의 답은 "성능을 좋게 한다" 를 어떻게 해석하느냐에 따라 갈림.*

| 관점 | 의미 | 1등 |
|---|---|---|
| **A. 절대 수준** (90.7 점을 가능하게 한 것) | "이게 없으면 점수가 0 에서 시작" | **Qwen 인코더 + 따옴표 char-level (토대)** |
| **B. 격차** (다른 모델 대비 +30~68 점) | "있는 사람끼리 누가 더 올라가나" | **OCR 기반 동적 샘플링 (증폭기)** |

두 답 모두 옳음. 표현하자면:

```text
토대 (Enabler) 없으면 → 점수 자체가 0
증폭기 (Amplifier) 없으면 → 다른 모델 수준(6~50)에 머묾
LongCat 의 90.7 = 토대 × 증폭기 (곱셈)
```

### 12.4 반사실(counterfactual) ablation 추정

*논문에 항목별 ablation 표가 없어 메커니즘 논리로 추정. 각 메커니즘을 하나씩 제거했을 때의 가상 결과.*

| 빼는 것 | 추정 ChineseWord 종합 | 추정 Level 3 | 왜? |
|---|---|---|---|
| **(아무것도 안 뺌) LongCat 원본** | 90.7 | 70.3 | 기준선 |
| Qwen 인코더 → T5/CLIP | **0 ~ 5** | 0 | T5/CLIP 은 한자 토큰화 자체가 안 됨 |
| 따옴표 char-level → 일반 BPE | **20 ~ 30** | 0 | 글리프 정보 손실, 한 자씩 정확히 그릴 수 없음 |
| SynthDoG 합성 데이터 빼기 | Level 1 ≈ 80, Level 3 ≈ 0 | 0 | 자연 이미지엔 희귀자 거의 없음 |
| **동적 OCR 샘플링만 빼기** | **약 50 ~ 60** | **5 ~ 10** | 다른 팀이 SynthDoG 만 쓸 때의 실제 결과 |
| 마지막에 합성 끊기 빼기 | 종합 약간 ↓ | 글자 텍스처 어색 | "스티커 효과" 발생 |
| AIGC 차단 빼기 | -5 ~ -10 | -5 ~ -10 | 글자 흐릿함 천장 |

(주의: 정확한 ablation 이 아니라 메커니즘 논리상 추정. 논문엔 이 표가 없음)

### 12.5 정렬된 최종 순위

**관점 A — 절대 수준 (점수 자체를 만든 것):**

1. ⭐ **Qwen2.5-VL 인코더 + 따옴표 character-level encoding** — 한자를 다룰 수 있는 토대 자체. 빠지면 0
2. SynthDoG 희귀자 합성 데이터 — 학습 재료 공급. 빠지면 Level 3 = 0
3. 동적 OCR 샘플링 — 재료 중 어려운 것 골라 반복
4. 마지막 합성 끊기 / AIGC 차단 — 마무리·정착

**관점 B — 다른 모델 대비 격차 (+30~68 점):**

1. ⭐ **동적 OCR 샘플링** — 같은 토대 위에서 LongCat 만의 트릭. Level 3 격차의 결정적 원천
2. 마지막에 합성 끊기 — domain gap 메꿈
3. AIGC 차단 — 천장 제거
4. (인코더·char-level 은 모두 가져서 격차 기여 X)

**한 줄 결론**: 토대(Qwen + char-level) 와 증폭기(동적 OCR 샘플링) **두 축이 같이 1등**. 토대 없으면 점수가 0 에서 시작하고, 증폭기 없으면 다른 모델 수준(6~50)에 머묾. **둘이 곱셈으로 작용해 90.7 을 만듦**.

### 12.6 다른 팀이 흉내 내려면 — 재현 체크리스트

LongCat 의 글자 성능을 따라잡으려면 다음을 **전부** 갖춰야 함 (하나만 빠져도 큰 폭 하락):

```text
□ 강한 중국어 토큰화 인코더 (Qwen3 / Qwen2.5-VL 급)
□ 따옴표 트리거 + character-level encoding 파이프라인
□ 자체 합성 한자 데이터 (희귀자 5000+ 포함, SynthDoG 류)
□ OCR 모델 (검증용) + 학습 루프 안 실시간 평가 통합
□ 문자별 정확도 추적 → 동적 샘플링 가중치 갱신 로직
□ Pre-training 후반 합성 비율 감소 스케줄
□ Pre-training 마지막에 합성 → 실사 전환
□ AIGC 분류기로 학습 데이터 오염 차단
□ 캡션에 OCR 텍스트를 명시적으로 포함 (MGC photographic 풍)
□ 고해상도(1024+) mid-training 단계
□ RLHF 평가에 텍스트 정렬 차원 포함
```

→ 다른 팀이 한두 개 흉내 내서는 따라잡기 어려운 이유: **시스템 효과** (각 메커니즘이 다른 메커니즘의 신호를 받아 작동). 예: 동적 샘플링은 OCR 평가 신호를 받음, RLHF 는 사전학습된 글자 능력을 정렬.

---

## 13. 공개 코드 레포

*"내 환경에서 돌릴 수 있나" 에 답하는 절. 실용 정보 위주.*

### 12.1 공개 모델 4 종

| HuggingFace ID | 용도 |
|---|---|
| `meituan-longcat/LongCat-Image` | 표준 T2I |
| `meituan-longcat/LongCat-Image-Dev` | Mid-training 체크포인트 (파인튜닝 init) |
| `meituan-longcat/LongCat-Image-Edit` | 이미지 편집 |
| `meituan-longcat/LongCat-Image-Edit-Turbo` | 편집 가속 (10×) |

### 12.2 의존성 (`infer_requirements.txt`)

```text
accelerate==1.11.0
safetensors==0.6.2
torch==2.6.0
torchvision==0.21.0
transformers==4.57.1
openai==2.8.1
```

추가로 diffusers (최신) — **공식 통합 됨** (가장 큰 실용 가치).

### 12.3 추론 예제 — T2I

```python
import torch
from diffusers import LongCatImagePipeline

pipe = LongCatImagePipeline.from_pretrained(
    "meituan-longcat/LongCat-Image",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()  # VRAM 17-18 GB

image = pipe(
    prompt='Make a poster with the text "您好世界"',
    height=768,
    width=1344,
    guidance_scale=4.0,
    num_inference_steps=50,
    enable_cfg_renorm=True,        # CFG 정규화
    enable_prompt_rewrite=True,    # 내부 Qwen2.5-VL 로 prompt 재작성
).images[0]
```

**⭐ 텍스트는 반드시 단일/이중 따옴표로 감싸야 함** — 이게 "character-level encoding" 트리거.

### 12.4 추론 예제 — Edit

```python
from diffusers import LongCatImageEditPipeline

pipe = LongCatImageEditPipeline.from_pretrained(
    "meituan-longcat/LongCat-Image-Edit",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

# 표준
image = pipe(image=ref, prompt="将猫变成狗",
             guidance_scale=4.5, num_inference_steps=50).images[0]

# Turbo
image = pipe(image=ref, prompt="将猫变成狗",
             guidance_scale=1.0, num_inference_steps=8).images[0]
```

### 12.5 학습 코드 구조 (`train_examples/`)

| 디렉토리 | 용도 | 데이터 포맷 |
|---|---|---|
| `sft/` | T2I 지도학습 | JSONL: `img_path`, `prompt`, `width`, `height` |
| `lora/` | T2I LoRA | JSONL (동일) |
| `dpo/` | T2I DPO | TXT: 동일 prompt 에 win/lose 이미지 쌍 |
| `edit_sft/` | 편집 SFT | TXT: 편집된 이미지 + 참조 + 명령 |
| `edit_lora/` | 편집 LoRA | TXT |
| `edit_dpo/` | 편집 DPO | TXT: win/lose 편집 결과 + 참조 |

공통 옵션:
- `aspect_ratio_type`: `mar_256` / `mar_512` / `mar_1024` (해상도 버킷)
- `diffusion_pretrain_weight`: 외부 가중치 init 지원
- `resume_from_checkpoint`: 최신 / 특정 체크포인트 복귀

### 12.6 라이선스
**Apache 2.0** — 상용 가능. (단 Flux 가중치 워밍업 가능성을 고려하면 Flux 라이선스와의 관계는 사용자가 별도 확인 필요)

---

## 14. 💬 Q&A

### Q1. 왜 dual-stream 19 + single-stream 38 비율인가? Flux 와 같은데 우연인가?

같지 않을 확률이 매우 낮음. Flux1.dev 가 정확히 19+38 = 57 층이고, LongCat 도 같음. VAE 도 Flux1.dev 그대로. 즉 **Flux1.dev 가중치를 워밍업 init 으로 사용** 했을 가능성이 매우 높음.

본문은 이를 "구현 효율성" 으로 정당화하지만 — 코드와 차원이 정확히 일치하는 정황 증거가 강함. 이건 [reference_pretrained_backbone_reuse_landscape](memory/reference_pretrained_backbone_reuse_landscape.md) 의 **분기 B (Diffusion 가중치 재사용)** 에 해당.

라이선스 영향: Apache-2.0 으로 풀었지만, Flux1.dev 의 비상업 라이선스와의 관계 정리가 본문에 없음. 상업 사용 전 별도 확인 필요.

### Q2. 6B 모델이 20B 모델과 동급인 진짜 이유는?

데이터 + RLHF 입니다. 아키텍처는 평이.

- **데이터**: AIGC 오염 제거 + MGC 4 단계 비율(0.05/0.1/0.2/0.65) + 동적 한자 샘플링 — 이 셋이 같이 작용.
- **RLHF**: DPO 안정성 + GRPO 탐험 + MPO throughput 의 3 단 결합.
- **아키텍처**: Flux 디자인 거의 그대로. inner dim 3072 = 24 × 128, 19+38 층.

핵심 메시지: **"파라미터 수가 모델 능력의 1차 결정자가 아니다"** — 데이터 큐레이션과 정렬 알고리즘이 더 중요할 수 있다는 가설을 강화하는 두 번째 데이터 포인트 ([Z-Image](PAPER_Z-Image.md) 가 첫 번째).

### Q3. 중국어 한자 70.3% 가 진짜 의미하는 것은?

**합성 데이터로 만들어낸 결과** 라는 점이 중요. 본문이 솔직히 밝힘:
- SynthDoG 로 1000만+ 한자 합성 샘플
- 약 3,000 흔한 자 + 5,000+ 희귀자
- 학습 중 정확도에 따라 동적 샘플링

따라서:
- **다양한 폰트·텍스처에서 재현될 가능성 ↑** (합성 시 다양화함)
- **분포 밖 한자 (즉 합성 안 된 한자) 에서는 미지수**
- **실제 손글씨 같은 자연 변형에서는 검증 안 됨**

다른 팀이 SynthDoG 만 갖다 쓰면 6% 수준에 그치는 게 그 증거 — **합성 + 동적 샘플링 + 최종 단계 합성 제외** 라는 전체 레시피가 필요.

### Q4. MPO 가 GRPO 대비 정말 좋은가?

본문에서 MPO 단독 ablation 이 약함. DPO 단독 vs DPO+GRPO vs DPO+MPO 표가 부족.

이론적 장점:
- **묶음 동기화 불필요** → throughput G 배
- **KL 적응 망각** → 정책 변화 빠른 프롬프트 안정화
- **UCB 커리큘럼** → 탐험 다양성

그런데 "본 모델 성능에 MPO 가 얼마나 기여했나" 는 정량으로 모름. 향후 후속 작업에서 검증 가능한 영역.

### Q5. 편집 모델이 Mid-training 에서 init 한다는 결정의 의미는?

**Post-training 후 모델은 너무 굳어서 편집 같은 새 task 로 갈아끼우면 catastrophic forgetting 일어남.**

직관: post-train (SFT + RLHF) 이 모델을 특정 분포에 강하게 정렬 → 새 task 의 그래디언트가 그 정렬을 깨뜨림 → 편집과 T2I 둘 다 망가짐.

해결: 미학·사실성은 이미 잡혀 있되 아직 가소성이 남은 **mid-training 체크포인트** 에서 init → 편집 학습 + T2I SFT 데이터 공동 학습으로 둘 다 보존.

이게 일반적으로 새 task fine-tune 할 때 적용 가능한 원칙. [PAPER_UniCustom.md](PAPER_UniCustom.md) 도 같은 LongCat 팀에서 같은 결정.

### Q6. "텍스트는 따옴표로 감싸야 함" 은 어떻게 동작하나?

코드 상으로 정확한 메커니즘은 공개된 추론 스크립트만으로는 보이지 않음 (pipeline 내부 처리). 추정:

- 프롬프트 파싱 단에서 따옴표 안 문자열을 별도 토큰 시퀀스로 분리
- 한자 문자 단위 (character-level) 로 잘라서 Qwen2.5-VL 의 vocabulary 로 인코딩
- 따옴표 밖은 단어 단위 정상 인코딩

이렇게 하면 **렌더링 대상 텍스트와 묘사 텍스트의 의미가 섞이지 않음** — Qwen-Image 가 이미 도입한 트릭과 유사. LongCat 의 추가 기여는 캐릭터 단위 인코딩과 동적 합성 샘플링의 결합으로 추정.

### Q7. 다른 6B 모델 (Z-Image) 와 비교했을 때 선택 기준은?

| 우선순위 | 추천 |
|---|---|
| 중국어 한자 렌더링 | **LongCat-Image** (압승) |
| 영어 prompt 정렬 | Qwen-Image / LongCat 동률 |
| 추론 속도 | **Z-Image-Turbo** (DMD 증류 더 적극) |
| 편집 | **LongCat-Image-Edit** (오픈 SOTA) |
| 모바일 | [DreamLite](PAPER_DreamLite.md) (0.39B) |
| Diffusers 즉시 통합 | **LongCat-Image** (공식 PR 머지) |

### Q8. 한계는 무엇인가?

§15 참조.

### Q9. FLUX.2 klein 도 Qwen 을 쓰는데 LongCat 만 글자가 강한 이유는?

klein 은 Qwen2.5-VL 이 아니라 **Qwen3 (텍스트 전용)** 을 씀. 그래도 강한 중국어 인코더라는 점에서 토대는 LongCat 과 거의 같음.

그런데도 ChineseWord 격차가 큰 이유: **인코더는 토대일 뿐, 격차는 데이터·커리큘럼에서 나옴** (§12.3 참조).

- klein·Z-Image: 강한 Qwen 인코더 ◯, 동적 OCR 샘플링 ✗ → Level 3 = 약 5~10
- LongCat: 같은 토대 ◯, 동적 OCR 샘플링 ◯ → Level 3 = 70.3

**같은 토대 위에서 +60~65 점 격차를 만드는 게 동적 샘플링**이라는 게 klein 비교의 핵심 교훈. §12.2 참조.

### Q10. VL 인코더가 텍스트 전용보다 더 좋은가? LongCat 이 VL 을 고른 이유는?

VL 이 절대적으로 우월하지 않음. 트레이드오프:

| 축 | VL (Qwen2.5-VL-7B) | 텍스트 전용 (Qwen3) |
|---|---|---|
| 이미지 입력 | 가능 (편집·레퍼런스에 유리) | 불가 |
| 한자 토큰화 | 강함 | 강함 (차이 없음) |
| 같은 파라미터당 텍스트 깊이 | 비전 타워에 분산 | 텍스트에 집중 |
| 세대 | Qwen2.5 (구) | **Qwen3 (신)** |
| 무게 | 무거움 | 가벼움 |

LongCat 이 VL 을 고른 진짜 이유: **편집 모델까지 한 인코더로 통합** 하려는 설계 판단. T2I-Edit 가 같은 Qwen2.5-VL 에 (이미지 + 명령) 함께 투입 — 텍스트 전용이면 이미지를 받는 별도 경로를 따로 만들어야 함. 글자 능력 때문이 아님.

### Q11. "pre-training 마지막에 합성 텍스트 제거" 는 글자 학습을 중단한다는 뜻인가?

**아니. 정반대.** 글자 학습은 pre-training 내내 끝까지 계속됨. 빠지는 건 데이터의 **출처**:

```text
합성 텍스트 이미지(SynthDoG 인공)  →  실제 이미지 속 텍스트(자연 사진)
```

둘 다 "글자가 들어 있는 학습 데이터" 이고, 단지 만들어진 방식이 다를 뿐.

왜 마지막에 합성을 빼나: 합성 이미지는 글자 모양 배우기엔 좋지만 텍스처가 **인위적** 임 — 폰트 깔끔, 배경 가짜, 조명·원근·노이즈 없음. 합성만으로 끝내면 모델이 "글자는 깔끔한 합성 배경 위에 떠 있어야 한다" 고 잘못 배움 → 실제 사진에 글자 넣으면 **스티커 효과** 발생.

마지막에 실사로 갈아타면 이미 익힌 글자 모양을 실제 텍스처 위에 자연스럽게 올리는 법 학습. 비유: "연습장(합성)에서 글씨체 익히고 → 실전(실제 사진)에서 자연스럽게 쓰는 마무리".

학술 용어: **커리큘럼 학습 (쉬운 합성 → 어려운 실사)** + **도메인 갭 (synthetic-real gap) 메꿈**. §7.5 참조.

### Q12. LongCat 의 글자 강화 방법 중 가장 효과적인 단일 메커니즘은?

답이 두 개 — "성능을 좋게 한다" 를 어떻게 해석하느냐에 따라.

| 관점 | 1등 |
|---|---|
| 절대 점수(90.7)를 가능하게 한 것 | **Qwen 인코더 + 따옴표 char-level encoding** (토대) |
| 다른 모델 대비 격차(+30~68 점)를 만든 것 | **OCR 기반 동적 합성 샘플링** (증폭기) |

둘 다 1 등이고, **곱셈으로 작용** 함:
- 토대 없으면 점수가 0 에서 시작 (T5/CLIP 으로는 한자 불가)
- 증폭기 없으면 다른 모델 수준(6~50)에 머묾 (klein·Z-Image 가 증거)
- LongCat 의 90.7 = 토대 × 증폭기

자세한 반사실 ablation 추정은 §12.4, 정렬된 순위는 §12.5 참조.

---

## 15. 한계 (Limitations)

*"이 모델로 못 하는 것 / 못 검증된 것" 을 정직하게.*

1. **신규성 농도가 낮음.** 아키텍처는 Flux1.dev 그대로, RLHF 도 DPO+GRPO 위에 MPO 한 겹. 엔지니어링 보고서로 읽어야 마땅.
2. **학습 비용 미공개.** [Z-Image (314K H800h)](PAPER_Z-Image.md) 와 대조적. 재현성 평가가 어려움.
3. **DPG-Bench 가 Qwen-Image 보다 낮음** (86.80 vs 88.32) — 명령 따르기에서 약간 손해. Photographic 캡션 65% 의 부작용일 수 있음.
4. **희귀자 70.3% 의 일반성 미지수.** 합성 데이터로 끌어올렸기 때문에 분포 밖 한자나 자연 폰트에서 재현될지 불확실.
5. **MPO 단독 ablation 약함.** DPO 단독 vs DPO+GRPO vs DPO+MPO 정량 비교 부족.
6. **Flux 라이선스 관계 불명확.** Apache-2.0 으로 풀었으나, layer 수/VAE 가 Flux1.dev 와 일치하는 정황 → Flux 비상업 라이선스와의 관계 본문에 명시 없음.
7. **데이터 분류기들이 비공개.** AIGC 탐지기 / 미학 모델 / OCR 분류기 등 — 재현에 결정적이지만 구체 구조나 가중치는 미공개.

---

## 16. 한 줄 요약 (전체)

> **LongCat-Image 는 Flux 계열 6B 하이브리드 DiT (dual 19 + single 38) 를, AIGC 오염을 적극 차단한 데이터 + 다중 입도 캡셔닝 + 동적 한자 샘플링 + DPO/GRPO/MPO 3단 RLHF 로 학습해, 20B+ 모델과 동급의 일반 성능 + 압도적 중국어 한자 정확도 + 오픈 SOTA 편집을 한 6B 안에 담아낸 효율 중심 이미지 생성 기반 모델이다.**

---

## 17. 관련 메모리 / 문서 링크

### 같은 효율 중심 6B 모델군
- [PAPER_Z-Image.md](PAPER_Z-Image.md) — Tongyi MAI 의 6B Single-Stream DiT + DMD 증류
- [PAPER_DreamLite.md](PAPER_DreamLite.md) — ByteDance 의 0.39B 모바일 우선 모델

### 이미지 편집 관련
- [PAPER_UniCustom.md](PAPER_UniCustom.md) — 같은 LongCat 팀의 멀티 레퍼런스 grounding
- [PAPER_UniRef-Image-Edit.md](PAPER_UniRef-Image-Edit.md) — MSGRPO (flow matching 위 closed-form RL)
- [PAPER_Any2AnyTryon.md](PAPER_Any2AnyTryon.md) — RoPE 채널 재해석으로 mask-free 편집

### 학습 기법 관련
- [PAPER_Min-SNR.md](PAPER_Min-SNR.md) — timestep 가중치 w(t) 일반론
- [PAPER_DMD.md](PAPER_DMD.md), [PAPER_DMD2.md](PAPER_DMD2.md) — Edit-Turbo 가속 비교군
- [PAPER_Lumina-Next.md](PAPER_Lumina-Next.md) — Sandwich Norm / tanh-AdaLN / 2D RoPE 같은 DiT 안정화

### 백본 재사용 풍경
- [reference_pretrained_backbone_reuse_landscape](memory/reference_pretrained_backbone_reuse_landscape.md) — 분기 B (Diffusion 가중치 재사용) 에 해당
