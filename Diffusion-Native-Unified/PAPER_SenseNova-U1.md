# SenseNova-U1: NEO-unify 아키텍처로 이해와 생성을 하나로 통합한 native 멀티모달 모델

## 📋 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | SenseNova-U1: Unifying Multimodal Understanding and Generation with NEO-unify Architecture |
| **저자/기관** | Haiwen Diao 외 57인, **SenseTime** (SenseNova 라인) |
| **공개일** | 2026 (arXiv 2605.12500) |
| **분야** | Unified Multimodal(통합 멀티모달) — Understanding(이해) + Generation(생성), Pixel-space(픽셀 공간) Diffusion Transformer |
| **논문 링크** | [arXiv abstract](https://arxiv.org/abs/2605.12500) · [HTML](https://arxiv.org/html/2605.12500v1) · [PDF](https://arxiv.org/pdf/2605.12500) |
| **코드 (공식)** | https://github.com/OpenSenseNova/SenseNova-U1 |
| **체크포인트** | https://huggingface.co/collections/sensenova/sensenova-u1 |
| **데모** | https://unify.light-ai.top/ |
| **Backbone 초기화** | **NEO** (EvolvingLMMs-Lab의 native VLM, https://github.com/EvolvingLMMs-Lab/NEO) |
| **뿌리 LLM(추정)** | 논문 미명시. 크기 표기(8B dense / 30B-A3B MoE)가 Qwen3 계열 구성과 일치 → Qwen3 계열 추정 |
| **추론 엔진** | **LightLLM**(이해) + **LightX2V**(생성) 분리 배포(disaggregated) |

---

## 📖 주요 용어 사전 (Glossary)

> 규칙: 주요 학술 용어는 **영어(한국어)** 순서로 표기. 처음 나올 때만 풀이하고 이후는 영어 단독.

### 아키텍처
- **native(네이티브) 통합 모델**: 부품을 따로 두지 않고, **날 것의 픽셀(raw RGB pixel)을 하나의 Transformer가 직접 받아 이해도 하고 생성도 하는** 방식. (기존 모델은 이해=비전 인코더+LLM, 생성=VAE+diffusion으로 부품이 갈라져 있음)
- **NEO-unify**: 본 논문의 아키텍처 이름. **Visual Encoder(비전 인코더, VE)도 VAE도 없이** 픽셀과 단어를 한 모델 안에서 깊이 얽어(deeply correlated) 처리하는 native VLM 원시 블록(NEO)을 통합 생성까지 확장한 것.
- **MoT (Mixture-of-Transformers, 트랜스포머 혼합)**: ⚠️ MoE(전문가 혼합)와 다름. **모달리티(modality, 입력 종류)별로 가중치를 완전히 분리(full parameter decoupling)**한 Transformer. 텍스트용/이미지용 projection·정규화(normalization)·FFN을 따로 두고 **Q/K/V까지 분리**, 토큰 종류로 라우팅(routing). 단 **self-attention 계산(어텐션 안에서 서로 보는 것)은 공유**.
- **Pre-Buffer(프리버퍼) layer**: 백본 본체 앞에 두는 **얕은 층 몇 개**. raw pixel과 text 입력을 통합 표현(unified representation)으로 바꿔주고, dense 8B 변종에서는 **사전학습된 LLM의 언어·추론 능력을 보존**하는 역할. (A3B 변종에는 없음)
- **Native RoPE(회전 위치 임베딩)**: 시간·공간 위치를 한 표현으로 통합. **텍스트는 시간축(T)**, **이미지는 높이(H)·너비(W) 축**을 사용.
- **visual interface(비주얼 인터페이스)**: 무거운 VAE 대신 이미지를 넣고 빼는 **가벼운 입출력 모듈**. 입력=2층 convolution(합성곱), 출력=이해는 linear, 생성은 MLP.

### 핵심 개념
- **understanding(이해) 스트림 vs generation(생성) 스트림**: 같은 시각 토큰(visual token)을 **다르게 처리하는 두 경로**. 이해=깨끗한(clean) 토큰으로 내용 파악, 생성=노이즈 낀(noise-conditioned) 토큰으로 그림 만들기.
- **hybrid attention mask(혼합 어텐션 마스크)**: 한 시퀀스 안에서 토큰 종류별로 보는 범위를 다르게 줌. 이해=이미지 블록 안은 양방향(bidirectional)+텍스트엔 causal(앞만 보기), 생성=노이즈 토큰이 양방향.
- **resolution-adaptive noise scaling(해상도 적응 노이즈 스케일링, σ_R)**: 해상도가 커져 토큰 수가 늘어도 **토큰 하나당 노이즈 에너지를 일정하게** 유지해서 SNR(신호대잡음비)을 맞추는 트릭.
- **flow matching(플로우 매칭)**: 노이즈에서 진짜 이미지로 가는 곧은 경로를 학습. 본 논문은 픽셀 공간에서 **x-예측(clean 이미지 직접 예측) + v-loss(velocity, 속도 손실)** 사용.

### 학습·후처리
- **Flow-GRPO**: flow matching 모델용 **GRPO(Group Relative Policy Optimization, 그룹 상대 정책 최적화)** — 후보를 여러 개 뽑아 상대 비교로 학습하는 RLHF. 텍스트 렌더링·스타일·미감(aesthetic) 보상.
- **DMD (Distribution Matching Distillation, 분포 일치 증류)**: 많은 스텝(100)짜리 teacher의 생성 분포를 적은 스텝(8)짜리 student가 따라하도록 압축(distillation).

### 평가 지표
- **MMMU / MMBench / OCRBench / MathVista**: 이해 능력 평가. 각각 대학 수준 멀티모달 추론 / 종합 인지 / 글자 읽기(OCR) / 수학+그림.
- **GenEval / DPG-Bench**: 생성 평가. 요청한 구성요소를 정확히 그리는지(compositional) / 빽빽한 프롬프트를 얼마나 충실히 따르는지(dense prompt).
- **CVTG-2K / LongText-Bench / TIIF-Bench**: 이미지 내 복잡한 텍스트 렌더링 / 장문 텍스트 / 지시문 충실도.

---

## 🎯 논문 요약 (TL;DR)

**한 줄 요약**: **VAE도, 별도 비전 인코더도, diffusion 헤드도 다 제거**하고 raw RGB pixel을 얇은 convolution으로 바로 백본에 넣는 native 구조(NEO에서 초기화) 위에서, **understanding(이해)과 generation(생성)을 하나의 과정으로** 통합한 SenseTime의 모델. dense **16.4B**(8B-MoT)와 MoE **38.2B/활성 ~3B**(A3B-MoT) 두 변종을 공개.

**핵심 문제**:
1. 기존 통합 모델은 겉만 통합이고 속은 (비전 인코더+LLM)과 (VAE+diffusion)로 **구조가 파편화(fragmentation)**되어 있음.
2. 이해와 생성이 서로 다른 표현 공간에서 놀아 **시너지(synergy)가 안 남**.

**해결책**:
1. **NEO-unify**: 인코더·VAE 없이 픽셀·텍스트를 한 백본에. 이해와 생성을 "하나의 과정을 보는 두 시선(synergistic views of a single underlying process)"으로 취급.
2. **MoT 백본**: 모달리티별 가중치 완전 분리 + self-attention 공유로, 매 layer에서 이해와 생성이 native하게 상호작용.
3. **가벼운 비주얼 인터페이스**: 입력 2층 conv(32배 다운샘플), 출력은 이해=linear·생성=MLP로 **픽셀 패치 직접 예측**.
4. **6단계 학습 + Flow-GRPO(RL) + DMD(증류, 100→8 step)**.

**검증**:
- 이해: MMMU 80.55·OCRBench 91.9·MathVista 85.3(A3B) — Qwen3-VL-8B 같은 강한 이해 전용 모델과 대등/상회.
- 생성: GenEval 0.91, DPG 88.14, CVTG-2K 0.940 — Qwen-Image·BAGEL급.
- **VLA(vision-language-action)·WM(world model)** 확장까지 예비(preliminary) 증거 제시.

---

## 🚀 핵심 기여 (Contributions)

1. **native 통합 아키텍처 NEO-unify**: 비전 인코더·VAE를 모두 없애고 픽셀·텍스트를 한 백본에서 처리, 이해와 생성을 단일 과정으로 통합.
2. **MoT 백본**: 모달리티별 파라미터를 완전 분리하되 self-attention을 공유해, 지각(perception)과 합성(synthesis)이 매 layer에서 native하게 상호작용.
3. **두 변종 공개**: dense 8B-MoT(16.4B)와 MoE A3B-MoT(38.2B/활성 ~3B). 이해·생성 양쪽에서 강한 성능.
4. **상용급 풀스택 레시피**: 6단계 학습 + Flow-GRPO(RL) + DMD(증류) + 분리형 추론 엔진(LightLLM/LightX2V).
5. **로드맵 제시**: VLA·World Model 시나리오로의 확장 가능성을 예비 실험으로 시사.

---

## 🏗️ 주요 알고리즘 설명

### 0. 큰 그림 — 왜 "부품을 없애는" 설계인가

> *왜 이 절을 두냐: 이 모델의 모든 선택은 "기존 통합 모델의 내부 분리(separation)를 어떻게 없앨까"라는 한 질문에서 나온다. 그 답이 NEO-unify.*

기존 통합 모델은 이렇게 갈라져 있다.
- **이해**: Visual Encoder(비전 인코더) → LLM (이미지를 이해용 특징으로 뽑음)
- **생성**: VAE Encoder → Diffusion → VAE Decoder (이미지를 잠재(latent)로 압축했다 복원)

문제는 두 경로가 **서로 다른 표현 공간**을 써서, 한 모델 안에 있어도 진짜로 협력하지 못한다는 것. NEO-unify는 이 부품들을 다 걷어내고 **raw RGB pixel을 하나의 Transformer가 직접** 다루게 만든다. 이해와 생성은 "같은 과정을 다른 각도에서 본 것"이라는 관점.

이는 native UMM(native Unified Multimodal Model) 계보 — [[paper_tuna_2]](인코더 0개, Conv2d 주입), [[paper_hidream_o1_image]](pixel-level 통합), [[paper_pixeldit]]·[[paper_minit2i]]·[[paper_pid]](VAE 없는 pixel diffusion) — 와 같은 흐름이며, 그중 **이해까지 SOTA로 노리는** 버전이다.

<p align="center">
  <img src="figures/sensenova_u1_fig3_overview.png" alt="SenseNova-U1 Figure 3 — Overview" width="900"/>
</p>

> **Figure 3** | SenseNova-U1 개요. 극도로 가벼운 인코딩/디코딩 인터페이스로 픽셀-단어(pixel-word) 대응을 하나의 end-to-end 구조 안에서 처리.

#### 통합 아키텍처의 3분류 (Fig 4가 정리한 taxonomy)

> *왜 이 그림이 중요하냐: "우리가 무엇을 없앴는지"를 경쟁 구조와 나란히 놓아야 native의 의미가 잡힌다.*

<p align="center">
  <img src="figures/sensenova_u1_fig4_neo_unify.png" alt="SenseNova-U1 Figure 4 — three unified architectures" width="950"/>
</p>

> **Figure 4** | 통합 멀티모달 아키텍처 세 갈래. SenseNova-U1은 (3)에 해당.

| 갈래 | 구성 | 이미지 입출력 | 대표 모델 |
|---|---|---|---|
| **(1) Sequential** | Vision Encoder + Projector + **VAE 인코더/디코더** | 이해=Vision Encoder, 생성=VAE | Omnigen2, Qwen-Image |
| **(2) Parallel (shared attn)** | Vision Encoder + **VAE** + **shared MHSA** (FFN만 분리) | 이해=Vision Encoder, 생성=VAE | **MoT**(원조), LMFusion, BAGEL |
| **(3) Parallel Native** ⭐ | **가벼운 인코딩/디코딩 층만** (Patch Emb1/Emb2 + MLP Head), shared MHSA | 이해=Patch Emb1(clean 이미지), 생성=Patch Emb2(noise pixel) | **NEO-unify, SenseNova-U1** |

포인트 2가지:
- SenseNova-U1의 "**-MoT**" 이름은 (2)/(3)의 **parallel + shared self-attention** 계보에서 왔다. 즉 MoT는 "attention은 공유하되 FFN 등 모달리티별 가중치를 분리"하는 병렬 백본을 가리키는 말.
- SenseNova-U1의 진짜 기여는 (2)→(3) 이동, 즉 **Vision Encoder와 VAE를 통째로 걷어내고** 가벼운 patch embedding + MLP head로 대체한 것. (Fig 4에서 (3)만 인코더·VAE 상자가 사라진 게 눈에 보인다.)

### 1. Visual Interface — 인코더/디코더 없이 픽셀을 넣고 빼기

> *왜 이 절을 두냐: "VAE를 없앤다"는 말은 곧 "VAE가 하던 압축·복원을 무엇으로 대신하나"라는 질문. 그 답이 이 가벼운 입출력 모듈.*

- **입력(인코딩)**: 2층 convolution(합성곱) + GELU 활성화로 raw 이미지를 시각 토큰(visual token)으로 변환.
  - stride 16×2로 **32배 다운샘플(downsample)** → 32×32 패치 토큰 + 2D 사인파 위치 임베딩(sinusoidal positional encoding).
  - 무거운 VAE encoder가 이 conv 두 층으로 대체됨.
- **출력(디코딩)**:
  - understanding 스트림: **linear projection**으로 다음 텍스트 토큰 예측.
  - generation 스트림: **MLP 헤드가 픽셀 패치를 직접 예측** — 깊은 diffusion 헤드도, VAE decoder도 거치지 않음(bypass).
- **효과**: 32배 압축을 하면서도 semantic(의미)과 pixel-level(픽셀 수준) 충실도(fidelity)를 함께 유지.

### 2. MoT 백본 — 모달리티별 가중치를 나누되 어텐션은 공유

> *왜 이 절을 두냐: "이해와 생성을 한 모델에" 넣을 때 가장 어려운 건 두 작업의 성격이 너무 달라 서로 방해한다는 점. MoT가 이 충돌을 푸는 장치.*

MoT(Mixture-of-Transformers)의 세 요소:
1. **full parameter decoupling(완전 파라미터 분리)**: 각 layer에서 projection·normalization·FFN을 **텍스트용/이미지용으로 따로** 두고, **Q/K/V projection까지 분리**. 토큰 종류(token type)에 따라 동적으로 라우팅.
2. **shared self-attention(공유 self-attention)**: 그럼에도 **어텐션 계산 자체는 한 시퀀스 안에서 공유** — 모든 모달리티가 서로를 본다. hybrid mask로 보는 범위만 조절.
3. **Native RoPE**: 텍스트=시간축(T), 이미지=높이(H)·너비(W) 축으로 위치를 통합.

> 비유: MoT는 "한 회의실(공유 attention) 안에 앉은 **두 전문가(이해 담당·생성 담당)**"다. 각자 자기 노트(분리된 가중치)를 쓰지만, 같은 테이블에서 서로의 말을 듣는다. → HiDream-O1의 "**한 사람이 두 일을 다 함(완전 공유)**"과 대비되는 지점([[paper_hidream_o1_image]]).

### 3. 같은 토큰, 다른 어텐션 — 이해와 생성의 분기

> *왜 이 절을 두냐: 이해와 생성이 "같은 시각 토큰"을 쓴다면, 둘을 가르는 실제 스위치가 무엇인지 밝혀야 한다. 그 스위치가 노이즈 여부 + 어텐션 방향.*

- **understanding(이해)**: 깨끗한(clean) 이미지 토큰이 이미지 블록 안에서 **양방향(bidirectional)**으로 서로 보고, 텍스트에는 **causal(앞만 보기)**로 조건화.
- **generation(생성)**: **노이즈 낀(noise-conditioned) 토큰**이 깨끗한 입력과 다른 노이즈 토큰을 **양방향**으로 봄 → 텍스트를 조건으로 한 확산.
- **resolution-adaptive noise scaling(σ_R)**: 해상도가 바뀌어 토큰 수 N(H,W)=(H·W)/32²가 달라져도 SNR을 일정하게. `σ_R(H,W) = σ₀·√(N(H,W)/N₀)` — √ 스케일링으로 토큰당 노이즈 에너지를 대략 일정하게 유지.

<p align="center">
  <img src="figures/sensenova_u1_fig6_hybrid_attn.png" alt="SenseNova-U1 Figure 6 — hybrid attention" width="920"/>
</p>

> **Figure 6** | 통합 prefill용 hybrid attention. **텍스트 행(text rows)은 표준 causal mask**(앞만 봄), **이미지 행(image rows)은 앞쪽 텍스트 prefix + 이미지 전체 구간(full image span)**을 봄. 오른쪽은 M-block 단위 커널 선택(이미지 토큰이 있으면 K 범위를 이미지 구간 끝까지 확장, 없으면 그냥 causal). Triton·FA3로 구현.

### 4. 학습 목표 (Objective)

> *왜 이 절을 두냐: 성격이 다른 두 손실을 어떻게 한 저울에 올리는지가 통합 학습의 핵심.*

- 텍스트(이해): 표준 cross-entropy(다음 토큰 예측).
- 이미지(생성): **픽셀 공간 flow matching** — rectified-flow 보간 `z_t = t·x + (1−t)·σ_R·ε` (즉 진짜 이미지 x와 노이즈 ε를 t 비율로 섞되 노이즈에 σ_R를 곱함)에서 **x-예측(clean 이미지 직접 예측) + v-loss(velocity MSE)**.
- 합산: `L_total = λ₁·L_Und + λ₂·L_Gen`, 대표값 **λ₁=0.1(이해), λ₂=1.0(생성)** — 생성이 훨씬 어려워 무게를 더 실음.

### 5. 학습 레시피 (6단계) + 후처리

> *왜 이 절을 두냐: native 통합 모델이 이해·생성 둘 다 잘하려면 순서가 중요하다. 어느 능력을 먼저 심고, 언제 합치는지의 지도.*

| 단계 | 초점 | 규모 | 핵심 |
|---|---|---|---|
| 1 | **이해 워밍업** | ~120K step | NEO 백본에서 이어받아 attention 융합 → 전체 학습 |
| 2 | **생성 사전학습** | ~300K step (3 phase) | 256~1024px → 512~2048px → 편집/interleaved |
| 3 | **통합 mid-training** | ~84K step | 이해+생성 섞어 공동 최적화 |
| 4 | **SFT** | ~9K step | 지시문 정렬(instruction alignment) — 상세는 아래 소절 |
| 5 | **RL 후처리** | — | **Flow-GRPO** (텍스트 렌더링·스타일·미감 보상) |
| 6 | **증류** | — | **DMD** 100 step → 8 step |

<p align="center">
  <img src="figures/sensenova_u1_fig8_data_dist.png" alt="SenseNova-U1 Figure 8 — data distribution" width="900"/>
</p>

> **Figure 8** | SenseNova-U1 학습 corpus의 계층적 데이터 분포. **왼쪽부터 4개 sunburst = ① 이해(Understanding) ② 생성 T2I(Generation) ③ 편집(Editing) ④ 인터리브(Interleaved)**. 안쪽 고리=대분류, 바깥 고리=세부.

**Fig 8의 4개 차트 상세 분해**

① **이해(Understanding)** — 대분류 4개:
- General 39.23% (General VQA / Captioning / Multi-turn 대화 / Multi-image / OCR ~7.19% / Cognition 등)
- Agent and Spatial 22.27% (Perspective-Taking / Spatial Relations / Grounding / Counting / GUI·Open perception 등 — **공간지능이 두 번째로 큼**)
- Knowledge Reasoning 19.28% (Comprehension / Science / Math)
- Pure Text 19.23% (Math ~5.16% / Knowledge / Code / Other)

② **생성 T2I(Generation)** — 대분류 5개:
- Nature 40.53% (Objects가 최대, Landscape ~5.5% / Cityscape ~4.20% / Animals / Food / Plants / Indoor)
- People 26.65% (English Text / Chinese Text ~3.20% / Human-Object / Sports / Portrait / Social Scene / Body Parts — **텍스트 렌더링이 People 안에 포함**)
- Design 20.68% (Arts ~8.40% / Posters ~5.11% / Slides / Cartoon / UI)
- Synthetic 12.14%
- Infographics 2.06%

③ **편집(Editing)** — 대분류 4개:
- Nature 52.30% (Composite ~19.72% / Subject Add ~12.65% / Color Alteration ~7.81% / Background Change ~3.12% / Portrait Edit ~2.26% / Motion Change ~1.37% / Identity Transfer / Subject Removal·Replacement / Reasoning)
- Synthetic 18.66% (대부분 Infographics Edit)
- People 14.68%
- Design 11.05%

④ **인터리브(Interleaved)** — 대분류 4개:
- Lifestyle 43.85% (튜토리얼·일상·그림책 등)
- Infographics 29.24%
- Video 19.22% (VBRS ~7.42% / 추론 / 롱·숏 비디오)
- Reasoning ~8%

> ⚠️ 세부(바깥 고리) 수치는 그림에서 읽은 근사값이며, 대분류 수치가 논문 본문과 일치하는 확정값.

#### 학습 데이터 상세 구성

> ⚠️ **중요**: 논문은 데이터를 **token(토큰) 예산과 비율(%)로만** 공개하고, **이미지·샘플 절대 개수(count)는 밝히지 않는다.** 아래 절대량 수치는 모두 token 기준이며, "몇 억 장" 같은 이미지 수는 이 논문에서 알 수 없다.

**단계별 token 예산(Table 2)** — 메인 학습 총 **~3.29T token**:

| 단계 | Steps | Tokens | 데이터 믹스 |
|---|---|---|---|
| 1 이해 워밍업 | 120K | 0.75T | 이해 100% |
| 2-I 생성 사전학습 | 120K | 0.25T | 생성 100% (256~1024px) |
| 2-II 생성 사전학습 | 60K | 0.25T | 생성 100% (512~2048px) |
| 2-III 생성 사전학습 | 120K | — | 생성 + 편집 + 인터리브 |
| 3 통합 mid-training | 84K | 0.88T | 이해 33% / T2I 37% / 편집(editing) 24% / 인터리브 6% |
| 4 SFT | 9K | 0.13T | 3단계와 동일 믹스 |

**이해(understanding) corpus 비율**:
- 사전학습: image-text pairs 32% / captions 17% / infographics 14% / pure text 37% (교차소스 중복제거·안전 필터·품질 필터·CLIP 비율 균형 re-captioning)
- mid-training: General 39.2% / Agent·Spatial 22.3% / Knowledge Reasoning 19.3% / Pure Text 19.2%
  - General 세부: visual QA 26.6% / 멀티턴 대화 26.4% / captioning 20.3% / OCR 18.6% / multi-image 8.2%
  - Knowledge Reasoning 세부: 지식형 12.0% / 추론형 7.2%
- SFT 분포 → 아래 "SFT 상세" 소절 참조

**생성(generation) corpus 비율**:
- T2I 대분류: Nature ~40.5% / People ~26.7% / Design ~20.7% (+ infographics·이중언어 텍스트 렌더링)
- 편집(editing): 자연 장면 ~52.3% / 인물 ~14.7% / 나머지 infographic·합성. 작업 종류 = 추가·제거·배경/색 변경·identity transfer·모션 조작·인물 편집·합성·추론형 변형
- 인터리브(interleaved): lifestyle ~44%(튜토리얼 26%·일상 14%·그림책 4%) / infographics ~29% / video ~19% / reasoning ~8%

#### 데이터 정제 파이프라인 (Fig 7 이해 · Fig 9 생성)

> *왜 이 절을 두냐: "무엇을 얼마나 썼나"(비율)만큼 "어떻게 걸러냈나"(파이프라인)가 품질을 좌우한다. 두 corpus는 정제 철학이 다르다 — 이해는 "다양성·정답성", 생성은 "화질·중복·안전".*

**생성(generation) corpus — 4단계 필터링 (Fig 9)**

<p align="center">
  <img src="figures/sensenova_u1_fig9_gen_pipeline.jpg" alt="SenseNova-U1 Figure 9 — generation data pipeline" width="950"/>
</p>

> **Figure 9** | 생성 corpus 처리 파이프라인. Data Source → (A) 저수준 필터 → Filtered Corpus → (B) 중복제거 → Unique Corpus → (C) VLM 캡셔닝 → (D) 품질 필터 → Training Data.

| 단계 | 하는 일 | 세부 |
|---|---|---|
| **(A) Low-Level Filter(저수준 필터)** | 화질 나쁜 이미지 자동 탈락 | Resolution(해상도)·Lighting(조명)·Color Saturation(채도)·Sharpness(선명도)·Texture Complexity(질감 복잡도)·Energy(에너지)·Compress Rate(압축률)·JPEG Entropy(JPEG 엔트로피)·**SigLIP**(의미 점수) |
| **(B) Deduplication(중복제거)** | 거의 같은 이미지 제거 | **Perceptual Hash(지각 해시)**로 near-duplicate 삭제 → Unique Corpus |
| **(C) VLM Captioning(캡셔닝)** | 이미지마다 설명·태그 자동 생성 | **Qwen3-VL-32B-Instruct**가 Short Caption(짧은 설명)·Long Caption(긴 설명)·Tagging(태그)·Category(분류) 생성 |
| **(D) Quality Filter(품질 필터)** | 미감·유해요소 최종 거름 | Aesthetic Filter(미감 필터) + **Abnormal-Element Filter**(Watermark 워터마크·Blur/Mosaic 흐림·모자이크·NSFW 등) |

→ 핵심 포인트 3가지: ① 캡션을 **Qwen3-VL-32B로 재생성**(원본 alt-text가 아니라 모델이 다시 씀) ② 워터마크·모자이크·NSFW를 명시적으로 제거 ③ SigLIP으로 이미지-텍스트 의미가 안 맞는 것도 저수준에서 컷.

**이해(understanding) corpus — 다양성·정답성 중심 (Fig 7)**

<p align="center">
  <img src="figures/sensenova_u1_fig7_und_pipeline.png" alt="SenseNova-U1 Figure 7 — understanding data pipeline" width="900"/>
</p>

> **Figure 7** | 이해 corpus 처리 파이프라인. (A) 분포 균형 큐레이션 → (B) 프롬프트 증강 → (C) 다기준 품질 필터링.

| 단계 | 하는 일 | 세부 |
|---|---|---|
| **(A) Distribution-balanced Curation(분포 균형 큐레이션)** | 편중 없이 다양하게 뽑기 | **CLIP 기반 다양성 샘플링**(CLIP 인코더 → visual embedding → K-means 클러스터링) + **Attribute Profiling**(perceptual·semantic 지표, 해상도, 선명도, 정보 밀도로 High/Med/Low tier 나눠 stratified sampling) |
| **(B) Prompt Augmentation(프롬프트 증강)** | 질문·응답을 다채롭게 재작성 | Semantic Expression(의미 표현)·Format Structure(Length/List/Code 형식)·Role & Scenario(Teacher/Expert/User 역할)·Task complexity(기초 인식↔복합 인지 합성) → Prompt Augmentation & Uniform Response Rewriting(응답 통일 재작성) |
| **(C) Multi-criteria Quality Filtering(다기준 품질 필터)** | 틀린 QA 쌍 버리기 | Correctness Verification(정답 검증)·Hallucination Detection(환각 탐지)·Instruction-Following(지시 준수) 3기준으로 QA Pair 평가 → Final Refined Set / Rejected Data |

→ 생성이 "예쁘고 안 겹치는 이미지"를 고른다면, 이해는 "**골고루 다양하고 정답이 맞는 QA**"를 고른다.

#### SFT(Supervised Fine-Tuning, 지도 미세조정) 상세 — 4단계

> *왜 이 절을 두냐: SFT는 "새 지식 주입"이 아니라 "이미 배운 능력의 정렬"이다. mid-training과 무엇이 같고 무엇이 다른지를 갈라야 이 단계의 정체가 잡힌다.*

- **목표**: "instruction alignment(지시 정렬)과 작업별 성능을 날카롭게 다듬는다" — 3단계까지 익힌 능력을 **사용자 지시를 잘 따르는 형태로 정렬**.
- **규모·설정**: 9K step / ~0.13T token (3단계의 1/7). 학습률 cosine 스케줄 **2e-5 → 0**. 손실 가중치는 3단계와 동일(λ₁=0.1, λ₂=1.0). **full model**(이해+생성 스트림 모두) 조정.
- **데이터 = 같은 믹스, 다른 선별** ⭐: 완전히 새 데이터가 아니라, mid-training과 **동일한 4종 믹스**(이해+T2I+편집+인터리브)를 유지하되, mid-training **candidate pool(후보 풀)을 quality & difficulty(품질·난이도) 이중 기준으로 정제한 부분집합**만 사용. 새 도메인을 추가하면 능력이 흔들리므로 종류는 유지하고 질만 올리는 방식 — [[paper_z_image]]·[[paper_qwen_image_2]] 등 최근 모델의 공통 SFT 관행.

| | 3단계 mid-training | 4단계 SFT |
|---|---|---|
| 성격 | 넓은 커버리지·규모 확장 | **정제·난이도 집중** |
| 데이터 | 대규모 혼합 pool | 같은 pool에서 **품질+난이도**로 추린 부분집합 |
| 규모 | 84K step / 0.88T | 9K step / 0.13T |
| 믹스·손실 | 이해33%·T2I37%·편집24%·인터리브6% / 0.1:1.0 | **동일** |

- **이해 SFT 능력 분해**(understanding 몫 기준): 공간지능 ~15% / 일반 멀티모달 이해 ~13% / 추론 ~12% / 일반 NLP ~11% / OCR·문서 ~11% / agentic function calling ~10% / 롱컨텍스트 대화 ~8% / 코드 ~6% / 멀티턴 ~4% / 복합 구성 이해 ~4% (나머지 보조). → **공간지능이 SFT에서도 1순위** — VLA/월드모델 지향과 일관.

**후처리 데이터·하이퍼파라미터**:
- RL(Flow-GRPO) 텍스트 렌더링: 600 epoch, epoch당 프롬프트 N=48 × 이미지 K=16(=768장), 10-step, guidance 4.0, noise 0.7, 앞 200 epoch 동적해상도 워밍업
- RL(Flow-GRPO) 통합 일반: **8B 1,600 epoch / A3B 200 epoch**, 보상군 2개 교대, lr 1e-5, KL β=0.01
- 증류(DMD2): 100→8 step, generator는 fake-flow 5회당 1회 갱신, generator lr 2e-6·fake-flow 4e-7, Euler·timestep shift 3.0·CFG 4.0

### 6. 추론 인프라 — 분리형 두 엔진

> *왜 이 절을 두냐: 한 모델이라도 이해와 생성은 최적 병렬화 방식이 달라, 실서비스에선 나눠 굴리는 게 유리하다.*

- **LightLLM**: 멀티모달 이해·텍스트 스트리밍·요청 오케스트레이션 담당 (Tensor Parallelism).
- **LightX2V**: 이미지 생성 담당 (CFG Parallelism + sequence parallelism).
- 두 엔진은 pinned shared memory로 상태를 주고받되 **API는 하나로 통일**. 2048×2048 생성 시 5090 GPU·TP2+CFG2에서 스텝당 ~0.415초.

<p align="center">
  <img src="figures/sensenova_u1_fig5_inference.png" alt="SenseNova-U1 Figure 5 — disaggregated inference" width="880"/>
</p>

> **Figure 5** | 분리형(disaggregated) 추론 구조. LightLLM이 이해·텍스트 스트리밍·제어 흐름을, LightX2V가 이미지 생성을 담당.

---

## 🧪 실험 요약

### 1. Understanding(이해) 벤치마크

| Benchmark | 8B-MoT | A3B-MoT | 의의 |
|---|---|---|---|
| **MMMU** | 74.78 | **80.55** | 대학 수준 멀티모달 추론 |
| **MMBench-EN** | 90.25 | **91.59** | 종합 인지 |
| **OCRBench** | 82.10 | **91.90** | 글자 읽기 강함 |
| **MathVista** | 84.20 | **85.30** | Qwen3-VL-8B 등 상회 |

→ **이해 전용 최상위 모델과 대등하거나 상회**. 특히 수학·텍스트 리치(text-rich) 이해에서 강점.

### 2. Generation(생성) 벤치마크

| Benchmark | SenseNova-U1 | 비교 |
|---|---|---|
| **GenEval** Overall | **0.91** | Qwen-Image 0.87, BAGEL 0.82 상회 |
| **DPG-Bench** (A3B) | 88.14 | Qwen-Image 88.32와 대등 |
| **CVTG-2K** (8B / A3B) | 0.940 / 0.881 | 복잡 텍스트 렌더링 |
| **LongText-Bench EN** (8B) | 0.979 | 장문 텍스트 |
| **TIIF-Bench** (8B) | 최고 | 짧은·긴 지시문 모두 |

→ **이해와 생성을 한 몸으로** 하면서 Qwen-Image·BAGEL급 생성 달성.

### 3. Ablation(제거 실험) 요지

- **5.2.1** native encoder-free 설계가 semantic·pixel 표현을 **둘 다 보존**함을 입증.
- **5.2.2** 이해와 생성이 **MoT 백본에서 시너지**를 냄(서로 도움).
- **5.2.3** native 구조가 **데이터 스케일링 효율**이 높음 — 더 적은 학습 자원으로 인코더 기반 최상위 구조와 대등.

**시너지 실측(Fig 12, 8B-MoT의 mid-training 전/후)** — 통합 학습(MT, mid-training)을 하면 이해·생성·편집 **네 항목이 동시에 오른다**(서로 방해가 아니라 도움):

| 항목 | Before MT | After MT |
|---|---|---|
| Multimodal Understanding (Avg) | 72.1 | **73.2** |
| Text-Only Understanding (Avg) | 77.7 | **77.8** |
| Image Generation (DPG) | 84.7 | **86.5** |
| Image Editing (GEdit) | 69.1 | **69.9** |

<p align="center">
  <img src="figures/sensenova_u1_fig12_mot_synergy.png" alt="SenseNova-U1 Figure 12 — understanding-generation co-training" width="720"/>
</p>

> **Figure 12** | 8B-MoT 백본에서 이해–생성 공동 학습(co-training) 효과. GEdit 점수는 0–100으로 정규화.

### 4. VLA / World Model 확장

- 논문은 **예비(preliminary) 증거**로 vision-language-action(VLA)과 world model(WM) 시나리오에서 강하다고 주장.
- 다만 구체적 태스크·데이터셋·수치는 제시되지 않음 — "더 넓은 로드맵을 가리킨다"는 방향 제시 수준.
- Fig 14는 **로봇 조작(robotic manipulation) 영상**에서의 vision-language-action 행동을, Fig 15는 **로봇 팔 시점(robotic arm view)의 미래 예측**(world-modeling prediction)을 시각화 — 통합 표현이 행동·미래 예측으로도 뻗을 수 있음을 정성적으로 보여줌.

<p align="center">
  <img src="figures/sensenova_u1_fig14_vla.jpg" alt="SenseNova-U1 Figure 14 — VLA" width="820"/>
</p>

> **Figure 14** | 로봇 조작 영상에서의 vision-language-action(VLA) 행동 시각화.

<p align="center">
  <img src="figures/sensenova_u1_fig15_world_model.png" alt="SenseNova-U1 Figure 15 — world modeling" width="820"/>
</p>

> **Figure 15** | 로봇 팔 시점의 world-modeling(미래 프레임) 예측 시각화.

---

## 💬 Q&A 섹션

### Q1. 이 모델의 실제 크기는? ("8B"가 전체인가?)

이름의 "8B"는 **스트림 하나당 크기**일 뿐, 전체가 아니다. MoT는 이해용·생성용 가중치를 완전 분리하고 Q/K/V까지 나누므로 합산해야 한다.

| 변종 | 저장(total) | 활성(per-token) | 구성 |
|---|---|---|---|
| **8B-MoT** (dense) | **16.4B** | **16.4B** (전부 활성) | 이해 8.2B + 생성 8.2B / 42 layer, hidden 4096, Q head 32·KV head 8 / Pre-Buffer 있음 |
| **A3B-MoT** (MoE) | **38.2B** | **~3B** | 이해 30B(전문가 128, top-8) + 생성 8.2B(전문가 32) / 48 layer, hidden 2048, Q head 32·KV head 4 / Pre-Buffer 없음 |

즉 실질 규모는 **16.4B**와 **38.2B**. "A3B"는 Active 3B(MoE라 토큰당 ~3B만 활성)를 뜻한다.

### Q2. 백본은 무엇이고, 원래(prior) 모델은?

- **직접 백본 = NEO**(EvolvingLMMs-Lab의 native VLM, "from first principles"). 인코더·VAE 없이 픽셀·단어를 한 dense 모델에서 처리하는 원시 블록. 원래 NEO는 2B·9B로 공개.
- 논문 표현: "pretrained NEO에서 초기화한 뒤 효율 개조 2가지 적용."
- **뿌리 LLM은 논문 미명시.** 다만 이해 베이스라인 크기(dense 8B / MoE 30B-A3B·전문가 128·top-8)가 **Qwen3 계열 구성과 거의 일치**해 Qwen3 계열로 **추정**(확정 아님).

### Q3. 가장 닮은 모델과의 비교 — HiDream-O1-Image

> 둘 다 "VAE 없음 + 외부 텍스트 인코더 없음 + 픽셀 직접 + hybrid causal/full 어텐션 + flow matching + GRPO + DMD"를 공유하는 같은 세대 native 통합 생성기. **결정적 차이는 파라미터 통합 방식.**

| 항목 | **SenseNova-U1** | **HiDream-O1-Image** ([[paper_hidream_o1_image]]) |
|---|---|---|
| 출처 | SenseTime, 2026 | HiDream.ai, 2026-05 |
| 핵심 목표 | 이해+생성 통합, **이해 SOTA + VLA/WM**까지 | **생성·편집·IP 통합** (이해는 부차적) |
| 백본 | **NEO** (native VLM) | **Qwen3-VL-8B-Instruct** (이해 모델 재사용) |
| 실제 규모 | **16.4B / 38.2B**(활성 ~3B) | **8B** 단일 / Pro **200B+** |
| VAE | 없음 | 없음 |
| 외부 텍스트 인코더 | 없음 | 없음 |
| 이미지 입력 | **2층 conv로 raw RGB 주입** (조건 이미지까지 통일) | 32×32 patchify(무손실). 단 **참조 이미지는 SigLip-2 인코더** |
| encoder-free 순도 | ✅ 조건까지 conv | △ 참조는 별도 인코더 |
| **파라미터 공유 (급소)** | **MoT — 모달리티별 완전 분리** (Q/K/V까지), attention만 공유 | **single-stream 완전 공유** (같은 Q/K/V weight) |
| 어텐션 | hybrid (이해=블록내 양방향+텍스트 causal / 생성=양방향) | hybrid (AR=causal / generation=full) |
| 생성 loss | flow matching (x-예측 + v-loss) | flow matching + **LPIPS + DINO** |
| RL / 증류 | **Flow-GRPO** / **DMD** 100→8 | **GRPO** / **DMD + GAN** 50→28 |
| 프롬프트 에이전트 | (명시 없음) | **Reasoning Prompt Agent** (Gemma-4-31B, "O1" 유래) |
| 대표 성적 | GenEval 0.91 / MMMU 80.55·OCRBench 91.9 | GenEval 0.90(8B)·0.92(Pro) / LongText EN 0.979 |

**세 줄 결론**:
1. 뼈대는 같은 세대(VAE·텍스트 인코더 제거 + 픽셀 + hybrid attention + flow matching + GRPO + DMD).
2. **급소 차이 = 통합 방식**. HiDream-O1은 "하나의 가중치를 전부 공유"(가벼움, 8B), SenseNova-U1은 "MoT로 이해·생성 가중치 완전 분리"(무거움, 16.4B/38.2B). "진짜 하나(공유) vs 한 몸 안 두 전문가(분리)"의 철학 차이.
3. 지향점: SenseNova-U1은 **이해 SOTA + VLA/WM까지 노리는 범용 멀티모달 두뇌**, HiDream-O1은 **생성·편집·IP 완성도**에 집중한 이미지 생성기. encoder-free 순도도 SenseNova-U1이 더 높음.

### Q4. 학습 표의 "step"이 의미하는 게 뭐야? 배치 사이즈는?

**step(스텝) = 가중치를 한 번 업데이트하는 단위.** 학습 한 바퀴는 ① batch(배치, 데이터 한 묶음)를 꺼내 ② loss(오차)를 재고 ③ 가중치를 한 번 고치는 것 — 이 한 바퀴가 1 step. "120K step" = 모델을 12만 번 고쳐 썼다는 뜻.

**비유 하나로 전부 이해하기: "문제집 채점"**

- **token** = 문제 하나하나
- **batch** = 한 번에 몰아서 채점하는 문제 묶음
- **step** = "채점하고 → 오답 원인을 분석해서 → 공부 방법을 한 번 고치는" 행위 1회
- **epoch** = 문제집 전체를 처음부터 끝까지 한 바퀴 다 푸는 것

핵심: **모델은 문제를 풀 때마다 똑똑해지는 게 아니라, "채점 후 방법을 고칠 때(step)" 똑똑해진다.** 그래서 학습 진행을 "몇 문제 풀었나"가 아니라 "몇 번 고쳤나(step)"로 센다.

**step당 학습량 역산** (총 token ÷ step 수) — **620만 token이 곧 1개 배치 = 1 step에서 소비하는 양**:

| 단계 | step | 총 token | → 1 step당 token(배치) |
|---|---|---|---|
| 1 이해 워밍업 | 120K | 0.75T | **~620만** |
| 2-I 생성 | 120K | 0.25T | ~210만 |
| 3 mid-training | 84K | 0.88T | ~1,050만 |
| 4 SFT | 9K | 0.13T | ~1,440만 |

**"620만 token = 이미지+캡션 몇 개?"** — 3단계 계산:

① 이미지 1장 = 몇 token? 32배 다운샘플이라 (가로÷32)×(세로÷32):

| 해상도 | 이미지 token |
|---|---|
| 256×256 | 8×8 = **64개** |
| 512×512 | 16×16 = **256개** |
| 1024×1024 | 32×32 = **1,024개** |

② "이미지+캡션" 1세트 = 몇 token? 캡션(짧은 것 ~50-100, 긴 것/QA ~200-500)을 더해:

| 구성 가정 | 1세트 token |
|---|---|
| 512px + 짧은 캡션 | 약 350~400 |
| 512px + 긴 캡션/QA | 약 600~800 |
| 1024px + 캡션 | 약 1,200~1,500 |

③ 620만 ÷ 세트 크기:

| 가정 | 배치 1개 안의 세트 수 |
|---|---|
| 512px 중심 (~400 token/세트) | **약 15,000개** |
| 혼합 (~800 token/세트) | **약 8,000개** |
| 1024px 중심 (~1,300 token/세트) | 약 5,000개 |

→ **한 배치(1 step)에 이미지+캡션 대략 5천~1만 5천 세트, 자릿수로 "약 1만 개"**. 단 1단계는 순수 텍스트 37%(길이 수천 token)가 섞여 있어 실제론 수천~1만여 개. 긴 멀티턴 대화 시퀀스(4K token) 기준으로 세면 약 1,500개. 전체로 환산하면 1단계 0.75T ÷ ~600 token/세트 ≈ **약 12억 세트 투입**(중복 포함 image-pass).

**epoch(에포크)와 구분**: epoch = 전체 데이터를 한 바퀴 = 수만~수십만 step. 사전학습은 데이터가 워낙 커서 step으로 세고, RL처럼 데이터가 작은 단계(프롬프트 48개)는 같은 세트를 여러 바퀴(1,600 epoch) 돌아 epoch로 셈.

⚠️ 실제 GPU 수·시퀀스 길이·배치 구성은 논문 미공개 — token/step만 확정, 세트 개수는 역산 추정.

### Q5. 한 번에 그렇게(1만 세트씩) 학습하는 게 맞나?

✅ 맞다 — 오히려 **대규모 학습의 표준**. "한 번에 1만 세트를 우겨넣는 게 낭비 아냐?"라는 의심이 들 수 있는데, 이유를 뜯어보면 납득이 된다.

**1) 유명한 선례: CLIP은 배치가 3만 2천 개였다**

OpenAI CLIP이 **한 배치에 이미지-텍스트 32,768쌍**으로 학습했고, SigLIP도 3만대 배치. Stable Diffusion 2,048장, 요즘 LLM들(Llama 3 등)이 배치 1,600만 token까지 쓴다. SenseNova-U1의 "수천~1만여 세트/배치"는 이 관행의 한가운데 있는 평범한 값.

**2) 왜 크게 잡는 게 "맞는" 학습인가**

- **이유 A — 방향이 정확해진다.** 샘플 1개의 gradient(수정 방향)는 잡음투성이. 고양이 사진 1장 보고 공부법을 고치면 "고양이 쪽으로" 확 쏠린다. 1만 개를 평균 내면 우연한 쏠림이 서로 상쇄되고 **데이터 전체가 원하는 진짜 방향**만 남는다. 통계에서 표본을 늘리면 평균이 정확해지는 것과 같은 원리.
- **이유 B — GPU가 놀지 않는다.** GPU 수백 장을 동원한 이상, 각 장이 뭔가를 물고 있어야 한다. 배치가 작으면 GPU들이 놀면서 통신(gradient 교환)만 기다린다. 큰 배치 = 계산 대 통신 비율이 좋아져 같은 시간에 더 많이 학습.
- **이유 C — 큰 모델일수록 안정성.** 16.4B짜리 모델은 gradient가 조금만 튀어도 loss가 폭주(spike)할 수 있다. 큰 배치는 gradient를 부드럽게 만들어 폭주를 막는 안전벨트.

**3) 무한정 키우면 되나? — 아니, 한계 있음**

**critical batch size(임계 배치 크기)**: 어느 지점까지는 배치를 키우면 step을 그만큼 줄여도 같은 성능(이득)이지만, 임계점을 넘으면 평균이 이미 충분히 정확해서 더 키워봤자 **계산 낭비**. 그래서 620만~1,440만 token이지 6억 token이 아니다. 현세대 경험칙으론 수백만~수천만 token이 임계점 근처. 또 배치를 키우면 **learning rate도 같이 키워야** 한다(수정 기회가 귀해지니 보폭을 크게) — 이 균형이 어긋나면 큰 배치가 일반화를 해친다는 논쟁(sharp minima)도 있었지만 현대적 스케줄링으로 대부분 해결.

**4) "한 번에"의 오해 정정** — 물리적으로 1만 세트가 동시에 한 GPU에 들어가는 게 아니다:

```
GPU 512장 × 장당 몇십 세트 (동시에)
+ 필요하면 gradient accumulation (한 장이 여러 번 나눠 계산 후 합침)
= 논리적으로 "한 배치 1만 세트"
→ 가중치 업데이트는 딱 1회
```

즉 "한 번에 학습한다" = **수정을 한 번에 한다**는 뜻이지, 메모리에 한꺼번에 다 올린다는 뜻이 아니다.

| 의심 | 실제 |
|---|---|
| 1만 세트씩 한 번에? 너무 크지 않나 | CLIP 3.2만, LLM 1,600만 token — 표준 범위 |
| 왜 크게? | 방향 정확도↑ + GPU 효율↑ + 안정성↑ |
| 무한정 좋나? | ❌ critical batch size 존재 — 그래서 딱 이 수준 |
| 물리적으로 가능? | 수백 GPU 분산 + accumulation으로 구현 |

한 줄: **큰 배치는 "몰아서 대충"이 아니라 "평균 내서 정확하게"** — 이 논문의 배치는 업계 임계점 안의 교과서적 크기.

### Q6. 총 3.29T 토큰이면 학습량이 너무 적은 거 아냐?

적어 보이지만 **맥락 두 개를 넣으면 적지 않다**:

1. **3.29T는 "추가분"**: pretrained NEO(뿌리는 Qwen3 계열 추정, LLM 사전학습은 통상 15~36T token)에서 초기화 → 언어·추론 능력은 이미 수십조 token짜리 유산으로 깔려 있고, 3.29T는 **멀티모달 통합 능력만 얹는 비용**.
2. **이미지 기준으론 표준~넉넉**: 생성 몫 ~0.8T token ≈ 수억~십억 회 이미지 투입(자릿수 추정) — T2I 관행 수준. 32배 다운샘플 덕에 이미지 1장당 token이 적어(1024px=1,024토큰) 같은 예산으로 더 많은 이미지를 봄.
3. 그리고 **적게 쓰는 것 자체가 논문의 주장**(ablation 5.2.3): native 구조는 데이터 스케일링 효율이 높아 더 적은 자원으로 인코더 기반 최상위와 대등 — 이해·생성이 같은 표현 공간을 써 서로의 학습을 돕기 때문(Fig 12 시너지).

### Q7. step 수(~51만)가 너무 적은 거 아냐? 다른 데는 몇 M step 학습하던데

배치가 어떻든 **수정 횟수 자체가 12만~50만 번이면 부족한 거 아니냐**는 의심 — 답은 "**step 수만 보면 착시**".

**1) "몇 M step 학습"하는 곳들의 진실: 배치가 작다**

| 모델 | step 수 | 배치 크기 | 총 학습량 (step×배치) |
|---|---|---|---|
| **DiT** (ImageNet 연구용) | **7M step** | 이미지 256장 | 18억 image-pass |
| SD 1.x | ~1M step | 2,048장 | ~20억 image-pass |
| **GPT-3** | **~9.4만 step** | 320만 token | 0.3T token |
| Llama 3 (405B) | ~100만 step | 최대 1,600만 token | 15.6T token |
| **SenseNova-U1 2단계** | 30만 step | ~200만 token | 0.5T+ token ≈ 수억~십억 image-pass |

**DiT가 700만 step인 건 배치가 고작 256장이라서**다. SenseNova-U1은 배치가 수천~1만 세트니까 30만 step으로 같은 총량을 본다:

> 7M step × 256장 = 18억 장 투입
> 0.3M step × 6천 장 = 18억 장 투입 ← **같은 공부량**

즉 "몇 M step"은 열심히 한 증거가 아니라 **한 입 크기가 작았다는 뜻**일 뿐.

**2) 진짜 비교 단위는 "총 token(또는 총 image-pass)"**

step은 "몇 번 씹었나", 배치는 "한 입 크기", 총량 = 둘의 곱. 영양(학습)은 총량이 결정한다. 극단적인 반례가 GPT-3 — **겨우 9.4만 step**으로 학습됐지만 아무도 "step이 적어서 덜 배웠다"고 하지 않는다. (총량 기준 평가는 → Q6 참조)

**3) 다만 step 수에 하한선은 존재**

아무리 배치를 키워도 step을 100번으로 줄일 수는 없다:
- 학습은 "고치고 → 고쳐진 상태에서 또 고치는" **순차적 과정**이라 연속 수정 횟수가 어느 정도 필요 (learning rate를 서서히 줄이는 스케줄, 커리큘럼도 step 위에서 굴러감).
- 이것이 critical batch size의 뒷면 — 배치를 무한정 키워 step을 무한정 줄이는 건 불가능.

하지만 그 하한선은 경험적으로 **수만 step 수준**이고, SenseNova-U1은 전 단계 합계 **약 51만 step**(120K+300K+84K+9K) = GPT-3의 5배가 넘는 수정 횟수라 하한선과 거리가 멀다.

| 착시 | 실상 |
|---|---|
| "다른 데는 7M step인데 여긴 0.3M" | 그쪽은 배치 256장, 여긴 수천~1만 세트 — **총량은 동급** |
| step 적음 = 덜 배움? | ❌ 배움의 양 = step × 배치. GPT-3는 9.4만 step |
| step 하한선은? | 있음(수만 수준) — 총 51만 step이라 여유 충분 |

한 줄: **"몇 step 했냐"가 아니라 "몇 token 봤냐"로 비교해야 하고, 그 기준으론 부족하지 않다.** 700만 step짜리 DiT는 밥을 오래 먹은 게 아니라 숟가락이 작았던 것.

### Q8. 그럼 적은 배치는 더 많은 step을 학습해야 하는 거구나?

정확하다. 그 관계를 한 줄로 쓰면:

> **총 학습량(token/이미지 수) = 배치 크기 × step 수**

총량이 정해져 있으면 배치와 step은 **시소 관계**:

| 전략 | 배치 | step | 예 |
|---|---|---|---|
| 큰 숟가락 | 크게 (6천~1만 세트) | 적게 (수십만) | SenseNova-U1, GPT-3, Llama 3 |
| 작은 숟가락 | 작게 (256장) | 많게 (수백만) | DiT, 연구실 규모 실험 |

밥 한 그릇(총량)은 같고, 숟가락이 작으면 떠먹는 횟수만 늘어나는 것.

**그럼 왜 어떤 곳은 일부러 작은 배치를 쓰나?**

1. **GPU가 몇 장 없어서** — 연구실은 8~64장으로 돌리니 물리적으로 큰 배치를 못 만든다. DiT의 배치 256이 바로 이 경우.
2. **작은 배치의 잡음이 가끔 약이 됨** — gradient가 조금 흔들리는 게 좁은 골짜기(sharp minima)를 탈출하는 데 도움이 된다는 관점도 있다. 소규모 학습에선 이 효과가 나쁘지 않음.
3. **step이 많으면 중간 관찰이 쉬움** — 연구 논문은 "학습 곡선"을 촘촘히 보여줘야 하니 step이 많은 쪽이 편하기도 하다.

**딱 하나만 더 기억하면 완성 — learning rate(학습 보폭)도 세트**

배치를 바꾸면 learning rate도 따라 바꿔야 한다:

- 배치 ↑ (수정 기회가 귀해짐) → 한 번에 크게 고쳐야 → **lr ↑**
- 배치 ↓ (자주 고침) → 조금씩 고쳐야 → **lr ↓**

이 균형(보통 배치 2배면 lr도 √2~2배)이 맞아야 두 전략이 같은 결과에 도달한다. 그래서 논문들이 배치 크기와 lr을 항상 세트로 보고하는 것 — SenseNova-U1도 SFT에서 "lr 2e-5"를 배치 1,440만 token과 세트로 공개했다.

### Q9. 텍스트 렌더링(이미지 안 글자 생성) 능력을 올리기 위해 어떻게 했어?

**데이터·구조·RL** 세 층에서 동시에 접근했다.

**축 1 — 데이터: 글자 데이터를 처음부터 크게 심음**

- T2I corpus에 **이중언어(영어+중국어) 텍스트 렌더링 데이터와 복잡한 인포그래픽**을 명시적으로 포함. Fig 8에서 English Text·Chinese Text가 People 카테고리 안에 별도 슬라이스로 존재하고, 포스터·차트 같은 글자 많은 하위 클래스를 롱테일로 보강.
- 인터리브 데이터의 **인포그래픽 29%**가 "글자 빽빽한 페이지 합성(text-rich page synthesis)에 밀도 높은 감독 신호"를 제공 — 글+그림 섞인 문서류를 대량으로 보여줘서 긴 글 렌더링을 훈련.

**축 2 — 구조: 픽셀 직접 예측이라 글자가 안 뭉개짐**

- MLP 헤드가 **VAE decoder를 거치지 않고 픽셀 패치를 직접 예측** — VAE 압축은 얇은 획, 작은 글자 같은 high-frequency(고주파) 디테일을 영구적으로 뭉개는 주범인데, 이 경로 자체가 없다.
- HiDream-O1이 "VAE 제거 → 텍스트 렌더링 강세"를 보인 것과 같은 원리([[paper_hidream_o1_image]]). 구조가 곧 텍스트 렌더링의 유리한 출발점.

**축 3 — RL: OCR로 채점하는 전용 강화학습 단계**

Flow-GRPO 후처리에 **텍스트 렌더링 전용 phase**를 따로 둠:

- **보상(reward) 설계**: 생성된 이미지에서 **PaddleOCR로 글자를 읽어내고**, 정답 텍스트와 비교해 **IoU(Intersection-over-Union) 기반 점수**를 매김. "사람이 보기 좋냐"가 아니라 "**기계가 읽어서 맞느냐**"로 채점.
- **규모**: 600 epoch, epoch당 프롬프트 48개 × 프롬프트당 이미지 16장(=768장), 10-step flow matching으로 빠르게 샘플링, lr 1e-5, KL 계수 β=0.01.
- **커리큘럼**: 처음 **200 epoch은 dynamic-resolution warmup** — 난이도 점수를 매긴 해상도 구성에서 **쉬운 것부터 시작해 점점 어려운 것**(작은 글자, 복잡한 배치)으로 올림.

**결과**

| 벤치마크 | SenseNova-U1 8B | 의의 |
|---|---|---|
| CVTG-2K 평균 | **0.940** | 오픈소스 최고 (A3B는 0.881) |
| LongText-Bench EN | **0.979** | 32B짜리 Emu3.5 추월 |
| LongText-Bench ZH | 0.962 | 중국어 장문도 최상위권 |

**한 줄 정리**: **"글자 데이터를 많이 먹이고(데이터), 글자가 뭉개질 통로를 없애고(픽셀 직접 예측), OCR이 읽어서 맞는지로 상벌을 주는(RL)"** 3단 결합. 특히 OCR-IoU 보상 + 해상도 커리큘럼의 RL 설계가 이 논문의 텍스트 렌더링 마무리 한 방이고, [[paper_firered_image_edit]]의 OCR 보상, [[paper_longcat_image]]의 한자 샘플링 전략과 같은 최근 정석 흐름.

### Q10. 다른 리뷰 논문 중 같은 계보는?

- **가장 가까운 형제 = [[paper_tuna_2]]** (인코더 0개 native UMM, Conv2d 주입, decoder-less). SenseNova-U1은 이 철학을 상용 규모로 완성한 버전.
- **[[paper_hidream_o1_image]]**: pixel-level 통합 + hybrid attention이 판박이(Q3 참조).
- 생성 전용이라 절반만 겹침: **[[paper_pixeldit]]**(픽셀 이중레벨), **[[paper_minit2i]]**(noise_scale=σ_R 발상 동일), **[[paper_pid]]**(sigma-aware 노이즈 처리), **[[paper_asymflow]]**(픽셀 flow matching).
- 후처리 정석 공유: DMD 증류는 [[paper_z_image]]·[[paper_qwen_image_2]]·[[paper_dreamlite]], Flow-GRPO는 [[paper_qwen_image_2_rl]]·[[paper_uniref_image_edit]] 라인.

---

## 🧷 한 줄 요약 (전체)

SenseNova-U1은 **NEO의 인코더·VAE 없는 native 철학**([[paper_tuna_2]]) + **HiDream-O1식 pixel-level 통합·hybrid attention**([[paper_hidream_o1_image]]) + **Z-Image/Qwen-Image-2.0식 DMD·GRPO 후처리**를 합쳐, **이해(understanding)와 생성(generation)을 MoT로 한 몸에 담은** SenseTime의 상용 규모 통합 멀티모달 모델(dense 16.4B / MoE 38.2B·활성 ~3B)이며, 이해 SOTA·생성 Qwen-Image급을 동시에 달성하고 VLA·World Model 확장까지 예비 제시한다.

---

## 🔗 관련 메모리 링크

- [[paper_tuna_2]] — 가장 가까운 형제 (인코더 0개 native UMM)
- [[paper_hidream_o1_image]] — pixel-level 통합 + hybrid attention (Q3 상세 비교)
- [[paper_pixeldit]] · [[paper_minit2i]] · [[paper_pid]] · [[paper_asymflow]] — VAE 없는 픽셀 생성 계보
- [[paper_z_image]] · [[paper_qwen_image_2]] — DMD 증류 라인
- [[paper_qwen_image_2_rl]] · [[paper_uniref_image_edit]] — GRPO RL 라인
- [[reference_pretrained_backbone_reuse_landscape]] — 백본 재사용 분기 분류
