# REPA: Representation Alignment for Generation

> **한 줄 요약**
> diffusion transformer(확산 트랜스포머)가 노이즈를 지우는 동안 내부에서 "이미지 의미(semantic)"를 어설프게 배우고 있다는 걸 진단하고, 그 중간 표현을 **이미 잘 배운 self-supervised(자기지도) 인코더 DINOv2의 깨끗한 이미지 표현과 cosine으로 끌어당기는 보조 손실(auxiliary loss)** 하나로 학습을 17.5배 가속한 논문.

---

## 📑 메타 정보

| 항목 | 내용 |
|---|---|
| **제목** | Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think |
| **저자** | Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie (KAIST · NYU · Korea Univ. 등) |
| **공개일** | 2024-10 (arXiv 2410.06940) · **ICLR 2025 Oral** |
| **분야** | Class-conditional image generation, Diffusion / Flow Transformer 학습 가속 |
| **논문 링크** | [abstract](https://arxiv.org/abs/2410.06940) · [PDF](https://arxiv.org/pdf/2410.06940) · [project](https://sihyun.me/REPA/) |
| **코드** | [github.com/sihyun-yu/REPA](https://github.com/sihyun-yu/REPA) |
| **외부 모델/데이터** | ImageNet-1K(256²), SD-VAE(잠재 인코딩), **DINOv2(표현 정렬 target, 동결)** |
| **PIERROT 구현** | 본 알고리즘의 PIERROT/PRX 코드 구현은 별도 문서 **[[ALG_REPA]]** 참조 (DINOv3 + 단일 MLP + crop 근사) |

---

## 📖 주요 용어 사전 (Glossary)

### 아키텍처
- **DiT / SiT (Diffusion / Scalable interpolant Transformer)** — 이미지를 잠재 공간(latent space)에서 만들어내는 Transformer 기반 생성 모델. SiT는 flow matching 계열. 이 논문이 가속하려는 baseline(기준 모델).
- **denoiser(디노이저)** — 노이즈 낀 입력을 받아 노이즈(또는 velocity)를 예측하는 본체 네트워크. 여기선 DiT/SiT.
- **projector / projection head(사영기)** — denoiser의 중간 hidden을 인코더 차원에 맞춰 쏘는 작은 MLP. REPA에서 새로 학습하는 유일한 추가 모듈(3층 SiLU).

### 핵심 개념
- **representation alignment(표현 정렬)** — denoiser 내부 중간 표현(hidden)을, 외부 사전학습 인코더의 표현과 닮도록 맞추는 것. 이 논문의 제목이자 방법.
- **self-supervised encoder(자기지도 인코더)** — 라벨 없이 이미지만으로 표현을 배운 vision 모델. 여기선 DINOv2. 학습 내내 동결(frozen).
- **patch token(패치 토큰)** — ViT가 이미지를 격자로 쪼개 patch마다 출력하는 feature 벡터. 정렬은 이 patch 단위로 patch끼리 짝지어 이뤄짐.
- **clean vs noisy(깨끗한 입력 vs 노이즈 입력)** — denoiser는 *노이즈 낀* 입력을 보지만, 정렬 target은 *깨끗한 원본* 이미지를 인코더에 넣은 표현이다. 이 비대칭이 핵심.

### 비교·진단 도구
- **CKNNA (Centered Kernel Nearest-Neighbor Alignment)** — 두 모델의 표현이 얼마나 닮았는지를 상호 최근접이웃(mutual k-NN) 기반으로 재는 정렬 지표. 디퓨전 모델의 의미 표현이 약함을 진단하는 데 사용.
- **linear probe accuracy(선형 분류 정확도)** — 표현을 고정한 채 선형 분류기만 얹어 분류 정확도를 재는 것. 인코더의 "표현 품질"을 나타내는 대리 지표.
- **cosine similarity(코사인 유사도)** — 두 벡터의 방향이 얼마나 같은지. 길이(scale)에 무관. REPA가 정렬에 쓰는 척도.

### 평가 지표
- **FID (Fréchet Inception Distance)** — 생성 이미지가 진짜 이미지 분포와 얼마나 가까운지. **낮을수록 좋음.**
- **CFG (Classifier-Free Guidance)** — 조건을 강하게 반영하도록 샘플링을 조정하는 표준 기법. **guidance interval**은 일부 타임스텝 구간에만 CFG를 거는 변형.

---

## 🎯 논문 요약 (TL;DR)

**한 줄:** diffusion transformer가 내부 의미 표현을 느리고 약하게 배운다는 걸 CKNNA로 진단하고, denoiser 중간 hidden을 동결된 DINOv2의 깨끗한 이미지 표현과 patch별 cosine으로 정렬하는 보조 손실(REPA) 하나로 **학습을 17.5배 가속**, SiT-XL/2에서 **FID 1.42(SOTA)** 를 달성했다.

- **핵심 문제:** diffusion 모델도 의미 표현을 배우긴 하지만, DINOv2 같은 자기지도 인코더에 한참 못 미치고(약한 정렬), 그 표현을 혼자 익히느라 학습이 느리다.
- **해결책:** 표현을 모델 혼자 배우게 두지 말고 **외부의 좋은 표현을 떠먹인다.** noisy 입력이 denoiser를 지나는 중간 layer(앞쪽 몇 블록)의 hidden을 projector로 쏴서, clean 이미지의 DINOv2 patch feature와 cosine 정렬하는 보조 손실을 더한다.
- **검증:** SiT-XL/2가 400K step만에 FID 7.9 — 이는 vanilla가 **7M step** 돌린 것보다 좋다(17.5배 가속). 최종은 CFG+guidance interval로 **FID 1.42**.
- **원리적 발견:** **인코더가 좋을수록(linear probe 높을수록) 생성 FID도 좋아진다** — 단순 트릭이 아니라 "이해 표현 = 생성 품질"이라는 상관.

---

## 🏆 핵심 기여 (Contributions)

1. **진단** — diffusion transformer가 내부에서 의미 표현을 배우긴 하나, 자기지도 인코더(DINOv2)에 비해 약하고 중간에서 정체됨을 CKNNA로 정량 측정. (2장)
2. **REPA 제안** — denoiser 중간 hidden을 projector로 사영해 동결 인코더의 clean 이미지 patch feature와 cosine 정렬하는 **단일 보조 손실**. 구조 변경·추론 오버헤드 없음. (3장)
3. **대규모 가속** — SiT/DiT 학습을 **17.5배 이상** 가속, 모델이 클수록 가속 효과가 더 커짐. (4장)
4. **인코더-생성 상관** — 정렬 인코더의 표현 품질(linear probe)이 높을수록 생성 FID가 좋아진다는 원리적 상관 + layer 깊이·정렬 강도 ablation. (4장)
5. **SOTA** — SiT-XL/2에서 FID 1.42(CFG+guidance interval) 달성.

---

## 🧠 주요 알고리즘 설명

### 1️⃣ 왜 시작했나 — "디퓨전 모델은 표현을 못 배운다"는 진단

> *왜 이 절을 두나: REPA의 모든 설계는 "diffusion 모델이 의미 표현을 어설프게 배운다"는 관찰에서 출발한다. 처방 전에 병을 먼저 본다.*

핵심 질문은 이것이다 — diffusion transformer가 노이즈를 지우는 동안, 내부적으로 "이 이미지가 무엇인가" 하는 **의미(semantic) 표현**을 제대로 배우고 있는가?

이를 재기 위해 **CKNNA**라는 정렬 지표(상호 최근접이웃 기반, mutual k-NN kernel alignment)로 denoiser의 중간 hidden과 잘 배운 자기지도 인코더 DINOv2의 feature가 얼마나 닮았는지 숫자로 잰다. 결과:

- diffusion 모델도 의미 표현을 **배우긴 한다** — 학습이 길어지고 모델이 커질수록 DINOv2와의 정렬도가 올라간다.
- 그러나 그 정렬이 **약하고 중간에서 정체(plateau)** 된다. DINOv2 수준에 한참 못 미친다.
- 즉 diffusion 모델은 그림 그리는 법을 배우느라 정작 그림을 이해하는 표현을 **느리고 어설프게** 익히고 있었다.

→ 통찰: **표현 학습을 모델 혼자 시키지 말고, 이미 잘 배운 외부 인코더의 표현을 떠먹이자.**

### 2️⃣ 가르치는 선배 비유 (직관)

> *왜 이 절을 두나: 수식 전에 한 문장으로 그림을 잡아두면 이후 식·코드가 쉽게 읽힌다.*

신입(denoiser)이 그림 그리는 법을 맨바닥부터 익히려면 오래 걸린다. 근처에 **이미 그림을 잘 보는 선배(DINOv2)** 가 있다. 선배는 그림을 그리진 않지만 "이 patch가 저 patch와 비슷하다"는 의미는 잘 안다. 신입이 그림을 그리는 도중 **중간 단계의 캔버스**를 보고, 선배가 보는 "이 자리 patch의 의미 벡터"와 닮도록 cosine으로 끌어당긴다.

핵심 두 가지:
1. **frozen teacher(동결 선배)** — DINOv2 가중치는 절대 안 건드린다. 신입(denoiser)과 projector만 학습.
2. **중간 layer 정렬** — 마지막이 아니라 앞쪽 중간 layer에 신호를 건다(이유는 4장 ablation).

### 3️⃣ REPA 손실 — 식과 동작

> *왜 이 절을 두나: 방법 자체는 손실 한 줄이다. 입력의 비대칭과 patch 정렬만 이해하면 끝난다.*

한 step의 흐름:

1. **noisy 입력 통과** — 노이즈 낀 입력이 denoiser를 지날 때, 특정 중간 블록의 hidden을 꺼낸다(patch별 벡터들).
2. **projector 사영** — 학습 가능한 작은 MLP(3층 SiLU)로 denoiser hidden을 인코더 차원에 맞춰 쏜다.
3. **clean target 추출** — **깨끗한 원본 이미지**를 동결 DINOv2에 넣어 patch feature를 얻는다. 이게 정답(target).
4. **patch별 cosine 정렬** — 사영 결과와 target을 patch마다 cosine similarity로 최대화. 손실은 그 평균에 음수를 붙인 것.
5. **합산** — denoising 손실(예: flow matching/MSE)에 가중치 람다(기본 0.5)로 더해 함께 backprop. **DINOv2는 동결**이라 갱신 안 됨.

말로 풀면: *"노이즈 입력의 중간 표현을, 깨끗한 이미지의 DINOv2 표현과 같은 방향이 되도록 patch마다 끌어당긴다."*

설계에서 미묘하지만 중요한 두 가지:
- **입력의 비대칭** — denoiser는 noisy 입력을 보지만 target은 clean 원본의 표현이다. "노이즈 속에서도 의미를 뽑아내라"는 압력이 걸린다.
- **patch 단위 정렬** — 이미지 한 벡터가 아니라 patch 격자끼리 맞춘다. 공간적 의미 구조까지 전달.

> 사영기 파라미터 수는 본체 대비 매우 작아(수 M 규모) 메모리·속도 부담은 미미하다. 인코더 forward는 동결+no_grad라 비용이 작다. 구현 세부(hook 등록, token 수 불일치 처리, TREAD 결합)는 [[ALG_REPA]] 참조.

### 4️⃣ Ablation — 논문에서 가장 값진 부분

> *왜 이 절을 두나: "어디에·무엇으로 정렬하느냐"가 효과를 좌우한다. 이 답들이 실전 하이퍼파라미터를 정한다.*

**(A) 어느 layer에 정렬을 거나** — *앞쪽 몇 개 블록*만 정렬해도 충분하다(SiT-XL 기준 8번째 블록 근처). 앞단에서 정렬을 끝내주면 **뒤쪽 layer는 high-frequency(고주파, 질감·디테일)에 집중**할 여유가 생긴다. 너무 깊은 곳에 걸면 오히려 denoising task를 해친다.

**(B) 어떤 인코더가 좋은가** — MoCov3, CLIP, DINOv1/v2, MAE, I-JEPA를 비교 → **DINOv2가 최고**. 그리고 결정적 상관:

> 인코더의 **linear probe 정확도(표현 품질)** 가 높을수록 → 그걸로 정렬한 diffusion 모델의 **생성 FID가 좋아진다.**

즉 "좋은 이해 표현 = 좋은 생성"이 거의 선형으로 연결된다. REPA가 단순 트릭이 아니라 원리적임을 보여주는 증거다. (PIERROT가 DINOv2 대신 더 최신 DINOv3를 쓰는 이유도 이 논리의 연장선 — 더 좋은 인코더면 더 좋아야 하니까. [[ALG_REPA]])

**(C) 부수 효과** — REPA를 쓰면 diffusion 모델 *자신의* linear probe 정확도도 올라간다. 생성 모델인데 표현 학습 능력까지 좋아진다.

---

## 📊 실험 요약

> *왜 이 절을 두나: "17.5배 가속"이 어떤 비교에서 나온 수치인지 정확히 봐야 다른 문서의 "30% 단축" 같은 표현과 혼동하지 않는다.*

ImageNet 256×256, SiT-XL/2 기준.

| 설정 | 결과 | 의미 |
|---|---|---|
| 수렴 가속 | **17.5배 이상** | vanilla가 같은 FID에 도달하는 step 대비 |
| REPA 400K step (CFG 없음) | **FID 7.9** | vanilla SiT를 **7M step** 돌린 것보다 좋음 |
| 최종 (CFG + guidance interval) | **FID 1.42** | 당시 SOTA |
| 모델 크기 효과 | 클수록 가속 ↑ | 큰 모델일수록 혼자 표현 배우기가 비효율 → 떠먹이는 이득 ↑ |

> ⚠️ **"17.5배"의 정확한 의미** — 이는 "vanilla가 REPA의 어떤 FID에 도달하는 데 걸리는 step" 대비의 극단 비교다. PIERROT 24h 레시피 문서([[ALG_REPA]])에서 말하는 "수렴 ~30% 단축"은 TREAD 등 다른 가속과 섞인 현실적 체감치라 수치가 다르다. 같은 현상의 다른 측정 기준.

---

## 💬 Q&A

> *왜 이 절을 두나: 본문에서 한 번씩 짚은 핵심을 독립 질문으로 모아 빠르게 참조하기 위함.*

### Q1. 왜 MSE가 아니라 cosine으로 정렬하나?

cosine은 벡터의 **방향만** 맞추고 길이(scale)에 무관하다. MSE는 절대 크기·노름까지 일치시켜야 해서 projector가 scale까지 학습하는 부담이 커진다. REPA는 "의미 방향"만 잡으면 되므로 cosine이 더 안정적이고 학습이 쉽다.

### Q2. 왜 마지막 layer가 아니라 앞쪽 중간 layer인가?

- 첫 layer: low-level(edge·color)이라 의미 표현과 mismatch.
- 앞쪽 중간 layer: mid-level semantic이라 DINOv2 patch feature와 정합도가 가장 높다.
- 마지막 layer: denoising에 특화된 출력이라 의미 정렬을 걸면 task가 손상된다.

앞단에서 정렬을 끝내면 뒤쪽은 디테일(고주파)에 집중할 수 있다는 분업 효과도 있다. (4장-A)

### Q3. DINOv2를 굳이 동결하는 이유는?

DINOv2는 이미 라벨 없이 방대한 이미지로 좋은 표현을 배워둔 "선배"다. 학습하면 그 좋은 표현이 망가질 위험 + 비용 증가. REPA의 목적은 그 표현을 **떠먹는 것**이므로 target은 고정해야 신호가 일관된다. (후속 연구 REPA-E는 인코더까지 함께 학습하는 변형이다.)

### Q4. 추론(생성)할 때도 비용이 늘어나나?

아니다. REPA는 **학습 때만 쓰는 보조 손실**이다. projector와 DINOv2는 추론에 관여하지 않으므로 생성 속도·구조는 baseline과 동일하다. 순수하게 학습 가속용.

### Q5. 토큰 수가 다르면(denoiser token ≠ DINO patch) 어떻게 맞추나?

원논문/공식 구현은 **입력 이미지를 denoiser token 격자에 맞춰 resize**해 patch 위치를 정확히 일치시킨다. PIERROT는 짧은 쪽으로 그냥 crop하는 단순 근사를 쓴다(속도↑, 정렬 정확도 약간↓). 자세한 구현 비교는 [[ALG_REPA]] §3 참조.

---

## 🧭 후속·계보

> *왜 이 절을 두나: REPA가 이후 어떻게 확장됐는지 알면 관련 논문 분석 시 위치를 잡기 쉽다.*

- **iREPA (arXiv 2512.10794)** — projector를 MLP에서 Conv2d / Attention(+RoPE)로 바꾸고, 인코더 feature에 spatial normalization을 더해 공간 정렬을 정교화한 개선판.
- **REPA-E** — 동결 대신 인코더까지 end-to-end로 함께 학습하는 변형.
- **REPA를 채택한 모델들** — DDT([[paper_ddt]])는 REPA를 인코더에만 건다. PRX/PIERROT는 24h 레시피의 필수 가속 축으로 채택([[ALG_REPA]], [[paper_prx]]).

---

## ✅ 한 줄 요약 (전체)

> **REPA = "diffusion 모델은 의미 표현을 어설프게 배운다(CKNNA로 진단) → noisy 입력의 중간 hidden을 clean 이미지의 동결 DINOv2 patch feature와 patch별 cosine으로 정렬하는 보조 손실 하나를 추가 → 학습 17.5배 가속, FID 1.42, 게다가 인코더가 좋을수록 생성도 좋아진다"** 를 보인 ICLR 2025 Oral 논문. 구조 변경·추론 오버헤드 없이 손실 한 줄로 학습 효율을 바꾼 대표적 표현-정렬 기법.

---

## 🔗 관련 메모리·문서 링크

- **[[ALG_REPA]]** — 본 알고리즘의 PIERROT/PRX 코드 구현 (DINOv3 + 단일 MLP + crop + TREAD gather, N=4 미니 예제)
- **[[paper_ddt]]** — REPA를 인코더에만 적용한 Decoupled Diffusion Transformer
- **[[paper_prx]]** — REPA를 24h 레시피 필수 축으로 채택한 픽셀 학습 레시피
