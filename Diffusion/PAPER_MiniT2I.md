# MiniT2I: A Minimalist Baseline for Text-to-Image Generation

## 📋 메타 정보

| 항목 | 내용 |
|---|---|
| **제목** | MiniT2I: A Minimalist Baseline for Text-to-Image Generation |
| **저자** | Xianbang Wang(프로젝트 리드), Hanhong Zhao, Yiyang Lu, Kangyang Zhou, Linrui Ma, **Kaiming He**(지도) |
| **소속** | MIT (학부 UROP + Kaiming He 연구실) |
| **공개일** | 2026-06-16 (블로그 포스트, 논문 아님) |
| **분야** | text-to-image(텍스트→이미지 생성), pixel-space diffusion(픽셀 공간 확산), flow matching(흐름 정합) |
| **블로그** | [peppaking8.github.io](https://peppaking8.github.io/#/post/minit2i) |
| **코드** | [PyTorch: Hope7Happiness/minit2i-torch](https://github.com/Hope7Happiness/minit2i-torch) · [JAX(원본): PeppaKing8/minit2i-jax](https://github.com/PeppaKing8/minit2i-jax) |
| **체크포인트** | [HuggingFace: MiniT2I/MiniT2I](https://huggingface.co/MiniT2I/MiniT2I) (B/16, L/16) |
| **외부 모델/데이터** | FLAN-T5-Large(341M, 동결 텍스트 인코더) · CC12M LLaVA 재캡션(사전학습) · BLIP3o-60K + DALL·E3 + ShareGPT-4o-Image(120K 정렬셋) |
| **계보** | JiT(Li & He 2025, 픽셀+x예측+시간조건 제거) → Mean Flow(Geng et al., few-step) → **MiniT2I**(이 둘을 T2I로 통합) |

> ⚠️ 이 문서는 **PyTorch 재현 코드(diffusers/mmdit.py 등)를 직접 읽고 블로그와 교차 검증**해 작성했다. 코드의 일부 사실(특히 timestep 조건이 네트워크에서 실제로는 버려지는 점)은 블로그 본문보다 더 과격하다.

---

## 📖 주요 용어 사전 (Glossary)

### 아키텍처
- **pixel-space diffusion(픽셀 공간 확산)**: 이미지를 VAE로 압축하지 않고 원본 RGB 픽셀(raw RGB pixel) 그대로 확산을 돌리는 방식. MiniT2I는 VAE를 **아예 안 쓴다**.
- **MM-DiT(멀티모달 DiT)**: SD3가 제시한 표준 T2I 백본. 이미지 토큰과 텍스트 토큰이 각자 가중치(double-stream, 이중 스트림)를 갖되 joint attention(공동 어텐션)에서 섞인다.
- **MM-JiT (이 논문의 백본)**: MM-DiT에서 **AdaLN 조건 분기를 통째로 삭제**하고, 텍스트 어댑터(text adapter) 블록 2개를 앞에 붙인 변형. "JiT(He 그룹) 철학을 MM-DiT에 적용"한 형태.
- **double-stream joint attention(이중 스트림 공동 어텐션)**: 이미지/텍스트가 따로 QKV를 만들되, attention 계산은 두 토큰열을 이어붙여(concat) 함께 수행. 텍스트→이미지 정보 전달의 통로.
- **AdaLN (Adaptive LayerNorm, 적응 정규화)**: timestep·pooled text 같은 전역 조건으로 각 블록 정규화의 scale·shift·gate를 만들어 주입하는 표준 DiT 조건 방식. MiniT2I는 이걸 **제거**한다(핵심 기여).
- **BottleneckPatchEmbed(병목 패치 임베딩)**: 패치를 Conv로 `pca_channels=128`까지 압축 → 1×1 Conv로 hidden 차원 확장하는 2단 임베딩. JiT의 "PCA 병목" 트릭으로, 고차원 픽셀 패치를 안정적으로 토큰화.
- **text adapter(텍스트 어댑터)**: joint attention 전에 동결 T5 토큰을 다듬는 텍스트 전용 Transformer 블록 2개(`PlainTextTransformerBlock`). T5는 픽셀 denoiser용으로 학습된 게 아니라서 값싼 보정이 필요.
- **mask_token(마스크 토큰)**: 패딩 위치와 CFG 무조건부(uncond) 드롭 위치를 **동시에** 대체하는 학습된 토큰. 패딩 처리와 CFG가 한 메커니즘으로 통합됨.

### 핵심 개념
- **flow matching(흐름 정합)**: 노이즈에서 이미지로 가는 직선 경로 `z_t = t·x + (1−t)·ε` 위에서 velocity(속도)를 맞추도록 학습. t=1이 깨끗한 이미지, t=0이 노이즈.
- **x-prediction(깨끗한 이미지 직접 예측)**: 네트워크가 velocity·noise가 아니라 **clean image x0를 직접 출력**. 고차원 픽셀 공간에서 유일하게 안정적(아래 §2 ablation). MiniT2I는 x-예측 + v-loss 조합.
- **noise_scale = 2(노이즈 2배 증폭)**: 노이즈를 표준편차 2배로 키워 섞는 트릭. 입력 크기로 노이즈 레벨을 읽을 수 있게 만드는 숨은 장치(아래 Q3).
- **CFG (Classifier-Free Guidance, 분류기 없는 가이던스)**: `v_cfg = v_u + s·(v_c − v_u)`. 조건/무조건 예측을 외삽해 프롬프트 추종을 강화. ImageNet은 s≈2~3, T2I는 s≈6 필요.
- **logit-normal timestep 샘플링**: t를 정규분포→sigmoid로 뽑되 `mu=−0.8`로 치우치게 해, 노이즈 많은 구간을 더 학습.
- **pretrain → alignment fine-tune(사전학습 → 정렬 미세조정)**: LLM의 pretrain→SFT와 같은 구도. 사전학습=커버리지, 미세조정=프롬프트 추종 학습. 둘의 분업이 MiniT2I의 가장 실용적 교훈(§4).

### 비교/확장
- **JiT (Li & He 2025)**: "Back to Basics: Let Denoising Generative Models Denoise". 픽셀+x예측+시간조건 최소화의 원류. MiniT2I의 직계 부모.
- **Mean Flow(평균 속도 흐름)**: few-step 생성용 distillation(증류). MiniT2I-B/16을 4-step으로 압축, 처리량 ~50배(§7).
- **LoRA(저랭크 적응)**: 어댑터만 학습하는 경량 미세조정. 작은 모델인데도 스타일 전이가 잘 됨(§7).

### 평가 지표
- **MSCOCO-30K FID(낮을수록 좋음)**: 분포 수준 사실감. **프롬프트 추종은 측정 못 함**(이 한계가 §4의 반전).
- **GenEval(높을수록 좋음)**: 객체 존재·개수·색·위치 추종.
- **DPG-Bench(높을수록 좋음)**: 긴 프롬프트 정렬.
- **PRISM-Bench**: 텍스트 렌더링·스타일·상상력·고유명사 등 넓은 스트레스 테스트. 저자들이 "진짜 잣대"로 삼는 벤치(§6).

---

## 🎯 논문 요약 (TL;DR)

**한 줄**: "사전학습된 언어모델의 텍스트 토큰을 그냥 in-context 조건으로 attention에 같이 넣어주면, T2I는 ImageNet 클래스 조건 생성과 거의 다를 게 없다" — VAE·캐스케이드·RL·보조손실을 전부 빼고 픽셀 공간 flow matching만으로 8×H100 3일(ImageNet급 컴퓨팅)에 GenEval 0.87~0.88을 달성한 미니멀 baseline.

**핵심 문제**: T2I는 거대 인프라·복잡한 파이프라인이 필요한 "닫힌 산업 기술"로 여겨진다. 이 진입장벽이 학계의 실험을 막는다. 정말 ImageNet 수준의 자원과 머릿속에 담길 만큼 짧은 레시피로 T2I가 가능한가?

**해결책**: 표준 SD3 MM-DiT에서 **제거 가능한 모든 것을 제거**한다. ① 잠재공간 → 픽셀(VAE 삭제), ② x-예측으로 픽셀 학습 안정화, ③ AdaLN 조건 분기 삭제(=MM-JiT), ④ 저품질 웹 캡션 → VLM 재캡션 데이터, ⑤ 사전학습(커버리지) + 짧은 정렬 미세조정(추종)의 2단 분업.

**검증**: B/16(258M)·L/16(912M)이 더 큰 1B+ 픽셀 모델들(DeCo-XXL, PixelDiT, PixNerd)을 GenEval/DPG에서 추월. 단, 저자 스스로 GenEval/DPG가 정렬셋에 흔들린다고 인정하고 PRISM-Bench에서는 텍스트 렌더링·고유명사가 여전히 약하다고 솔직히 공개.

---

## 🔑 핵심 기여 (Contributions)

1. **T2I ≈ in-context 조건 ImageNet 생성이라는 관점**: 텍스트를 joint attention의 토큰으로만 넣으면 구조·컴퓨팅·데이터량이 클래스 조건 생성과 놀랄 만큼 닮았음을 실증.
2. **MM-JiT 백본**: MM-DiT에서 AdaLN 조건 분기를 제거하고 텍스트 어댑터 2블록을 추가 → 구조가 **더 단순해지면서 FID도 개선**(17.4→13.7).
3. **픽셀 공간 안정화 레시피**: x-예측이 필수임을 9가지 조합 ablation으로 증명(ε/v 직접 출력은 붕괴).
4. **2단 데이터 분업의 명확한 검증**: 사전학습=커버리지, 120K 정렬셋=추종. 어느 한쪽도 단독으론 둘 다 못 함.
5. **스케일 안정성 + 확장성**: 같은 레시피가 B/32→B/16→L/16에서 안정 학습. LoRA·Mean Flow 4-step 증류 같은 후속 트릭도 그대로 작동.
6. **완전 공개 + 정직한 한계 공개**: 공개 데이터만 사용. PRISM에서 텍스트/엔티티 약점을 숨기지 않음.

---

## 🧩 주요 알고리즘 설명

![MiniT2I 샘플 (B/16)](figures/minit2i_fig1.png)

### 1️⃣ 전체 파이프라인 — 픽셀 + 동결 T5 + flow matching

*VAE·복잡한 조건 모듈을 다 빼도 T2I가 되는지 보려면, 가장 단순한 한 줄짜리 데이터 흐름부터 세워야 한다.*

```
프롬프트 → [동결 FLAN-T5-L] → 텍스트 토큰(256개, 1024차원)
                                      ↓ (텍스트 어댑터 2블록으로 다듬기)
노이즈 이미지 z_t → [BottleneckPatchEmbed] → 이미지 토큰
                                      ↓
       [DoubleStreamDiTBlock × 17] (이미지·텍스트 joint attention)
                                      ↓
              [FinalLayer] → unpatchify → 예측한 깨끗한 이미지 x0
```

- 입력 해상도 512×512. B/32는 패치 32(토큰 256), B/16·L/16은 패치 16(토큰 1024).
- 텍스트 인코더는 **동결**. T5-XXL·Gemma2-2B·Qwen3-VL-2B로 키워봤지만 컴퓨팅만 늘고 점수는 거의 안 올라 T5-L(341M) 유지.
- 이미지 토큰에 **고정 2D sincos 위치 임베딩**을 더하고, **추가로 attention에서 2D RoPE**까지 적용(절대+상대 위치 둘 다).

### 2️⃣ x-prediction + v-loss — 픽셀 공간에서 유일하게 안 터지는 조합

*고차원 픽셀에서 노이즈(ε)나 속도(v)를 직접 맞히려 하면 학습이 붕괴하므로, 깨끗한 이미지(x0)를 직접 예측하게 한다.*

학습 손실 ([diffusion.py](https://github.com/Hope7Happiness/minit2i-torch/blob/main/mini_t2i/diffusion.py) `training_loss`):

```python
noise = randn_like(images) * noise_scale          # noise_scale = 2.0 (노이즈 2배 증폭)
x_t = images * t + noise * (1 - t)                # 진짜 이미지와 노이즈를 t 비율로 섞음
pred_x0 = model(x_t, t, text, mask)               # 네트워크는 '깨끗한 이미지'를 직접 출력
target = (images - x_t) / (1 - t).clamp_min(0.05) # 진짜 velocity
v_pred = (pred_x0 - x_t) / (1 - t).clamp_min(0.05)# 예측 x0를 velocity로 환산
loss = (v_pred - target)² .mean()                 # v-loss
```

즉 **출력은 x0(깨끗한 이미지), 손실은 v(속도) 좌표**. 수학적으로는 x-MSE에 `1/(1−t)²` 가중치를 준 것과 같아 저노이즈(t→1) 구간을 강조하고, `clamp_min(0.05)`이 가중치 폭주(최대 400배)를 막는다.

**9가지 조합 ablation (B/32, CC12M 250K step 후 MSCOCO FID↓):**

| 손실 \ 출력 | x-pred | ε-pred | v-pred |
|---|---|---|---|
| x-loss | 15.3 | 523.8 | 229.1 |
| ε-loss | 15.2 | 524.8 | 231.4 |
| **v-loss** | **13.7** | 524.0 | 230.1 |

→ **출력을 ε나 v로 두면 무조건 붕괴(FID 200~500), x-출력만 안정.** 잠재공간 직관과 정반대인 JiT의 핵심 발견을 재현. (왜 시간 임베딩 없이도 되는지는 → Q3)

### 3️⃣ MM-JiT 블록 — AdaLN을 통째로 삭제

*작은 모델에서는 프롬프트를 joint attention으로 한 번만 넣어도 충분하고, AdaLN으로 한 번 더 넣는 건 군더더기였다.*

MM-DiT는 프롬프트를 **두 번** 주입한다: ① 이미지·텍스트 joint attention, ② pooled text+timestep을 AdaLN scale/shift/gate로. MM-JiT는 ②를 통째로 버린다.

`DoubleStreamDiTBlock` ([mmdit.py](https://github.com/Hope7Happiness/minit2i-torch/blob/main/diffusers/mmdit.py)):

```python
def forward(self, x, txt, vec):              # vec를 받긴 받지만…
    x_norm   = self.img_norm1(x)             # 평범한 RMSNorm (AdaLN 아님)
    txt_norm = self.txt_norm1(txt)
    # … 이미지·텍스트 QKV를 concat해 joint attention …
    q = self.rope(cat([q_t, q_i]), txt_len=lt)   # 텍스트=1D RoPE, 이미지=2D RoPE
    # … attention 후 각 스트림으로 분리 …
    x   = x   + self.img_attn_proj(out_img)
    txt = txt + self.txt_attn_proj(out_txt)
    x   = x   + self.img_mlp(self.img_norm2(x))   # SwiGLU MLP
    txt = txt + self.txt_mlp(self.txt_norm2(txt))
    return x, txt
    # ← vec는 본문 어디에도 안 쓰임 (Q1~Q3에서 상술)
```

**ablation (B/32, FID↓):**

| 단계 | Layers | Params | GFLOPs | FID↓ |
|---|---|---|---|---|
| MM-DiT 픽셀 | 12 | 258M | 265 | 18.7 |
| + 텍스트 어댑터 2블록 | 12 | 276M | 273 | 17.4 |
| **− AdaLN (= MM-JiT)** | 17 | 260M | 313 | **13.7** |

AdaLN 파라미터를 깊이(12→17층)로 재배치한 셈. 구조가 단순해지며 더 잘 학습.

### 4️⃣ CFG와 샘플링 — mask로 무조건부를 만든다

*프롬프트 추종을 강하게 하려면 CFG가 필요한데, 별도 uncond 임베딩 없이 mask만 0으로 만들어 처리한다.*

```python
# context = where(mask>0.5, T5출력, mask_token)  ← 패딩·드롭 위치를 학습된 mask_token으로
# uncond = mask 전체를 0 → 전부 mask_token = 무조건부 예측
v_cfg = v_uncond + (v_cond - v_uncond) * cfg_scale   # velocity에 CFG 적용
x = x + (t_next - t_cur) * v_cfg                      # 100-step Euler 적분
```

- 학습 시 `label_drop_rate=0.1`로 mask를 0으로 떨궈 무조건부 분기를 학습.
- 샘플링: 시작 노이즈도 `randn * 2`(noise_scale 일관성), n_T=100 Euler, CFG는 B/16 일반 2.5 / L/16 6.0 / 벤치 5.0.

### 5️⃣ 2단 학습 — 사전학습(커버리지) + 정렬 미세조정(추종)

*사전학습만 하면 FID는 좋아도 프롬프트를 안 따르는 "흐릿하고 평균적인" 그림이 나온다. 이 문제는 구조가 아니라 데이터로 푼다.*

| 단계 | 데이터 | step | 역할 |
|---|---|---|---|
| 사전학습 | CC12M LLaVA 재캡션 | 250K | 세상의 폭넓은 모습(커버리지) |
| 정렬 미세조정 | 120K 합성 고품질(BLIP3o-60K + DALL·E3 19K + ShareGPT-4o 41K) | 40K | "프롬프트에 맞는 좋은 답"의 모습 |

**분업 증명 ablation:**

| 사전학습 | 미세조정 | 최종 GenEval↑ | 최종 DPG↑ |
|---|---|---|---|
| CC12M | **120K mix** | **0.826** | **82.3** |
| CC12M | CC12M | 0.529 | 78.7 |
| 120K mix | 120K mix | 0.408 | 49.2 |
| CC12M+120K(1:1) | 120K mix | 0.587 | 60.2 |

→ 사전학습=커버리지, 미세조정=정렬. 섞으면 둘 다 희석. 두 단계 모두 "무릎(knee)"에서 포화(250K / 30~40K).

> 데이터 관련 디테일: 저품질 웹 캡션(raw CC12M)은 학습을 크게 늦춰 VLM 재캡션을 씀. 재캡션은 평균 ~100토큰(원래 <20)으로 길어져 **256토큰 예산**(프롬프트의 ~99% 커버)을 채택.

---

## 📊 실험 요약

### 스케일업 (레시피 고정, 패치/크기만 변경)

*B/32에서 정한 레시피가 더 큰 설정에서도 그대로 버티는지 확인 — 미니멀 레시피의 신뢰성 검증.*

| 모델 | 이미지 토큰 | Params | GFLOPs | FID↓ | GenEval↑ | DPG↑ |
|---|---|---|---|---|---|---|
| MiniT2I-B/32 | 256 | 260M | 313 | 13.69 | 0.826 | 82.3 |
| MiniT2I-B/16 | 1024 | 258M | 570 | 10.51 | 0.873 | 84.2 |
| **MiniT2I-L/16** | 1024 | 912M | 1493 | **8.99** | **0.883** | **85.9** |

### 픽셀 공간 T2I 동급 비교 (512 해상도)

*데이터·인코더가 제각각이라 공정 비교가 어려우므로, 픽셀 공간 512로 범위를 좁혀 비교.*

| 모델 | Params(생성+텍스트) | GenEval↑ | DPG↑ |
|---|---|---|---|
| PixelFlow-XL/4 | 882+800M | 0.60 | 77.9 |
| PixNerd-XXL/16 | 1.2+1.7B | 0.73 | 80.9 |
| PixelDiT-T2I | 1.3+2.6B | 0.78 | 83.7 |
| DeCo-XXL/16 | 1.1+1.7B | 0.86 | 81.4 |
| **MiniT2I-B/16** | **258+341M** | **0.87** | **84.2** |
| **MiniT2I-L/16** | 912+341M | **0.88** | **85.9** |

→ 훨씬 작은 파라미터로 1B+ 모델들을 추월. 컴퓨팅도 JiT-L/32 ImageNet 200ep과 동급(0.139 ZFLOPs). B/32는 8×H100 **3일**.

### PRISM-Bench — "진짜 실력" (저자들이 신뢰하는 잣대)

*GenEval/DPG는 작은 정렬셋에 흔들려 더 이상 중립 심판이 아니므로, 넓은 스트레스 테스트로 약점을 드러낸다.*

| 모델 | Avg | Entity(고유명사) | **Text(렌더링)** | Imag.(상상력) | Style |
|---|---|---|---|---|---|
| SD3-Medium | 66.1 | 66.3 | 50.9 | 51.0 | 77.0 |
| FLUX.1-dev | 68.5 | 66.2 | 62.6 | 54.2 | 73.4 |
| Qwen-Image | 74.1 | 72.0 | 64.5 | 56.5 | 85.5 |
| MiniT2I-B/16 | 55.8 | 47.1 | 22.4 | 56.1 | 73.4 |
| MiniT2I-L/16 | 62.4 | 60.3 | **30.6** | 57.9 | 79.9 |

→ **텍스트 렌더링(30.6 vs FLUX 62)과 고유명사가 명확히 약함** = 공개 데이터 레시피의 한계. 반대로 **상상력·스타일은 SD3-Medium(~2B)에 필적/우위**.

### 확장 — LoRA & Mean Flow 증류

| 항목 | 설정 | 결과 |
|---|---|---|
| LoRA | B/16, rank-32, Naruto/Pokemon ~1K장, 400 step | 스타일 전이 성공 (작은 모델도 적응 OK) |
| Mean Flow 증류 | B/16 → 4-step, 50K step | 처리량 2.6→128 img/s (**~50배**), GenEval 0.874→0.842 유지 |

---

## 💬 Q&A 섹션

### Q1. `out = self.final_layer(combined, vec)`에서 vec를 넘기는데, 왜 "안 쓴다"고 하나?

핵심은 **"인자로 넘기는 것(pass)"과 "함수 본문에서 실제로 쓰는 것(use)"의 차이**다.

`final_layer(combined, vec)`는 vec를 **전달**한다. 맞다. 하지만 **받는 쪽이 무시**한다:

```python
class FinalLayer(nn.Module):
    def forward(self, x, vec=None):              # ← vec를 받긴 받는다
        return self.linear(self.norm_final(x))   # ← 본문에 vec가 한 번도 안 나온다
```

`DoubleStreamDiTBlock.forward(self, x, txt, vec)`도 동일 — vec가 시그니처엔 있지만 본문 어디에도 등장하지 않는다. 참고로 파일 맨 위에 `modulate(x, shift, scale)` 함수까지 정의돼 있지만 **아무도 호출하지 않는다**(AdaLN을 썼다면 여기서 vec→shift/scale로 변조됐어야 함).

| | 전달(시그니처) | 사용(본문 연산) |
|---|---|---|
| `final_layer`, `DoubleStreamDiTBlock` | O | **X (받고 버림)** |

→ "넘기긴 하는데 죽은 인자"이고, 결과적으로 **timestep 조건이 실제 연산에 영향을 못 준다.**

### Q2. vec(=AdaLN 조건)를 왜 제거했고, 그 vec의 의미는 무엇인가?

**왜 제거했나** — AdaLN 삭제가 곧 MM-JiT의 핵심 기여다(§3 ablation: 17.4→13.7). 작은 모델에선 프롬프트를 joint attention으로 한 번 넣는 걸로 충분했고, AdaLN은 군더더기였다. AdaLN을 지우니 vec를 소비할 곳이 사라진 것.

**왜 시그니처엔 남겼나** — ① MM-DiT 블록(vec 사용)과 호출부를 통일해 **drop-in 교체**로 ablation하려고, ② SD3 코드에서 파생되며 남은 흔적, ③ 체크포인트 키 호환.

**vec의 의미** — vec는 원래 **"전역 조건 벡터(global conditioning vector)"**다:

```python
t_vec = t_embedder(t)                          # ① 시간(노이즈 레벨) 임베딩
pooled_text = context.mean(dim=1)              # ② 텍스트 토큰 평균(문장 줄거리)
vec = t_vec + pooled_embedder(pooled_text)     # ① + ②
```

즉 "지금 노이즈가 얼마인지 + 만들 그림의 대략적 주제"를 한 벡터로 합쳐 모든 블록에 broadcast하려던 신호. 그런데 MiniT2I는 이 두 정보를 각각 **다른 경로로** 흘려보내며 vec를 무용지물로 만든다:

| 정보 | 원래 경로(vec/AdaLN) | MiniT2I의 실제 경로 |
|---|---|---|
| ① 시간 t | AdaLN 변조 | **입력 이미지 크기**로 암묵 전달(→ Q3) |
| ② 텍스트 | pooled AdaLN + attention | **joint attention** 하나로 통합(토큰별 직접) |

→ vec의 현재 "의미"는 기능이 아니라 **"MM-DiT에서 출발했다는 화석 + 호환용 빈 슬롯"**. vec·t_embedder·pooled_embedder·modulate를 전부 지워도 모델 동작은 동일하다.

### Q3. 명시적 시간 임베딩이 없는데, 노이즈 레벨 신호를 어떻게 받나?

**별도 신호로 안 받고, 입력 이미지 자체에서 읽어낸다.** 노이즈 레벨이 입력의 **크기**와 **통계 구조**에 이미 새겨져 있다.

**(1) 입력 크기 = 노이즈 레벨 (noise_scale=2의 진짜 목적)** — 이미지는 [−1,1]이라 표준편차 ~0.5인데 노이즈는 일부러 std 2.0으로 키운다. 그래서 `x_t = image·t + (2ε)·(1−t)`의 크기가 t에 따라 단조롭게 변한다:

| t | 상태 | 대략 표준편차 |
|---|---|---|
| 1.0 | 깨끗 | ~0.5 |
| 0.5 | 중간 | ~1.0 |
| 0.0 | 순수 노이즈 | ~2.0 |

→ 입력 크기만 봐도 노이즈 레벨이 0.5~2.0 범위로 또렷이 드러난다. (사람도 노이즈 낀 사진을 보면 "얼마나 거친지" 누가 안 알려줘도 안다.)

**(2) 크기는 RMSNorm에서 안 죽나? → pre-norm 잔차가 보존** — 입력은 정규화 없이 곧장 `img_embedder`(Conv, 선형 → 큰 입력은 큰 feature)로 들어간다. pre-norm 구조라 norm은 **attention/MLP에 넣을 복사본에만** 적용되고, skip(잔차) 경로 `x`는 정규화되지 않아 임베딩 크기가 끝까지 보존된다. 첫 norm에서 죽지 않는다.

여기서 "잔차 경로 x는 정규화 안 함"이 정확히 **어느 코드**인지가 헷갈리기 쉽다. `DoubleStreamDiTBlock.forward`를 보면:

```python
def forward(self, x, txt, vec):
    x_norm = self.img_norm1(x)               # ① norm 결과는 'x_norm'(복사본)으로 감
    qkv_i  = self.img_qkv(x_norm)...          #   x_norm은 QKV 계산에만 쓰이고 버려짐
    ...
    x = x + self.img_attn_proj(out...)        # ② ← 맨 앞 'x'(정규화 안 된 원본)가 잔차 경로
    x = x + self.img_mlp(self.img_norm2(x))   # ③ norm은 괄호 안 MLP 입력에만, 바깥 'x'는 원본
    return x, txt
```

- **①**: `img_norm1(x)` 결과가 원본 `x`를 덮어쓰지 않고 **새 변수 `x_norm`**에 담긴다 → QKV 계산에만 소비.
- **②③**: `x = x + (...)`에서 **더해지는 쪽의 `x`**(맨 앞)는 norm을 한 번도 안 거친 원본. 이게 "잔차/skip 경로"다.
- 정리: `x = x + Sublayer(Norm(x))` 형태 = **norm은 Sublayer 입력에만, 잔차에는 안 붙음**(= pre-norm의 정의). 반대로 post-norm `x = Norm(x + Sublayer(x))`였다면 잔차까지 정규화돼 크기가 죽는다. 블록 진입 전 `x = img_embedder(img); x = x + pos`에도 norm이 없어, noise_scale=2로 증폭된 입력 크기가 첫 잔차 스트림에 온전히 실린다.

**(3) 크기를 지워도 구조가 신호** — t≈0은 공간 상관 없는 백색잡음, t≈1은 매끄러운 자연 이미지. 통계적 패턴 차이만으로도 t 구분 가능.

이 모든 게 작동하는 근본 이유는 **x-예측**(§2)이라 노이즈 레벨을 느슨하게만 알아도 깨끗한 이미지를 직접 맞히면 되기 때문 = JiT의 핵심 발견.

### Q4. 헤드라인 GenEval 0.88이면 FLUX를 이긴 건가?

아니다. **저자 본인들이 GenEval/DPG는 더 이상 중립 심판이 아니라고 인정**한다 — 작은 120K 정렬셋이 이 점수를 크게 흔들고, 그 정렬셋의 합성 미감이 벤치 스타일과 가까워 다소 "게이밍"된 측면이 있다. **진짜 실력은 PRISM-Bench**(§6 표): 텍스트 렌더링(30.6)·고유명사에서 FLUX/Qwen-Image에 한참 못 미친다. "작고 디버깅 가능한 정직한 baseline"으로서의 가치가 SOTA 주장보다 훨씬 크다.

### Q5. 재현하려면? (코드 관점 주의사항)

| 원하는 것 | 가능 레포 |
|---|---|
| B/16·L/16 추론, Colab 데모 | **PyTorch**(minit2i-torch) — HF 체크포인트 바로 로드 |
| LoRA 적응 | PyTorch |
| B/32 ablation 학습(사전학습+미세조정) | PyTorch |
| **B/16·L/16 처음부터 학습** | **JAX 레포에만** 있음 — PyTorch엔 B/32 학습만 제공 |

즉 헤드라인 모델(0.88)을 PyTorch만으로 처음부터 학습할 수는 없다.

### Q6. 이 연구의 모델 크기는? (왜 "작다"가 핵심인가)

생성기(denoiser) 크기는 **B/32 260M · B/16 258M · L/16 912M**, 텍스트 인코더는 세 모델 공통 **FLAN-T5-Large 341M(동결)**이다(세부 표는 §실험 스케일업 참조). 즉 **가장 큰 L/16도 생성기 기준 ~0.9B**, 인코더 포함해도 ~1.25B.

이 "0.9B"가 메시지의 핵심 — 요즘 상용 T2I 대비 압도적으로 작다:

| 모델 | 생성기 크기 | MiniT2I-L/16 대비 |
|---|---|---|
| **MiniT2I-L/16** | **0.9B** | 1× |
| SD3-Medium | ~2B | 2.2× |
| FLUX.1-dev | 12B | 13× |
| Qwen-Image | 20B | 22× |

- **B/32→B/16은 파라미터 거의 동일**(260M≈258M). 패치만 32→16으로 줄여 토큰을 256→1024로 4배 늘린 것 = 크기가 아니라 토큰 밀도(해상도) 차이.
- **B/16→L/16이 진짜 스케일업**: hidden 768→1248, head 12개(dim 64)→24개(dim 52), 깊이 17 유지 → 258M→912M(약 3.5배).
- 0.9B로 GenEval 0.88이 나온 건 크기보다 **레시피(픽셀+x예측+2단 데이터 분업)** 덕. 저자도 "진짜 병목은 모델 크기가 아니라 데이터"라고 못박음 → **작은 크기 자체가 "T2I는 ImageNet급 자원으로 가능"이라는 주장의 증거**.

---

## 🧪 코드 품질 총평

*미니멀리즘이 코드에 실제로 반영됐는지, 흠은 없는지 점검.*

- **장점**: 전 모델이 `diffusers/mmdit.py` 한 파일에 담길 만큼 짧고 읽기 쉬움. mask_token으로 패딩+CFG 통합, x-예측 변환, velocity-CFG가 깔끔. `from_pretrained` 한 줄 추론 + Colab 제공.
- **흠(미니멀리즘의 잔재)**:
  1. **사용 안 되는 `vec`/`t_embedder`/`pooled_embedder`/`modulate`** — 매 forward마다 vec를 계산해 버림. 의도된 설계지만 코드 위생상 제거 가능(Q1~Q3).
  2. diffusers쪽 attention이 명시적 einsum softmax라 메모리 비효율(학습 config는 sdpa 옵션 별도).
  3. `build_transformer_from_checkpoint`의 if/else 두 분기가 동일 동작(무의미한 분기).
- **비표준 디테일**: noise_scale=2(시간조건 대체용), 절대 sincos + 상대 RoPE 이중 위치 인코딩, logit-normal `mu=−0.8` 치우침.

---

## 📌 한 줄 요약 (전체)

> **MiniT2I = "MM-DiT에서 뺄 수 있는 건 다 뺀(VAE·AdaLN·시간임베딩·RL·보조손실) 픽셀 flow-matching baseline."** x-예측 + noise_scale=2로 시간 조건마저 입력 크기에 녹여 없애고, 사전학습(커버리지)+짧은 정렬셋(추종)의 2단 분업으로 ImageNet급 자원에 GenEval 0.88을 찍었다. 단 그 점수는 정렬셋에 흔들린 것이고, PRISM 기준 텍스트/엔티티는 아직 약하다 — SOTA가 아니라 **"작고 정직하고 재현 가능한 출발선"**이 핵심 가치.

## 🔗 관련 메모리/문서

- [[paper_pixeldit]] — VAE 제거 + REPA 의존 픽셀 확산. MiniT2I는 REPA 같은 보조 정렬조차 없이 더 미니멀하게 가면서 T2I까지 확장.
- [[paper_asymflow]] — 픽셀 flow matching 부활(rank-asymmetric velocity).
- [[paper_tuna_2]] — 인코더 0개 native pixel 멀티모달(He 그룹 인용 계보와 겹침).
- [[paper_dmd2]] / [[paper_pid]] — few-step 증류 비교군(MiniT2I는 Mean Flow 4-step 사용).
- [[reference_pretrained_backbone_reuse_landscape]] — 동결 백본 재사용 분기(MiniT2I = 동결 T5 텍스트 인코더 재사용).
