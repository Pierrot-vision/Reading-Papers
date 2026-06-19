# Ovis-Image: 작은 멀티모달 인코더로 텍스트 렌더링 frontier에 도달한 7B T2I 모델

## 1. 메타 정보

| 항목 | 내용 |
|---|---|
| 논문 제목 | Ovis-Image Technical Report |
| 저자/소속 | Guo-Hua Wang, Liangfu Cao, Tianyu Cui 외 — Ovis Team, **Alibaba Group (AIDC-AI)** |
| 공개일 | 2025-11-28 (arXiv 2025-11) |
| 분야 | text-to-image generation(텍스트→이미지 생성), 특히 in-image text rendering(이미지 안 글자 렌더링) |
| 논문 링크 | [arXiv abstract](https://arxiv.org/abs/2511.22982) · [PDF](https://arxiv.org/pdf/2511.22982) |
| 코드/모델 | [GitHub AIDC-AI/Ovis-Image](https://github.com/AIDC-AI/Ovis-Image) · [HF Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B) (inference 코드+가중치만, 학습 코드 비공개) |
| 베이스 | [[paper_prx]] 계열이 아닌 **FLUX(MMDiT) 슬림화** + 자사 Ovis-U1 패턴 계승 |
| 사용한 외부 모델 | text encoder = **Ovis2.5-2B**(내부 LLM = Qwen3-2B) · VAE = **FLUX.1-schnell**(16채널, 동결) |

> 그림으로 본 전체 구조 (cf. 논문 Fig.2):
> ![Ovis-Image 아키텍처](figures/ovis_image_fig2.png)

---

## 2. 주요 용어 사전 (Glossary)

> 왜 먼저 두는가? 이 논문은 FLUX·Ovis·RLHF 용어가 뒤섞여 있어, 용어를 먼저 풀어두지 않으면 본문이 안 읽힌다.

### 아키텍처
- **MMDiT (Multi-Modal Diffusion Transformer, 멀티모달 확산 트랜스포머)**: 텍스트 토큰과 이미지 토큰을 **하나의 attention(어텐션) 안에서 같이 섞는** 확산 트랜스포머. FLUX/SD3가 쓰는 구조.
- **double-stream block(이중 흐름 블록)**: 텍스트와 이미지가 **각자의 가중치(파라미터)** 를 갖되 attention 단계에서만 합쳐지는 층. 코드 [layers.py `DoubleStreamBlock`](https://github.com/AIDC-AI/Ovis-Image).
- **single-stream block(단일 흐름 블록)**: 텍스트·이미지를 한 줄로 이어붙여(concat) **같은 가중치** 로 처리하는 층. double보다 가볍다.
- **flow matching(흐름 맞추기)**: 노이즈에서 이미지로 가는 **직선 경로의 속도(velocity)** 를 예측하도록 학습하는 방식. diffusion(확산)의 한 종류이자 FLUX의 학습 목표.
- **RoPE (Rotary Position Embedding, 회전 위치 임베딩)**: 토큰 위치를 회전 각도로 넣는 방식. 여기선 (시간/텍스트축, 세로 H, 가로 W) 3축으로 분해 — axes_dim `(16,56,56)`.
- **SwiGLU**: gate(게이트)가 달린 MLP 활성함수. FLUX의 GELU보다 표현력이 좋다고 알려짐. 코드에선 `YakMLP`.
- **VAE (Variational Auto-Encoder, 변분 오토인코더)**: 이미지를 작은 latent(잠재) 텐서로 압축/복원하는 모듈. 여기선 FLUX 것을 **학습 안 하고 그대로(동결)** 씀.

### 핵심 개념 (이 논문의 차별점)
- **text encoder = 이해용 LLM 재활용**: T5 같은 전용 텍스트 인코더 대신, **이미지 이해 모델(Ovis2.5)의 내부 LLM(Qwen3-2B) hidden state(은닉 표현)** 를 텍스트 조건으로 사용.
- **mask & select**: Fig.2의 박스. 실체는 "패딩 토큰을 0으로 지우고(mask) + 앞쪽 고정 지시문 28토큰을 잘라낸다(select)" — 3.3절에서 자세히.
- **refiner 제거**: 이전작 Ovis-U1엔 텍스트 표현을 한 번 더 다듬는 refiner가 있었으나, Ovis-Image는 그걸 빼고 LLM 최종 표현을 **바로** DiT에 넣음.

### 학습/정렬 기법
- **SFT (Supervised Fine-Tuning, 지도 미세조정)**: 고품질 지시문-이미지 쌍으로 가르치는 단계.
- **DPO (Direct Preference Optimization, 직접 선호 최적화)**: "이긴 이미지(winner)에 더 높은 확률, 진 이미지(loser)에 낮은 확률"을 직접 부여하는 정렬 학습.
- **Diffusion-SDPO (Safeguarded DPO)**: 저자들의 선행연구(arXiv 2511.03317). loser 그래디언트가 winner와 **충돌하면 그 세기를 줄여**(식 3의 `λ_safe`) 이긴 샘플 품질이 망가지지 않게 보호.
- **GRPO (Group Relative Policy Optimization, 그룹 상대 정책 최적화)**: 한 프롬프트에 후보 여러 장을 뽑아, **그룹 평균 대비 얼마나 좋은지(advantage)** 로 보상하는 on-policy(현재 정책 기반) RL.
- **coefficients-preserving sampling**: GRPO 가속용(arXiv 2509.05952). 적은 denoising step(노이즈 제거 단계)으로 후보를 뽑되 품질 계수를 보존.

### 평가 지표
- **WA (Word Accuracy, 단어 정확도)**: 생성 이미지 속 글자가 프롬프트와 정확히 일치하는 비율.
- **NED (Normalized Edit Distance, 정규화 편집거리)**: 글자가 얼마나 비슷한지(오타 허용). 높을수록 좋게 정규화됨.
- **CLIPScore**: 이미지-텍스트 의미 일치도.
- **GenEval / DPG-Bench / OneIG-Bench**: 일반 T2I 능력(개수·색·위치·구도·스타일 등) 평가 벤치마크.
- **CVTG-2K / LongText-Bench**: **텍스트 렌더링 전용** 벤치마크(여러 영역 글자 / 긴 글자, 영·중).

---

## 3. 논문 요약 (TL;DR)

**한 줄**: "이미지 안 글자 렌더링은 거대 백본이 필수가 아니다 — 똑똑한 작은 멀티모달 인코더(2B) + 텍스트 중심 학습 레시피면, 20B급 모델을 글자에서 이긴다."

- **핵심 문제**: 강한 텍스트 렌더링은 지금까지 ① OCR 능력 좋은 거대 VLM에 강결합(Qwen-Image = 7B 인코더 + 20B DiT)하거나 ② 클로즈드 소스(Seedream/GPT4o)뿐이었다. 둘 다 배포가 어렵다.
- **해결책**: FLUX MMDiT를 6+27 블록으로 슬림화한 **7B DiT** + 이해용 모델 **Ovis2.5-2B를 텍스트 인코더로 동결 재활용** + **고정 지시문으로 표현을 글자 쪽으로 편향** + **합성 타이포 데이터** + **텍스트에 집중한 4단계 학습(Pretrain→SFT→DPO→GRPO)**.
- **검증**: 텍스트 렌더링(CVTG-2K WA 평균 **0.92**, LongText-中 **0.964**)에서 Qwen-Image·GPT4o를 능가. 일반 생성(GenEval 0.84, DPG 86.6)은 동급. H100 메모리 **24GB**로 Qwen-Image(59GB)의 절반 이하.

---

## 4. 핵심 기여 (Contributions)

1. **작은 인코더 + 중간 DiT로 텍스트 렌더링 SOTA**: 총 ~10B(2B 인코더 + 7B DiT)로 27B급 Qwen-Image를 텍스트 정확도에서 추월.
2. **이해용 멀티모달 LLM의 hidden state를 텍스트 조건으로 직접 사용** + refiner 제거로 구조 단순화.
3. **mask & select** — 고정 시스템 지시문을 붙여 LLM 표현을 글자/공간 속성 쪽으로 편향시키고, 지시문 토큰은 버려 추가 파라미터 0개로 텍스트 친화 표현을 얻음.
4. **텍스트 중심 데이터 레시피**: 렌더링 엔진으로 만든 합성 타이포(폰트·크기·배경 변형) 주입 + 중·영 대규모 recaptioning(재캡션).
5. **4단계 정렬 파이프라인**: SFT → Diffusion-SDPO(winner 보호) → GRPO(텍스트 프롬프트에만 예산 집중, coefficients-preserving 가속).
6. **실용 배포성 실증**: 단일 고급 GPU 1장, 절반 메모리로 near-frontier 텍스트 렌더링.

---

## 5. 주요 알고리즘 설명

### 5.1 전체 데이터 흐름

> 왜? 이 모델이 FLUX와 "어디가 같고 어디가 다른지"를 한눈에 보기 위해.

```
프롬프트 ──(고정 시스템 지시문 강제 prepend)──▶ Ovis2.5의 LLM(Qwen3-2B)
   │
   ├─ last_hidden_state  ──mask(패딩×0)──▶ select(앞 28토큰 잘라냄) ──▶ 텍스트 임베딩(256토큰, 2048차원)
   │
노이즈 latent(16ch) ──patchify(2×2)──▶ 이미지 토큰(64ch)
   │
   ▼
[텍스트 토큰 + 이미지 토큰] ──▶ 6× DoubleStreamBlock ──▶ concat ──▶ 27× SingleStreamBlock
   │ (RoPE 3축 / timestep 변조 / SwiGLU)
   ▼
이미지 토큰만 추출 ──▶ LastLayer ──▶ velocity 예측 ──▶ Euler 적분 ──▶ FLUX VAE decode ──▶ 이미지
```

| 구성요소 | 값 (코드 `ovis-image-7b` config로 확인) | FLUX-dev 대비 |
|---|---|---|
| double-stream 블록 | **6** | 19 → 대폭 축소 |
| single-stream 블록 | **27** | 38 → 축소 |
| hidden size | 3072 | 동일 |
| attention heads | **24** (head_dim 128) | 동일(24) |
| 활성함수 | **SwiGLU** | GELU → 교체 |
| context 입력차원 | **2048** (= Qwen3-2B hidden) | T5의 4096과 다름 |
| in/out channels | 64 (= 16ch VAE × 2×2 patch) | 동일 |
| RoPE axes_dim | (16, 56, 56) | 동일 |
| VAE | FLUX.1-schnell 16ch, scale 0.3611 / shift 0.1159, **동결** | 동일 |
| 파라미터(Table 1) | DiT 7.37B + 인코더 2.57B + VAE 0.08B = **10.02B** | — |

> ⚠️ 코드 함정: [`args.py`](https://github.com/AIDC-AI/Ovis-Image)의 기본값은 depth 19/38(FLUX 값)이지만 **미사용 디폴트**다. 실제로는 `__init__.py`의 `ovis_image_configs["ovis-image-7b"]`(6/27, SwiGLU, context 2048)가 덮어쓴다. 논문 본문 "6 double + 27 single"과 일치.

### 5.2 mask & select — 이 논문의 핵심 트릭 (코드로)

> 왜? Fig.2에서 가장 모호하게 그려졌지만 실제로는 이 모델의 텍스트 강점을 만드는 장치이기 때문.

[tokenizer.py](https://github.com/AIDC-AI/Ovis-Image)는 **모든 프롬프트 앞에 고정 시스템 지시문**을 강제로 붙인다:

> *"Describe the image by detailing the color, quantity, **text**, shape, size, texture, spatial relationships of the objects and background:"*

이 지시문 + 챗 템플릿 prefix가 정확히 **28토큰**이다. 그리고 [hf_embedder.py](https://github.com/AIDC-AI/Ovis-Image)의 forward가 하는 일:

```python
outputs = self.hf_module(input_ids=batch_tokens, attention_mask=attention_mask)  # Qwen3 디코더 통과
txt = outputs.last_hidden_state
txt = txt * attention_mask[..., None]      # (mask) 패딩 토큰을 0으로
txt = txt[:, self.user_prompt_begin_id:, :]  # (select) 앞 28토큰(=지시문) 잘라냄
```

**작동 원리** (한 줄 풀이 동반):
- Qwen3는 디코더(causal attention, 앞쪽만 보는 LLM)다. → 뒤따라오는 **실제 사용자 프롬프트 토큰**들이 앞의 지시문("...detailing the **text**, shape, spatial...")을 attention으로 흡수해, **글자·모양·공간 속성을 강조한 표현으로 물든다.**
- 그 다음 지시문 토큰 28개는 버린다(`[28:]`). → DiT엔 "텍스트 친화로 편향된 사용자 프롬프트 임베딩"만 들어간다. **지시문을 그대로 DiT에 넣으면** "color, quantity, text..." 같은 메타 단어를 그림으로 그리려 해서 방해되므로, **영향(물든 표현)만 남기고 토큰은 버리는** 것이 핵심.
- 결과: **추가 파라미터 0개·추가 모듈 0개**로 텍스트 렌더링에 유리한 조건부 표현을 얻는다. (비유: 시험 전 "글자랑 위치 특히 신경 써서 봐"라고 귀띔만 해두고, 답안엔 그 귀띔 문장은 안 적고 귀띔 들은 머릿속만 제출하는 것.)

> 한 줄 비유: priming(프라이밍, 미리 물들이기) + prefix strip(앞부분 잘라내기). causal attention(앞쪽만 보는 디코더) 덕분에 지시문의 영향이 사용자 프롬프트 토큰에 이미 새겨지므로, 원본 지시문 토큰은 버려도 된다.

**추론 일관성 (train-test consistency)**: 이 트릭의 전제는 "**학습·추론 모두 항상 붙이고 항상 자른다**"이다. 코드가 이를 자동 강제한다 — [tokenizer.py](https://github.com/AIDC-AI/Ovis-Image)의 `encode`는 사용자가 시스템 프롬프트를 안 넘기면 기본 지시문을 자동 주입(`if len(system_prompt)==0: system_prompt = self.system_prompt`)하고, embedder는 무조건 `[28:]` strip한다. 따라서 일반 사용자가 `pipe(prompt)`만 호출해도 "붙이기→자르기"가 한 쌍으로 돌아가, **추론 때 깜빡 안 붙여서 분포가 어긋나는(distribution shift) 사고가 구조적으로 막혀 있다.** CFG의 빈 프롬프트(`encode("")`)도 지시문은 그대로 붙으므로 무조건 분기까지 형식이 일관된다.

> ⚠️ 진짜 취약점은 "안 붙이는 것"이 아니라 **"다른 길이의 커스텀 시스템 프롬프트로 교체하는 것"**이다. `user_prompt_begin_id = 28`이 tokenizer·embedder **양쪽에 하드코딩**돼 있어, 지시문 문구를 바꾸면 토큰 길이가 28과 어긋나 사용자 프롬프트 앞부분이 잘리거나 지시문 꼬리가 남는다(코드 관점 한계, 7장 Q5 참조).

### 5.3 DiT 블록 내부 (FLUX 표준 MMDiT)

- **DoubleStreamBlock**: 이미지/텍스트가 각자 norm→modulation→QKV를 거친 뒤 q,k,v를 이어붙여(concat) **공동 attention**, 그 후 각자 MLP. **텍스트 스트림도 정상 업데이트**된다 (텍스트 갱신을 없앤 [[paper_prx]]와 다름).
- **SingleStreamBlock**: 텍스트+이미지를 한 줄로 이어 같은 가중치로 처리. SwiGLU일 땐 `linear1`이 qkv + mlp + mlp_gate를 한 번에 뽑음.
- **timestep 조건**: `timestep_embedding`(sinusoidal 256차원) → MLP → 각 블록의 `Modulation`이 shift/scale/gate 생성(AdaLN 방식).
- **초기화**: `LastLayer`와 `Modulation`의 마지막 Linear는 **0으로 초기화**(`init_weights`) → 학습 초기에 항등(identity)에 가깝게 시작하는 표준 DiT 안정화.

### 5.4 샘플링 (inference)

> 왜? FLUX와 같은 flow matching이라 추론 코드가 거의 동일함을 확인하기 위해.

[sampling.py](https://github.com/AIDC-AI/Ovis-Image) `denoise`:
- FLUX식 **time-shift 스케줄**(`base_shift 0.5`, `max_shift 1.15`) — 해상도(토큰 수)에 따라 노이즈 일정을 비틂.
- **CFG (Classifier-Free Guidance)**: 빈 프롬프트("")로 무조건 분기를 만들고 `pred_u + scale*(pred_c - pred_u)`. 기본 `cfg_scale 5.0`.
- **Euler 적분**: `latents = latents + (t_prev - t_curr) * pred` 한 줄. 기본 50 step.

---

## 6. 실험 요약

> 왜? "텍스트는 압도, 일반 생성은 동급"이라는 이 논문의 실제 성적표를 숫자로 못박기 위해.

### 6.1 텍스트 렌더링 — 압도적 (핵심 셀링포인트)

**CVTG-2K (영문 다중영역 단어정확도)**

| 모델 | #Params | WA(평균) | NED↑ | CLIPScore↑ |
|---|---|---|---|---|
| GPT4o | - | 0.857 | 0.948 | 0.798 |
| Qwen-Image | 7B+20B | 0.829 | 0.912 | 0.802 |
| **Ovis-Image** | **2B+7B** | **0.920** | **0.970** | **0.837** |

**LongText-Bench (긴 글자, 영·중)**

| 모델 | EN | ZN |
|---|---|---|
| GPT4o | **0.956** | 0.619 |
| Qwen-Image | 0.943 | 0.946 |
| **Ovis-Image** | 0.922 | **0.964** |

→ 중국어 텍스트는 전체 1위. 영어는 Qwen에 근소하게 밀림.

### 6.2 일반 생성 — 동급, 약간 아래

| 벤치 | Ovis-Image | Qwen-Image | GPT4o | 메모 |
|---|---|---|---|---|
| GenEval Overall | 0.84 | 0.87 | 0.84 | Counting 0.76이 약점, 자사 Ovis-U1(0.89)보다도 낮음 |
| DPG Overall | 86.59 | 88.32 | 85.15 | **Global 82.37**이 눈에 띄게 낮음(아래 Q&A) |
| OneIG-EN Overall | 0.530 | **0.539** | 0.533 | 단 Text 항목은 0.914로 Qwen(0.891) 추월 |
| OneIG-ZN Overall | 0.521 | **0.548** | 0.474 | Text 0.961로 Qwen(0.963)과 거의 동률 |

> Diversity(다양성) 점수가 전반적으로 낮음(0.186~0.198) — RLHF 정렬의 전형적 부작용(mode collapse 경향).

### 6.3 효율 (Table 8, 1024² 50step BF16)

| 모델 | #Params | H100 메모리 | H100 시간 |
|---|---|---|---|
| FLUX.1-dev | 11B+12B | 34.7GB | 11.0s (guidance distill) |
| Qwen-Image | 7B+20B | 59.4GB | 20.3s |
| Ovis-U1 | 2B+1B | 11.9GB | 4.3s |
| **Ovis-Image** | **2B+7B** | **24.3GB** | 13.7s |

→ Qwen-Image 대비 메모리 절반 이하. 다만 자사 Ovis-U1보단 무겁고, distill된 FLUX-dev보단 느림(정직하게 기재).

---

## 7. 💬 Q&A — 코드/비판 관점

### Q1. "7B 모델"이라는데 실제 크기는?

DiT만 7.37B이고, **총합은 10.02B**(DiT 7.37 + Ovis2.5 인코더 2.57 + VAE 0.08). 표기는 "2B+7B"로 정직하게 병기하지만, 제목의 "7B"는 DiT 기준이라 약간의 마케팅 뉘앙스가 있다.

### Q2. FLUX와 정확히 무엇이 다른가?

같은 것: MMDiT 골격, RoPE 3축, flow matching, time-shift 스케줄, FLUX VAE, CFG.
다른 것: ① 블록 수 6+27로 슬림화 ② 활성함수 SwiGLU ③ **텍스트 인코더가 T5/CLIP이 아니라 이해용 LLM(Qwen3-2B)** ④ mask & select로 고정 지시문 편향 ⑤ refiner 제거. 즉 "구조 발명"이 아니라 "조건부 입력과 데이터/RL 레시피의 재설계"가 본질.

### Q3. DPG Global 82.37이 왜 의심스러운가?

표에서 Ovis-Image의 DPG Global(전역 구도) 점수 **82.37**이 자사 이전작 **Ovis-U1과 소수점까지 똑같다**. 표 재활용이거나, 전역 구도 파악이 이 계열의 구조적 약점일 가능성. 검증 불가한 의심 지점으로만 기록.

### Q4. 재현 가능한가?

부분적으로만. **inference 코드 + 가중치만 공개**, 학습 코드·데이터·보상모델 세부는 비공개. SFT조차 재현 불가. GRPO 보상모델 구성(HPSv3/CLIP/PickScore 앙상블)은 DPO 데이터 설명에만 등장하고 RL 단계엔 구체성이 부족.

### Q5. 코드에서 발견한 허점은?

1. **매직넘버 28 하드코딩** (tokenizer·embedder 양쪽) — 시스템 지시문 변경 시 정렬 붕괴. 동적 계산이 아님.
2. **패딩 토큰을 attention에서 안 거름** — DiT forward에 attention mask 인자가 없어, 0으로 곱한 패딩이 RMSNorm+bias를 거쳐 상수 벡터가 된 채 어텐션됨(FLUX 관행과 동일하나 미세 비효율).
3. **편집 기능 없음** — Ovis-U1은 이해+생성+편집 통합이었으나 이번엔 **순수 T2I만**. 코드에 이미지 컨디셔닝 경로 없음.
4. **`double_block_type` 인자가 dead config** — 항상 DoubleStreamBlock으로 고정.

### Q6. 이 논문의 진짜 신규성은?

학술적 신규성은 낮다(구조=FLUX, RL=자사 선행연구 조합). 대신 **"텍스트 렌더링 = 거대 백본 필수"라는 통념을 깬 실증**과 엔지니어링 완성도가 가치. 핵심 발명은 mask & select 트릭 정도이며, 나머지는 "잘 통합한 레시피".

### Q7. 텍스트(글자) 학습셋은 무엇을 썼나?

3장(Data Composition) 기준, **단계마다 다른 비율**로 구성된다:

| 단계 | 텍스트 관련 데이터 |
|---|---|
| **Pretrain** | ① **렌더링 엔진으로 만든 합성 타이포(synthetic typography)** — 깨끗한 글자를 다양한 배경·레이아웃에 합성, 폰트·크기·배치를 통제한 정답 ② 글자가 두드러진 실제 슬라이스(포스터·배너·로고·UI) ③ 중·영 대규모 recaptioning(글자 내용까지 캡션에 반영) |
| **SFT** | 1024 고해상도 고품질 쌍, 자연 이미지 + 합성 데이터 일부(선명한 디테일·통제된 레이아웃·희귀 개념 보강) |
| **DPO** | 선호쌍. 약 90% 일상/객체 고품질 생성물 + **약 10% 디자인·창작 계열**(포스터·일러스트·스타일화 → 글자 많고 레이아웃 복잡) |
| **GRPO** | DPO와 **일부러 다른 분포**. 좁은 **텍스트 렌더링 전용 프롬프트**(포스터·제목 카드·UI·상품 라벨)에만 집중, 중·영, 짧은 슬로건~긴 다중 라인 |

> ⚠️ **구체적 공개 데이터셋명은 전혀 밝히지 않는다.** "rendering engine", "web/licensed/synthetic 혼합", "in-house collection" 같은 추상 출처만 있고 폰트셋·코퍼스·합성 엔진의 실체는 비공개. 데이터·학습 코드 모두 미공개라 **텍스트 학습셋은 재현 불가**.

### Q8. 학습 때 지시문을 붙였는데, 추론 때 안 붙이면 문제 없나?

추론 때도 **반드시 붙여야** 하고, **코드가 자동 강제**한다 → 사용자가 신경 쓸 필요 없음. 자세한 메커니즘은 5.2절 "추론 일관성" 참조. 요약: tokenizer가 기본 지시문을 자동 주입 + embedder가 무조건 28토큰 strip → "붙이기→자르기"가 한 쌍으로 돌아가 train-test 분포 어긋남이 구조적으로 차단됨. 진짜 위험은 안 붙이는 게 아니라 **다른 길이의 지시문으로 교체**할 때(28 하드코딩과 충돌).

---

## 8. 한 줄 요약 (전체)

**FLUX MMDiT를 6+27로 슬림화한 7B DiT에, 이해용 LLM(Ovis2.5-2B)을 텍스트 인코더로 동결 재활용하고 고정 지시문으로 표현을 글자 쪽으로 편향(mask & select)시킨 뒤, 합성 타이포 데이터 + 4단계 텍스트 중심 정렬(SFT→SDPO→GRPO)로 학습 — 총 10B로 27B Qwen-Image를 텍스트 렌더링에서 이기고 메모리는 절반으로 줄인 "경량 텍스트 특화 T2I".**

---

## 9. 관련 메모리 링크

- 백본 재사용 분류: [[reference_pretrained_backbone_reuse_landscape]] — 본 논문은 **분기 A**(이해 백본 동결 재사용 + 신규 확산 디코더).
- 같은 "동결 이해모델 인코더" 계보: [[paper_qwen_image]](Qwen2.5-VL 동결), [[paper_qwen_image_2]], [[paper_z_image]](경량 SOTA), [[paper_lumina_image_2]].
- 텍스트 스트림 갱신을 **제거**한 대조군: [[paper_prx]] (Ovis-Image는 반대로 텍스트 스트림 유지).
- 정렬 기법 대조: DMD 계열 증류([[paper_dmd2]])와 달리 본 논문은 DPO/GRPO 정렬 중심.
