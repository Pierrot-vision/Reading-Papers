# PAPER: Qwen-Image-2.1 — 2.0 설계를 처음 공개한 7B single-stream 생성·편집 모델

## 0. 이 문서를 읽는 법

이 문서는 Qwen-Image-2.1 의 **블로그 + HF 가중치 + GitHub 저장소 + diffusers 추론 코드**를 직접 대조한 리뷰입니다. 2.1 은 기술 리포트(논문)가 없어서, 구조는 **가중치 헤더 실측**과 **코드**에서 복원했습니다.

> **Qwen-Image-2.1 은 2.0 기술 리포트의 설계(동결 Qwen3-VL + single-stream DiT + f16c64 VAE)를 처음으로 공개한 가중치이자, block-causal 마스크 + prefix KV cache 로 다중 참조 편집을 가속하고 RGBA(투명 배경)까지 한 모델에 넣은 7B DiT 이다. 단 라이선스는 비상업으로 바뀌었다.**

1. **메타 정보·용어 사전**
2. **큰 그림(TL;DR)·핵심 기여**
3. **모델 구조**: 가중치 실측 + 코드에서 드러난 설계 6가지
4. **1.x(2512) vs 2.1 코드 비교**: "20B → 7B" 의 실체
5. **KV cache 의 실제 효과**
6. **실험(벤치마크)**
7. **Q&A**: 2.0 리포트와의 관계 / 급소
8. **한 줄 요약 / 관련 링크**

> Q&A: Q1 2.0 리포트와의 관계 · Q2 급소 · Q3 전체 구조·flow · Q4 block-causal · Q5 causal_condition + prefix KV cache · Q6 마스크 규칙 ↔ 실제 코드

> ⚠️ **공개 범위 주의**: 학습 코드·데이터·레시피·ablation·편집 벤치·속도 수치는 전부 미공개. 아래 수치 중 "실측"은 safetensors 헤더로 직접 센 값, "추정"은 코드 크기로 직접 계산한 값이다.

---

## 1. 메타 정보

| 항목 | 내용 |
|---|---|
| 모델 | Qwen-Image-2.1 |
| 소속 | Alibaba Qwen 팀 (라이선스 주체: Hangzhou Tongyi Laboratory) |
| 공개일 | 2026-09-20 (HF 저장소 생성 2026-09-14, diffusers PR 머지 2026-09-18) |
| 논문 | **없음** (블로그만) |
| 블로그 | https://qwen.ai/blog?id=qwen-image-2.1 |
| GitHub | https://github.com/QwenLM/Qwen-Image-2.1 — **README + `prompt_rewrite/` 뿐**, 모델 코드 없음 |
| 가중치 | https://huggingface.co/Qwen/Qwen-Image-2.1 · PE: `Qwen/Qwen-Image-2.1-PE-T2I`, `Qwen/Qwen-Image-2.1-PE-I2I` |
| 추론 코드 | diffusers `QwenImage21Pipeline` ([PR #14804](https://github.com/huggingface/diffusers/pull/14804)): `transformer_qwenimage21.py`, `pipeline_qwenimage21.py`, `autoencoder_kl_qwenimage21.py` |
| 분야 | Text-to-Image, Image Editing, 투명 배경(RGBA) 생성 |
| 외부 의존 모델 | Qwen3-VL-8B (조건 인코더, 동결), Qwen3.5 계열 9B (Prompt Enhancer) |
| 설계 원논문 | Qwen-Image-2.0 Technical Report (arXiv 2605.10730) → [PAPER_Qwen-Image-2.0.md](PAPER_Qwen-Image-2.0.md), Q1 참고 |
| 라이선스 | **Qwen Research License — 비상업 전용** (1.x 계열은 Apache 2.0) |

---

## 2. 주요 용어 사전 (Glossary)

*본문에 처음 나오는 용어가 헷갈리지 않게 한곳에 모아둠.*

### 아키텍처

| 용어 | 풀이 |
|---|---|
| single-stream DiT | 텍스트 토큰과 이미지 토큰이 **같은 가중치**(attention·MLP)를 공유하며 한 줄 시퀀스로 흐르는 Diffusion Transformer. 반대는 dual-stream(모달리티마다 가중치를 따로 가짐, 1.x·FLUX 방식) |
| modulation(변조) | timestep(노이즈 수준) 정보를 각 층 활성값에 곱하거나 더해 주입하는 장치(adaLN). 2.1 은 **곱셈(scale)·gate 만** 쓰고, 32층 전체가 **한 벌을 공유** |
| block-causal mask | attention 규칙 `(q ≥ kv) 또는 같은 이미지 블록`. 텍스트는 LLM 처럼 앞만 보고(causal), 이미지 한 장 안에서는 양방향으로 본다 |
| causal_condition | 텍스트·조건 이미지 토큰을 **t=0(깨끗한 상태)** 로 modulation 하는 설정. prefix 계산이 스텝과 무관해져 캐시가 가능해짐 |
| prefix KV cache | 앞쪽(시스템 프롬프트·참조 이미지·지시문)의 Key/Value 를 첫 스텝에 한 번 계산해 저장하고, 이후 스텝은 타깃 이미지 토큰만 계산하는 방식. LLM 의 prefill/decode 와 같은 구조 |
| mixed-granularity attention | 블로그 용어. 텍스트는 토큰 단위(token-level), 이미지는 덩어리 단위(chunk-level) 마스크를 섞는다는 뜻 = block-causal mask |
| f16c64 VAE | 공간 16배 다운샘플 + 잠재 64채널 VAE. 2.1 은 입출력을 **RGBA 4채널**로 확장 |
| patch_size | 잠재 몇 칸을 토큰 1개로 묶는가. 1.x 는 2(2×2 묶기), 2.1 은 1(안 묶음) |
| MSRoPE | 텍스트·이미지를 한 좌표계에 얹는 3축(frame·height·width) RoPE(회전 위치 인코딩) |

### 추론 / 학습

| 용어 | 풀이 |
|---|---|
| CFG (Classifier-Free Guidance) | 조건 있는 예측과 없는 예측을 섞어 조건을 강하게 따르게 하는 기법. 스텝마다 모델을 두 번 호출 |
| guidance-free(CFG 없음) | 2.1 기본값 `true_cfg_scale=1.0`. CFG 효과를 학습 때 모델 안에 흡수시켜 한 번 호출로 끝냄 |
| NFE (Number of Function Evaluations) | 이미지 한 장에 모델을 몇 번 호출하는가. 2.1 기본 40스텝 × CFG 없음 = 40 NFE |
| Prompt Enhancer(PE) | 짧은 사용자 입력을 긴 구조적 프롬프트로 재작성하는 LLM. 2.1 은 T2I 용·편집용 두 개를 공개 |
| dynamic shifting | 해상도(토큰 수)에 따라 노이즈 스케줄을 밀어주는 기법. 큰 이미지일수록 고노이즈 구간에 스텝을 더 배분 |

---

## 3. 큰 그림 (TL;DR) 과 핵심 기여

### 3.1 TL;DR

- **한 줄**: 2.0 리포트 설계(동결 Qwen3-VL + single-stream DiT + f16c64 VAE)를 7B DiT 로 공개하고, block-causal 마스크 + prefix KV cache 로 다중 참조 편집을 가속, RGBA 를 통합한 모델.
- **핵심 문제**: 1.x 는 생성(Qwen-Image)·편집(Edit)·투명 배경(Layered)이 **별도 모델**이고 DiT 만 20B 로 무거웠다. 참조 이미지가 늘수록 매 스텝 전체 시퀀스를 다시 계산하는 비용도 컸다.
- **해결책**: (구조) 중복 가중치 제거로 DiT 20.4B → 7.1B / (추론) prefix KV cache / (기능) RGBA VAE + 최대 10장 참조 + 마스크 편집.
- **검증**: 자사 Qwen-Image-Bench 60.28 로 7위 (2.0 Pro 57.84 보다 위). 편집·속도 수치는 없음.

### 3.2 핵심 기여 (Contributions)

1. **2.0 설계의 첫 공개 가중치**: DiT 7.12B + Qwen3-VL-8B + VAE 0.34B, diffusers·vLLM-Omni·SGLang·ComfyUI Day-0 지원.
2. **block-causal + causal_condition → prefix KV cache**: 참조 이미지·지시문을 한 번만 계산하고 재사용.
3. **통합**: T2I + 편집 + 투명 배경 생성·편집 + 사진에서 피사체를 RGBA 로 추출, 한 모델.
4. **편집 입력 다양화**: 참조 최대 10장, 원·페인트 표시·**원본+별도 마스크** 두 장 입력으로 영역 지정.
5. **Prompt Enhancer 공개**: Qwen3.5 계열 9B thinking 모델 2종 + 재작성 코드 (2.0 에선 비공개였던 부분).

---

## 4. 모델 구조

*리포트가 없으므로 "무엇이 들어 있나"를 가중치 헤더로 먼저 세고, "어떻게 도나"를 코드로 확인한다.*

### 4.1 부품별 파라미터 (실측)

| 부품 | 실측 | 내용 |
|---|---|---|
| DiT | **7.115B** (bf16, 14.23GB) | 32층, 폭 4096, 32헤드(head_dim 128), SwiGLU(×3), bias 없음 |
| text/vision encoder | **8.767B** (Qwen3-VL-8B) | **DiT 보다 큼** |
| VAE | **0.338B** (encoder 79M, decoder 259M, fp32) | 4채널(RGBA) 입력, z=64, 16배 공간 압축, residual 구조 |
| **합계** | **약 16.2B, bf16 약 32.5GB** | "7B" 는 전체 파이프라인의 44% |

**DiT 7.115B 내역**

| 모듈 | 크기 | 비중 |
|---|---|---|
| attention (q/k/v/out) | 32 × 67.1M = 2.15B | 30% |
| MLP (SwiGLU: gate·proj·out) | 32 × 151M = 4.83B | 68% |
| **modulation** | **67M, 딱 한 벌** (`modulation.1.weight` [16384, 4096]) | **0.9%** |
| 나머지 (img_in, txt_in, timestep embedder, norm_out, proj_out) | 약 0.07B | 1% |

### 4.2 블로그 그림 — mixed-granularity attention

*KV cache 가 왜 성립하는지는 마스크 모양 하나로 설명된다.*

![Qwen-Image-2.1 mixed-granularity attention](figures/qwen_image_2_1_fig2.png)

> 블로그 그림: *Qwen-Image-2.1 mixed-granularity attention architecture*

- 세로(Q) = 질의 토큰, 가로(K) = 참조되는 토큰. 색칸 = 볼 수 있음, 회색 = 못 봄.
- 시퀀스 순서: **[System prefix → Input image(선택) → Edit instruction → Target image]**.
- 파랑(텍스트)은 계단 모양 = 토큰 단위 causal. 초록(이미지)은 블록 안이 꽉 참 = 이미지 한 장 안에서 양방향.
- 타깃 이미지가 맨 뒤라 **앞쪽 prefix 는 타깃을 볼 수 없다** → prefix 는 스텝이 바뀌어도 그대로 → "Compute once · Reuse K/V". 타깃만 "Compute each step".

### 4.3 코드에서 드러난 설계 6가지

**① modulation 을 32개 블록이 완전히 공유**
- 가중치 모양 [16384, 4096] 에서 16384 = 4 × 4096 → scale1·gate1·scale2·gate2 네 벡터가 정확히 한 벌.
- 블록마다 `mod1, mod2 = modulation.chunk(2)` 로 **똑같은 값**을 받는다. shift(더하기 항)는 없고 곱셈만, gate 에는 tanh.
- 블록별 학습 벡터조차 없어서 PixArt 의 adaLN-single 보다 더 극단적이다.
- ⚠️ docstring 은 "every block slices its own scales and gates" 라고 쓰지만 **코드와 가중치 모양 모두 그 설명과 다르다** → 문서 오류.

```python
# transformer_qwenimage21.py (요약)
self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(4096, 4 * 4096, bias=False))  # 전 블록 공유 1벌
# block.forward
mod1, mod2 = modulation.chunk(2, dim=-1)          # 모든 블록이 같은 값
scale, gate = mod1.chunk(2, dim=-1)
x = x + gate.tanh() * attn(norm1(x) * (1 + scale))  # h' = αh, shift 없음
```

**② 진짜 single-stream**
- 가중치에 `txt_mlp` 가 없고 `img_mlp` 하나뿐 → 텍스트도 이미지와 같은 MLP 통과.
- 텍스트는 입구 `txt_in` (ZeroCenter RMSNorm → Linear → GELU → Linear) 한 번만 거친 뒤 이미지와 섞임.

**③ VLM 의 이미지 특징은 버리고 VAE 잠재로 갈아 끼움**
- Qwen3-VL 에서 이미지 칸(`<|image_pad|>`) 1개 = 32×32px (patch 16 × merge 2).
- 이 칸을 VAE 잠재 토큰 2×2(16px 네 개)로 늘린 뒤 **그 자리를 VAE 잠재로 덮어씀**:

```python
repeats = torch.where(img_mask, 4, 1)[0]                       # 이미지 칸은 4배로
joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
joint_hidden_states[:, image_pad_mask] = hidden_states         # VLM 값 → VAE 잠재로 덮어쓰기
```

- 따라서 VLM 이 이미지를 "이해한 결과"는 **이미지 뒤에 오는 지시문 토큰에만** 간접적으로 남는다 (2.0 리포트 Fig 8 의 빨간 X 와 동일).
- VLM 입력은 마지막 층의 **정규화(RMSNorm) 전** hidden state. transformers 5.x 는 정규화 후 값을 돌려주므로 파이프라인이 forward hook 으로 정규화를 무력화한다 (주석: 빼면 "글자 렌더링부터 망가진다").

**④ block-causal 마스크 + causal_condition → KV cache** (의미 풀이 → Q4·Q5, 코드 줄 매칭 → Q6)

```python
allowed = ((q_idx >= kv_idx) | same_image_block) & key_valid[b, kv_idx]
```

- `causal_condition=True`: timestep 배열 끝에 t=0 행을 하나 더 붙이고, 타깃이 아닌 토큰은 그 행의 modulation 을 읽는다 → prefix 활성값이 스텝과 무관.
- 첫 스텝 `kv_cache_mode="extract"` 로 prefix K/V 저장, 이후 `"cached"` 로 타깃 query 만 계산.
- 계보: 1.x **Edit-2511 에 이미 `zero_cond_t: true`** (조건 이미지를 t=0 으로 modulation) 가 있었다. 2.1 은 여기에 causal 마스크를 더해 **캐시 가능성**을 얻었다.

**⑤ RoPE 좌표 배치**
- 3축 (frame 16 + height 56 + width 56 = 128), theta 10000.
- 텍스트는 세 축 모두 같은 위치로 전진. 이미지는 frame 축을 직전 텍스트 위치에 고정하고, height·width 를 **0 중심 격자**로 배치.
- → 참조 이미지와 타깃이 **같은 공간 좌표를 공유** (Any2AnyTryon 의 condspa 와 같은 발상). 이미지끼리는 frame 값으로 구분.

**⑥ CFG 없이 샘플링**
- `true_cfg_scale` 기본 1.0, docstring "Qwen-Image 2.1 is meant to be sampled without guidance". SGLang 예시도 `--guidance-scale 1`.
- 2.0-RL 리포트의 "CFG is also integrated into the student model after OPD" 와 맞아떨어짐 (→ Q1).

### 4.4 추론 설정 (코드 기본값)

*입력 하나가 파이프라인 전체를 지나는 흐름·텐서 모양은 → Q3.*

| 항목 | 값 |
|---|---|
| 스텝 | 40 (Euler, flow matching) |
| 노이즈 스케줄 | `sigmas = linspace(1, 1/40, 40)` + dynamic shift (base 0.5 @256 토큰 → max 0.9 @8192 토큰, exponential, shift_terminal 0.02) |
| 기본 해상도 | **1024²** (`output_resolution=1024`) — README 의 "기본 2048²" 와 다름 (→ Q2) |
| 권장 2K 크기 | 1:1 2048², 16:9 2752×1536 등 7종 |
| 조건 이미지 | 면적 1024² 로 리사이즈 (VLM 과 VAE 가 같은 크기 사용), RGBA 는 VLM 쪽만 흰 배경 합성 |
| 투명 생성 프롬프트 | `This is an RGBA image with transparency. <설명>. The image has alpha channel and the background is transparent.` |

---

## 5. 1.x(2512) vs 2.1 코드 비교 — "20B → 7B" 의 실체

*파라미터가 1/3 이 됐다는 숫자가 곧 "3배 빠르다"는 뜻인지 확인하려는 절.*

| 항목 | Qwen-Image 1.x (2512) | **Qwen-Image 2.1** |
|---|---|---|
| DiT 파라미터 (실측) | **20.43B** | **7.12B** |
| 층 × 폭 | 60 × 3072 | 32 × 4096 |
| stream 구조 | dual (텍스트·이미지가 attention 가중치·MLP 를 각각 따로) | single (전부 공유) |
| modulation | 블록마다, stream 마다 6벡터(shift·scale·gate ×2), bias 있음 → **6.80B (33%)** | 전 블록 공유 4벡터, bias 없음 → **0.067B (0.9%)** |
| MLP | GELU, 4배 확장 | SwiGLU, 3배 확장 |
| text encoder | Qwen2.5-VL-7B | Qwen3-VL-8B |
| VAE | f8, 16채널 (+ patch 2) | f16, 64채널 (patch 1), RGBA |
| 토큰 1개가 담당하는 픽셀 | 16×16 | 16×16 (**같음**) |
| attention 마스크 | 양방향 전체 | block-causal |
| 편집 / 투명 | 별도 모델 (Edit-2509/2511, Layered) | 통합 |
| 라이선스 | **Apache 2.0** | **비상업** |

### 5.1 핵심 계산: 토큰당 계산량은 거의 같다

1.x 20.43B 실측 내역: attention 4.53B · 이미지 MLP 4.53B · 텍스트 MLP 4.53B · modulation 6.80B.

- 이미지 토큰 하나가 실제로 거치는 가중치 = 이미지용 attention(절반) + 이미지 MLP ≈ **2.27B + 4.53B = 6.8B**.
- 2.1 은 모든 토큰이 전 가중치를 거침 → **6.98B**.
- 토큰 1개 = 16×16px 도 동일 (1.x: f8 VAE × patch 2, 2.1: f16 VAE × patch 1).
- → **같은 해상도에서 선형층 계산량은 거의 같다.**
- attention 계산량(층 수 × 폭에 비례)만 보면 1.x 60 × 3072 vs 2.1 32 × 4096 → 2.1 이 약 **71%**.

> 한 줄: 13B 가 사라진 이유는 **텍스트 전용 가중치 복사본 제거 + 블록별 modulation 제거**. T2I 한 스텝 속도는 크게 달라지지 않고, 대신 VRAM 이 준다 (DiT 40.9GB → 14.2GB). 실제로 빨라지는 곳은 다중 참조 편집이며 그 이유는 KV cache (→ §6).

---

## 6. KV cache 의 실제 효과 (추정 계산)

*블로그가 "빠르고 메모리도 줄인다"고만 하고 수치를 안 주므로, 코드 크기로 직접 계산해 본다.* 참조 이미지는 기본값대로 면적 1024² (= 4,096 토큰/장) 가정.

| 상황 | 캐시 없을 때 스텝당 토큰 | 캐시 있을 때 스텝당 query | 선형층 계산 절감 | 캐시 메모리 (bf16, 32층) |
|---|---|---|---|---|
| T2I 2048² | 약 16.6K | 약 16.4K | **약 1%** | 작음 |
| 편집, 참조 1장 → 1024² | 약 8.4K | 4.1K | 약 2배 | 약 2.3GB |
| 편집, 참조 10장 → 1024² | 약 45K | 4.1K | **약 11배** | **약 22GB** |

- KV cache 는 **사실상 편집 전용 가속**. T2I 는 prefix 가 텍스트 수백 토큰뿐이라 효과가 거의 없다.
- 메모리 계산: 41.3K 토큰 × 4096 × (K+V) 2 × 2바이트 ≈ 677MB/층 × 32층 ≈ 22GB.
- 블로그는 "reduce memory usage" 라고 하지만 diffusers 구현은 prefix K/V 를 40스텝 내내 들고 있어야 해서 **상주 메모리가 오히려 늘어난다**. vLLM-Omni 가 "FP8 prefix KV 저장"을 따로 내세우는 이유로 보인다.
- docstring: `use_kv_cache` 를 켜고 끄면 **bf16 에서 같은 시드라도 다른 그림**이 나온다 (반올림 차이가 32층 × 40스텝 증폭). 재현성이 필요하면 이 값을 고정.

---

## 7. 실험 결과

*공개된 정량 수치가 이 그림 한 장뿐이라, 무엇을 말하고 무엇을 말하지 않는지 함께 본다.*

![Qwen-Image-Bench 비교](figures/qwen_image_2_1_fig1.png)

> 블로그 그림: *Qwen-Image-Bench evaluation comparison* (위 = 총점, 아래 = 파라미터 수, 자물쇠 = 비공개 모델 규모 미공개)

| 순위 | 모델 | 총점 | 비고 |
|---|---|---|---|
| 1 | GPT Image 2.5 Sunburst | 67.01 | 비공개 |
| 4 | Qwen Image 3 Pro | 62.36 | Qwen 내부 비공개 |
| **7** | **Qwen-Image-2.1** | **60.28** | **7B (DiT)** |
| 8 | Nano Banana 2.0 | 59.82 | 비공개 |
| 9 | GPT Image 1.5 | 59.65 | 비공개 |
| 13 | Qwen Image 2.0 Pro | 57.84 | 비공개 |
| — | FLUX 2 Max / Pro | 55.33 / 54.57 | 32B |
| — | Qwen Image 2512 | 52.06 | 20B |
| — | Qwen Image (1.0) | 49.23 | 20B |
| — | HiDream O1 | 46.17 | ≈8B |

- Qwen-Image-Bench (arXiv 2605.28091) 는 **Qwen 이 만든 자사 벤치**.
- 바로 아래 Nano Banana 2.0 과 **0.46점 차**, 신뢰구간 없음.
- 편집 벤치(GEdit·ImgEdit), GenEval·DPG, 속도 수치는 **없음**. (제3자 블로그의 "10장 편집 1.59초"는 공식 본문에서 확인되지 않음)

---

## 8. 💬 Q&A

### Q1. 2.0 기술 리포트와 얼마나 관련 있나? (2.0 코드와 비교해 줘)

*2.0 은 가중치·코드가 끝내 공개되지 않았고, 2.1 은 반대로 가중치·코드는 있지만 리포트가 없다. 둘을 맞대어 서로의 빈칸을 채우려는 질문.*

**먼저 — "2.0 코드"는 존재하지 않는다**
- [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) 저장소의 코드·가중치는 전부 **1.x 계열** (Qwen-Image, Edit-2509/2511, 2512, Layered; 20B dual-stream MMDiT + Qwen2.5-VL + f8 16채널 VAE).
- 2.0 은 2026-02 블로그·Qwen Chat 으로만 나왔고 `huggingface.co/api/models/Qwen/Qwen-Image-2.0` 은 401.
- 그래서 비교는 ① 코드↔코드 (1.x ↔ 2.1, → §5), ② 리포트↔코드 (2.0 리포트 ↔ 2.1) 두 갈래.

**결론**: 2.1 은 **2.0 기술 리포트가 설명한 모델 계열을 처음으로 공개한 구현체**로 보는 게 맞다. 2.0 리포트는 사실상 **2.1 의 빠진 기술 문서**. 다만 Qwen 이 공식적으로 "2.1 은 2.0 기반"이라고 밝힌 적은 없다 — 2.1 블로그·README 는 2.0 을 한 번도 언급하지 않고, 전작으로는 Qwen-Image-Layered 만 언급한다.

**① 결정적 증거: VAE 파라미터 수가 똑같다**

| | encoder | decoder | 설정 |
|---|---|---|---|
| 2.0 리포트 Table 1 | 79M | 259M | f16c64, residual |
| 2.1 가중치 실측 | **78.68M** | **259.04M** | f16c64 (z=64, 16배), `is_residual: true` |

- 디코더를 인코더보다 훨씬 크게 만드는 비대칭 구조까지 같다.
- 같은 표의 다른 f16 VAE 들은 수치가 다르다: Wan2.2(150M/555M), Stepvideo(110M/389M), HunyuanImage-3.0(389M/871M).
- → **2.1 의 VAE 는 2.0 VAE 를 RGBA(4채널) 입출력으로 확장한 것**으로 보는 게 가장 자연스럽다. 채널을 하나 늘려도 첫 conv 와 마지막 conv 만 조금 커져 파라미터 수는 거의 변하지 않는다.

**② 리포트 문장 ↔ 2.1 코드 1:1 대조**

| 2.0 리포트 원문 | 2.1 코드/가중치 | 일치 |
|---|---|---|
| "frozen Qwen3-VL" 조건 인코더 | Qwen3-VL-8B, 추론 시 hidden state 만 사용 | ✅ |
| "visual representation hₓ is **replaced** by the VAE latent" (식 1) | `joint_hidden_states[:, image_pad_mask] = hidden_states` | ✅ 문장 그대로 구현 |
| "h′ = αh", bias 제거한 곱셈 변조 (식 2) | `hidden_states * (1 + scale)`, shift 없음, 전 층 `bias=False` | ✅ |
| SwiGLU (식 3) | `QwenImage21SwiGLUFeedForward` | ✅ |
| QK-Norm 은 RMSNorm, 나머지는 LayerNorm | `norm_q/norm_k = RMSNorm`, `img_norm1/2 = LayerNorm` | ✅ 정규화 종류까지 일치 |
| MSRoPE 합동 위치 계산 | 3축 RoPE (16/56/56), 1.x 설정값과도 같음 | ✅ |
| "**unified stream**", "**shared** transformer backbone" | `txt_mlp` 없음, 모든 토큰이 가중치 공유 | ✅ (이름은 "MMDiT" 지만 실제 single-stream) |
| PE 는 Qwen3.5-9B 에서 초기화 | 공개 PE 2종이 `Qwen3_5ForConditionalGeneration`, 32층, 폭 4096 | ✅ |
| 2.0-RL 리포트: "CFG is also **integrated into the student** after OPD" | `true_cfg_scale` 기본 1.0, "guidance 없이 샘플링하도록 설계" | ✅ CFG 없이 도는 이유 |

→ 2.1 코드는 리포트의 식 1~3 과 Figure 8 캡션의 문장들을 **하나도 빠짐없이** 구현한다.

**③ 2.1 에서 새로 생긴 것 (2.0 리포트에는 없음)**

| 항목 | 2.0 리포트 | 2.1 |
|---|---|---|
| attention 마스크 | 언급 없음 (양방향으로 추정) | **block-causal** |
| 조건 토큰의 timestep | 언급 없음 | **t=0 고정** (`causal_condition`, Edit-2511 `zero_cond_t` 계승) |
| prefix KV cache | 없음 | 있음 |
| modulation 구조 | "bias 제거"만 서술 | **32층 전체가 한 벌 공유** (67M) |
| 투명 배경 | 없음 | VAE RGBA 확장, Layered 기능 흡수 |
| 참조 이미지 수 | "interleaved multi-image" | 최대 10장, 원본+별도 마스크 입력 |
| 4-NFE distillation(증류)판 | 있음 (Qwen-Image-2.0-Distillation) | **공개 안 됨** (40스텝만) |
| 파라미터 수 | 미공개 | 7.12B (DiT) |

변화를 한 줄로: **"2.0 블록 설계는 그대로 두고, 마스크와 시간 조건을 바꿔 LLM 식 캐시가 가능하도록 개조했다."** 블록 내부 가중치 구조는 그대로이고 attention 이 서로를 보는 규칙만 바뀌었다. 그래서 2.0 체크포인트를 이어서 학습했을 가능성도 있다 (근거는 없음).

벤치마크에서 **2.1(60.28) > 2.0 Pro(57.84)** → "2.0 설계 계열의 후속 체크포인트"로 보는 게 맞다.

**④ 관련도 판정**

| 관점 | 관련도 |
|---|---|
| 구조 (블록, 인코더, VAE, PE) | **매우 높음.** 서술과 코드가 1:1 대응, VAE 파라미터 수 일치 |
| 학습 레시피 (Pretrain → SFT → GRPO → OPD/CFG 통합) | **높음 (추정).** 2.1 은 레시피 미공개지만 CFG 없는 기본값이 2.0-RL 서술과 맞음 |
| 추론 방식 | **다름.** block-causal 과 KV cache 는 2.1 신규 |
| 공식 연결 | **없음.** Qwen 은 2.1 을 2.0 과 연결 짓는 말을 하지 않음 |

2.1 을 공부할 때 역할 분담:
- **2.0 리포트(+ 2.0-RL 리포트)**: 왜 이렇게 만들었는지, 어떻게 학습했는지 → [PAPER_Qwen-Image-2.0.md](PAPER_Qwen-Image-2.0.md), [PAPER_Qwen-Image-2.0-RL.md](PAPER_Qwen-Image-2.0-RL.md)
- **2.1 코드(이 문서)**: 실제로 어떻게 도는지, 새로 추가된 마스크와 캐시

(이 대조로 2.0 문서의 §4.2·§4.3·Q1·Q2 일부를 정정했다 — 정정 목록은 2.0 문서 Q6.)

### Q2. 쓰기 전에 알아야 할 급소는?

*공식 문서만 믿고 쓰면 막히거나 오해하는 지점들.*

| # | 급소 | 내용 |
|---|---|---|
| 1 | **벤치마크가 자사 벤치 하나** | Qwen-Image-Bench 7위, Nano Banana 2.0 과 0.46점 차, 신뢰구간 없음. "대부분의 비공개 모델 능가"는 이 차이에 기댐. 편집을 핵심 기능으로 내세우면서 편집 벤치 수치 0개 |
| 2 | **파라미터 막대 기준이 섞임** | 2.1(7B)·Qwen-Image(20B)·LongCat(6B)은 DiT 만, HiDream-O1(≈8B)은 인코더 포함 통합 모델. 2.1 도 인코더 포함하면 16.2B |
| 3 | **README 와 코드의 기본 해상도가 다름** | README 표 "기본 2048×2048" ↔ 파이프라인 `output_resolution=1024` → width/height 안 주면 **1024²**. Quick Start 예제도 1024² 로 생성. 2K 는 명시해야 함 |
| 4 | **PE 코드가 README 대로 안 돎 (재현 확인)** | README 는 `--ckpt Qwen/Qwen-Image-2.1-PE-T2I` (Hub ID) 를 안내하지만 `load_system_prompt` 가 이를 로컬 경로로 보고 `SystemExit: no system prompt` 로 종료. 우회: `--system-prompt prompts/system_prompt_t2i.txt` (HF 파일과 바이트 동일 확인). 메인 README 의 `client.py` 예시도 `--system-prompt` 누락으로 같은 이유로 실패 |
| 5 | **같은 환경에 설치 불가** | 메인 모델 `transformers>=5.17` ↔ PE `transformers==5.4.0` 고정 → 가상환경 2개 필요 |
| 6 | **배치 ≠ 단일 추론** | 길이가 다른 프롬프트를 한 배치로 돌리면 짧은 프롬프트 뒤 padding 칸도 RoPE 위치를 차지 → 타깃 이미지 위치가 단독 실행과 달라짐 |
| 7 | **라이선스 후퇴** | 1.x 전 라인은 Apache 2.0 → 2.1·PE 는 "FOR NON-COMMERCIAL PURPOSES ONLY", 상업 사용은 별도 계약 (model-business@notice.qwencloud.com) |
| 8 | **출력이 항상 RGBA** | VAE 가 늘 4채널을 내므로 불투명 이미지도 PIL `RGBA` → `.save("x.jpg")` 가 "cannot write mode RGBA as JPEG" 에러. `.convert("RGB")` 또는 PNG 저장 (→ Q3 ⑦) |
| 9 | **VAE 죽은 가중치** | 영상용 `time_conv` 12개 = 6.75M (VAE 의 2%) 가 이미지 경로에서 한 번도 안 쓰임 (→ Q3 ③) |
| 10 | **2K 에서 shift 외삽** | `calculate_shift` 가 clamp 없는 선형식이라 2048²(16,384 토큰)에서 mu ≈ 1.31 — config 의 "max_shift 0.9 @ 8,192" 를 넘음 (→ Q3 ⑥) |

추가로:
- PE 는 thinking 모델이라 `max_new_tokens` 가 T2I 16,256 / 편집 24,000 — 재작성 한 번이 길다. T2I 는 `presence_penalty=1.5`, 편집은 0 (서로 바꾸면 조용히 분포가 달라짐).
- `use_kv_cache` 토글 시 같은 시드도 다른 그림 (→ §6).

### Q3. 전체 코드를 보고 — 전체 네트워크 구조와 데이터 flow 는?

*§4 는 부품별 "무엇이 있나"를 정리했고, 이 답은 입력 한 건이 파이프라인·Transformer·VAE 코드를 차례로 지나며 텐서 모양이 어떻게 바뀌는지를 끝까지 따라간다.* 근거는 diffusers 의 `pipeline_qwenimage21.py` · `transformer_qwenimage21.py` · `autoencoder_kl_qwenimage21.py` 전부와 가중치 헤더 실측이다. 예시는 **참조 이미지 1장(1024²)으로 1024² 결과를 만드는 편집** 기준.

#### ⓪ 전체 그림 한 장

```
 [사용자 입력] prompt + (참조 이미지 0~10장)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 ① 전처리: 면적 1024², 32배수로 리사이즈
   ├─ VLM용 사본: RGBA → 흰 배경 합성 → RGB
   └─ VAE용 사본: RGBA 4채널, [-1,1]
        │                              │
        ▼                              ▼
 ② Qwen3-VL-8B (동결)          ③ VAE 인코더 (f16c64)
   시스템+이미지+지시문 →        4ch 이미지 → 64ch 잠재
   마지막 층 hidden(정규화 전)   → 채널별 mean/std 정규화
   [L, 4096]                      [4096 토큰, 64]
        │                              │
        └──────────┬───────────────────┘
                   ▼          ④ 노이즈 잠재 [4096 토큰, 64]
 ⑤ DiT 7B (32층 single-stream) ◀──────┘
   step 0 : 전체 계산 + prefix K/V 저장 (prefill)
   step 1~39: 타깃 토큰만 계산 (decode)
   출력 = velocity [4096, 64] → Euler 한 걸음
                   ▼
 ⑥ 정규화 해제 → VAE 디코더 → [-1,1] clamp → RGBA 이미지
```

부품별 규모(실측): Qwen3-VL-8B 8.77B, DiT 7.12B, VAE 0.34B.

#### ① 전처리 (`__call__` 1단계)

- 참조 이미지는 원본 비율을 유지한 채 **면적 1024²(`output_resolution`), 32 의 배수**로 리사이즈 (`calculate_dimensions`).
- 같은 리사이즈 결과에서 두 갈래로 나뉜다.
  - **VLM용**: RGBA 라면 알파를 흰 배경 위에 합성해 RGB 로. 코드 주석: "체크포인트가 그렇게 학습됐다".
  - **VAE용**: RGBA 4채널을 그대로 [-1,1] 로 정규화. 투명도 정보는 이쪽으로만 들어간다.
- 결과 크기는 width/height 를 주면 그 값, 안 주면 마지막 참조 이미지 비율로 1024² 면적. T2I 는 1024² 가 기본.

#### ② text/vision encoder: Qwen3-VL-8B (`_get_qwen_prompt_embeds`)

**프롬프트 템플릿** — chat template 을 거치지 않고 문자열을 직접 만든다. 코드 주석에 따르면 두 방식은 토큰화 결과가 달라서, 체크포인트가 기대하는 쪽이 이 직접 만든 문자열이다.

```
<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> <image2>...지시문<|im_end|>\n
<|im_start|>assistant\n
```

- 이미지가 지시문보다 **앞**에 온다. 그래야 지시문 토큰이 이미지를 보고 이해할 수 있다 (→ ④ ③ 에서 중요).

**Qwen3-VL 내부 구조 (config 기준)**

| 부분 | 구조 |
|---|---|
| ViT | 27층, 폭 1152, patch 16. 2×2 merge 후 **토큰 1개 = 32×32px**. DeepStack 으로 ViT 8·16·24층 feature 를 LLM 앞쪽 층에 더해 줌 |
| LLM | 36층, 폭 4096, GQA(32 query 헤드 / 8 KV 헤드), Interleaved-MRoPE |
| 예시 크기 | 참조 1024² → 64×64 패치 → merge → **1,024개 `<|image_pad|>` 토큰** |

**출력 추출**
- **마지막 층의 정규화(RMSNorm) 전** hidden state 를 쓴다. transformers 5.x 는 정규화 후 값을 돌려주므로 forward hook 으로 norm 층을 항등 함수로 바꿔 둔다.
- padding(왼쪽 padding)을 제거하고, 앞쪽 시스템 프롬프트 토큰(`_drop_idx`개)을 잘라 낸다.
- 결과 세 가지:
  - `prompt_embeds` [B, L, 4096] — 예시 기준 L ≈ 1,024 이미지 칸 + 약 40 텍스트 토큰
  - `prompt_embeds_mask` — 유효 토큰 표시
  - `image_pad_mask` — 어느 위치가 이미지 칸인지 표시

#### ③ VAE encoder (`autoencoder_kl_qwenimage21.py`)

**구조: Wan 2.x 의 3D VAE 골격을 이미지 전용으로 특수화한 것**
- `CausalConv3d` 가 실제로는 **`nn.Conv2d` 를 상속**한다. 시간 축을 squeeze 한 뒤 2D conv 만 수행하므로 이름만 3D.
- 인코더 채널 96 → 96 → 192 → 384 → 768 → 768, 2배 다운샘플 4번 = **16배 압축**.
- 디코더는 base 144 로 더 넓다 (79M vs 259M 비대칭).
- **residual 경로**(`is_residual=True`): 블록마다 주 경로(ResBlock ×2 + 다운샘플)에 `AvgDown3D` 를 더한다. `AvgDown3D` = 2×2 픽셀을 채널로 접은 뒤 채널 그룹 평균을 내는 **파라미터 없는 지름길**. 2.0 리포트의 "non-parametric shortcut" 이 바로 이것.
- 인코더 출력 128채널(평균 64 + 분산 64), 파이프라인은 샘플링 없이 **평균(mode)** 만 쓴다.

```
[1, 4, 1, 1024, 1024]  (RGBA, [-1,1])
 → 인코더 → [1, 128, 1, 64, 64] → 평균만 사용 → [1, 64, 1, 64, 64]
 → (z − latents_mean[c]) / latents_std[c]   ← 채널별 64개 상수 (config)
 → pack: [1, 4096, 64]   (patch_size=1 → 64×64 칸을 그대로 펼침)
```

- ⚠️ **죽은 가중치**: 영상용 `time_conv` 12개(**6.75M, VAE 의 2%**)가 체크포인트에 들어 있다. 한 장짜리 이미지는 코드상 첫 청크 경로("Rep")만 타기 때문에 한 번도 쓰이지 않는다.

#### ④ DiT 입력 조립 — 가장 헷갈리는 부분 (`transformer.forward` 앞부분)

**① 입구 투영**
- `img_in`: Linear 64 → 4096. 참조 잠재와 노이즈 잠재 모두 통과.
- `txt_in`: VLM 출력에 적용 (ZeroCenter RMSNorm → Linear 4096 → GELU → Linear 4096).
  - ZeroCenter RMSNorm = 가중치를 "scale − 1" 로 저장하고 계산할 때 +1 하는 RMSNorm.

**② 타깃 자리 만들기** — 파이프라인이 `image_pad_mask` 끝에 **타깃 칸 4096/4 = 1024개**를 True 로 덧붙이고, Transformer 는 텍스트 끝에 0벡터 1024개를 붙인다.

**③ 칸 1개 → 토큰 4개로 늘리고 덮어쓰기**

```python
repeats = torch.where(img_mask, 4, 1)[0]            # 이미지 칸만 ×4
joint = cat([txt, zeros(1024)]).repeat_interleave(repeats)
joint[:, image_pad_mask] = hidden_states           # 이미지 자리 = VAE 잠재(참조→타깃 순)
```

- VLM 이미지 칸 1개(32px) = VAE 토큰 2×2개(16px 네 개)와 정확히 대응.
- **VLM 이 만든 이미지 칸 벡터는 여기서 버려진다.** VLM 이 이미지를 이해한 결과는 뒤따르는 지시문 토큰에만 남는다 (2.0 리포트 Fig 8 의 빨간 X).

**완성된 시퀀스 (예시)**
```
[user 태그 ~5] [참조 이미지 4096] [vision_end + 지시문 ~30 + assistant 태그] [타깃 4096]
 ←──────────── prefix ≈ 4.1K (텍스트 + 조건) ─────────────────→ ← target 4096 →
                       총 약 8.2K 토큰 × 4096차원
```

**④ 3축 RoPE 좌표** (`QwenImage21Rope`, frame 16 + height 56 + width 56 = 128차원, theta 10000)

| 토큰 | frame 축 | height·width 축 |
|---|---|---|
| 텍스트 | 1, 2, 3, … (한 칸씩 전진) | frame 과 같은 값 |
| 이미지 블록 | 직전 텍스트 위치에 **고정** | **0 중심 격자** (예: −32 … 31) |
| 이미지 다음 텍스트 | 이미지 블록 이후 max(h, w) 만큼 건너뛰고 이어짐 | frame 과 같은 값 |

- 참조 이미지와 타깃이 **같은 공간 좌표를 공유** → "원본의 이 위치 = 결과의 이 위치"가 위치 인코딩 수준에서 맞춰진다.

**⑤ 블록 id 와 타깃 마스크** (`build_token_metadata`): 텍스트 −1, 이미지마다 고유 번호, 마지막 블록 = 타깃 (→ Q6 ①).

**⑥ 시간 조건**
```
t (0~1, 1 = 순수 노이즈)  ─┐
t = 0 행 하나 추가  ────────┴→ ×1000 → sinusoid 256차원 (cos 먼저, sin 나중)
   → Linear 256→4096 → SiLU → Linear 4096→4096 = temb [B+1, 4096]
   → modulation: SiLU → Linear 4096→16384 = [scale1 | gate1 | scale2 | gate2]
```

- 타깃 토큰은 실제 t 행을, **텍스트와 참조 이미지 토큰은 t=0 행**을 읽는다 (`causal_condition`, → Q5).

#### ⑤ Transformer block ×32 (전 층 동일, modulation 도 공유)

```python
# 모든 블록이 같은 modulation 텐서를 받음 (블록별 파라미터 없음)
scale1, gate1, scale2, gate2 = modulation.chunk(4)   # 토큰 종류별로 t 행 또는 t=0 행 선택

h = LayerNorm(x)                      # affine 없음
h = h * (1 + scale1)                  # 곱셈만 (shift 없음)
q, k, v = Wq·h, Wk·h, Wv·h            # 4096 → 32헤드 × 128, bias 없음
q, k = RMSNorm(q), RMSNorm(k)         # 헤드별 QK-Norm
q, k = RoPE(q), RoPE(k)
a = Attention(q, k, v, mask = block-causal)
x = x + tanh(gate1) * Wo·a

h = LayerNorm(x) * (1 + scale2)
x = x + tanh(gate2) * W_out( SiLU(W_gate·h) ⊙ W_proj·h )   # SwiGLU, 4096→12288→4096
```

**attention mask 규칙**: `(q 위치 ≥ kv 위치) 또는 (같은 이미지 블록)`, 그리고 padding 키 제외 (의미 → Q4, 코드 → Q6).

| query \ key | 앞쪽 텍스트 | 참조 이미지 | 지시문 | 타깃 |
|---|---|---|---|---|
| 앞쪽 텍스트 | causal | ✗ | ✗ | ✗ |
| 참조 이미지 | ✓ | **양방향** | ✗ | ✗ |
| 지시문 | ✓ | ✓ | causal | ✗ |
| 타깃 | ✓ | ✓ | ✓ | **양방향** |

- 구현 두 가지:
  - 기본 `QwenImage21AttnProcessor`: prefix 를 구간(segment)별로 나눠 SDPA 를 여러 번 호출. 정확하지만 느림.
  - `QwenImage21FlexAttnProcessor`: flex_attention 의 BlockMask 로 한 번에 계산. **컴파일해야** 빠르고, 안 하면 fp32 점수 행렬을 통째로 만들어 고해상도에서 메모리 부족.

**출구**: `norm_out`(LayerNorm × (1 + scale), scale 은 temb 에서 별도 Linear) → `proj_out` Linear 4096→64 → 토큰마다 velocity 64차원.

#### ⑥ denoising loop — prefill 과 decode

**스케줄 준비**
- sigma = 1.0 부터 1/40 까지 40등분.
- **토큰 수에 따른 shift**: mu = 0.5 + 0.4 × (타깃 토큰 수 − 256) / 7936
  - 1024²(4,096 토큰) → mu ≈ 0.69
  - 2048²(16,384 토큰) → mu ≈ 1.31. config 의 "max_shift 0.9 @ 8,192 토큰"을 넘어 **선형 외삽** (clamp 없음).
- exponential shift 적용 후, 마지막 sigma 가 0.02 가 되도록 늘림 (`shift_terminal`).

**루프**
```
step 0  (kv_mode="extract"):
   시퀀스 8.2K 전체 계산 → 32층 각각 prefix(≈4.1K) K/V 복제·저장
step 1~39 (kv_mode="cached"):
   입력 = 타깃 4096 토큰만 → 층마다 K,V = [저장된 prefix K/V ; 타깃 K/V]
   타깃 query 4096개가 prefix + 자기 블록 전체를 봄 (마스크 불필요)
매 step:
   v = 출력의 마지막 4096 토큰 (velocity)
   latents ← latents + (σ_next − σ) × v      (Euler)
   CFG: 기본 없음 (true_cfg_scale = 1.0 → 1 step = 1회 호출)
```

- prefix 는 t=0 으로 modulation 되고 마스크 때문에 타깃을 보지 못한다 → **첫 스텝의 K/V 가 39스텝 내내 정확히 유효**. 이것이 전체 설계의 핵심 (→ Q5).

#### ⑦ decoding 과 후처리

```
latents [1, 4096, 64]
 → unpack [1, 64, 1, 64, 64]
 → × latents_std + latents_mean
 → VAE 디코더 (post_quant_conv → mid → up ×4 (DupUp3D 지름길) → conv_out 4채널)
 → clamp [-1, 1] → [0, 1] → PIL
```

- ⚠️ 출력은 **항상 4채널 → PIL `RGBA` 모드**. README 는 "투명일 때만 RGBA 로 저장"이라 하지만 불투명 이미지도 알파 ≈ 255 인 RGBA → `.save("x.jpg")` 는 **"cannot write mode RGBA as JPEG" 에러**. `.convert("RGB")` 를 거치거나 PNG 로 저장.

#### ⑧ 한눈에 보는 텐서 모양 (참조 1장, 1024² → 1024²)

| 단계 | 모양 |
|---|---|
| VLM 입력 토큰 (시스템 포함) | 약 1,080 (이미지 칸 1,024) |
| VLM 출력 (시스템 제거 후) | [1, 약 1,060, 4096] |
| 참조 VAE 잠재 | [1, 4096, 64] |
| 노이즈 잠재 | [1, 4096, 64] |
| DiT 결합 시퀀스 (step 0) | [1, 약 8,230, 4096] |
| KV cache (층당) | K, V 각 [1, 약 4,130, 32, 128] → 32층 합계 약 2.2GB |
| DiT 입력 (step 1~39) | [1, 4096, 4096] |
| DiT 출력 | [1, 4096, 64] (velocity) |
| 최종 이미지 | [1, 4, 1024, 1024] → RGBA PIL |

T2I 라면 참조 이미지와 VLM 이미지 칸이 없어 prefix 는 텍스트 수십~수백 토큰뿐 → 캐시 효과는 거의 없고, 시퀀스는 거의 전부 타깃 토큰.

> 한 줄: **"Qwen3-VL 이 지시문을 읽고(이미지 칸은 버림) → VAE 잠재를 그 빈자리에 끼워 한 줄로 만들고 → t=0 prefix 와 block-causal mask 덕분에 첫 스텝에서 조건을 한 번만 계산해 캐시한 뒤 → 32층 공유 modulation DiT 가 타깃 4096 토큰의 velocity 만 39번 더 계산해 Euler 로 내려가고 → f16c64 RGBA VAE 가 복원한다."** 코드에서 새로 발견한 것: VAE 의 죽은 `time_conv` 6.75M, **항상 RGBA 로 나오는 출력(JPEG 저장 에러)**, 2K 에서 shift 외삽.

### Q4. block-causal ???

*§4.2 그림과 Q3 ⑤ 에 규칙만 나오고 "왜 이런 모양인가"가 없어서, 규칙 자체를 처음부터 풀어 보는 질문.*

**block-causal** = **"줄 단위로는 앞만 보고, 이미지 한 장 안에서는 전부 본다"** 는 attention 규칙. 익숙한 두 규칙을 섞은 것.

**① 두 가지 기본 규칙**

| 규칙 | 뜻 | 쓰는 곳 |
|---|---|---|
| **full attention** (양방향) | 모든 토큰이 모든 토큰을 봄 | 일반 DiT, Qwen-Image 1.x |
| **causal attention** | 각 토큰이 **자기보다 앞**에 있는 토큰만 봄 | LLM (GPT, Qwen 등) |

block-causal 은 이 둘을 섞는다.
- **기본은 causal**: 시퀀스에서 앞에 있는 것만 본다.
- **예외로 같은 블록 안은 full**: "블록" = 이미지 한 장. 한 장 안의 토큰끼리는 앞뒤 구분 없이 서로 다 본다.

```python
볼 수 있음 = (내 위치 >= 상대 위치) or (둘이 같은 이미지)
```

**② 작은 예시** — 시퀀스 `[텍스트 T1 T2] [참조 이미지 A1 A2] [지시문 T3] [타깃 이미지 G1 G2]`

| 보는 쪽 ↓ \ 보이는 쪽 → | T1 | T2 | A1 | A2 | T3 | G1 | G2 |
|---|---|---|---|---|---|---|---|
| T1 | ✓ | | | | | | |
| T2 | ✓ | ✓ | | | | | |
| A1 | ✓ | ✓ | ✓ | **✓** | | | |
| A2 | ✓ | ✓ | ✓ | ✓ | | | |
| T3 | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| G1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| G2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

- 텍스트(T)는 계단 모양 — LLM 과 똑같이 앞만 본다.
- **굵은 ✓** 가 예외: A1 은 순서상 뒤의 A2 도 본다 (같은 이미지).
- 오른쪽 위는 전부 비어 있다 → **앞쪽 토큰은 뒤에 오는 타깃(G)을 절대 보지 못한다.**
- 맨 아래 G 줄은 모든 칸이 참 → **타깃 입장에서는 full attention 과 똑같다.**

비유: **책 읽기**. 앞 페이지는 다시 볼 수 있지만 아직 안 넘긴 뒤 페이지는 못 본다(causal). 다만 한 페이지에 그림이 있으면 그 그림은 한눈에 전체를 본다(블록 안 full).

**③ 왜 이렇게 만들었나**

- **이미지까지 전부 causal 이면?** 안 된다. 이미지 토큰에는 "먼저·나중"이라는 자연스러운 순서가 없다. 왼쪽 위 조각이 오른쪽 아래 조각을 못 보면 그림 전체가 어긋난다 → 이미지 한 장 안은 양방향.
- **1.x 처럼 전부 full 이면?** 품질 면에선 문제없지만 **캐시가 불가능**해진다.
  - 타깃(G)은 노이즈가 걷히면서 **매 스텝 값이 바뀐다**.
  - full 이면 앞쪽 텍스트·참조 이미지도 G 를 보므로, G 가 바뀔 때마다 앞쪽 계산 결과도 달라진다 → 40스텝 내내 전부 재계산.
  - block-causal 이면 앞쪽은 G 를 안 보므로 **앞쪽 계산 결과가 스텝마다 같다** → 첫 스텝에 한 번 계산해 저장(KV cache), 나머지 39스텝은 G 만 계산.
- 여기에 조건 토큰을 **t=0 으로 고정**하는 설정(`causal_condition`)이 함께 들어간다. 마스크가 "G 를 못 보게", t=0 고정이 "노이즈 수준이 바뀌어도 영향받지 않게" 막는다. 둘 다 있어야 앞쪽 결과가 완전히 고정된다 (→ Q5).

**④ 대가는?**
- **타깃 품질 영향은 없다** — 타깃 줄은 full attention 과 똑같이 모든 것을 본다.
- 달라지는 건 **앞쪽(조건) 토큰**. 1.x 에서는 참조 이미지가 타깃을 보며 서로 맞춰 갈 수 있었지만, 2.1 에서는 조건이 일방적으로 정보를 주기만 한다.
- 이 차이가 품질을 얼마나 떨어뜨리는지는 **ablation 이 공개되지 않아 알 수 없다.**

> 한 줄: **block-causal = LLM 처럼 앞만 보되 이미지 한 장 안에서는 서로 다 보는 규칙. 앞쪽 조건이 뒤쪽 타깃을 못 보게 막아서, 조건을 한 번만 계산하고 재사용(KV cache)할 수 있게 해 준다.**

### Q5. causal_condition ?? prefix KV cache ?? — 이 3개는 뭐야?

*block-causal · causal_condition · prefix KV cache 가 각각 따로 등장해 관계가 헷갈리므로, 셋을 한 세트로 묶어 푸는 질문.*

세 가지는 따로 있는 기능이 아니라 **한 세트**다. 목표는 **prefix KV cache**(속도)이고, 나머지 둘은 그 캐시가 성립하기 위한 **전제 조건**.

- **block-causal**: 앞쪽이 뒤쪽을 보지 않게 막음 (공간 쪽 조건)
- **causal_condition**: 앞쪽이 스텝마다 흔들리지 않게 막음 (시간 쪽 조건)
- **prefix KV cache**: 두 조건이 갖춰지면 앞쪽을 한 번만 계산하고 재사용

#### ① prefix KV cache — 목표

**KV 가 뭔가**
- attention 에서 토큰마다 세 가지를 만든다.
  - Q(query): 내가 무엇을 찾는지
  - K(key): 나를 찾을 때 쓰는 색인
  - V(value): 내가 건네줄 내용
- 다른 토큰이 나를 참고할 때 쓰는 건 **내 K 와 V 뿐**.
- 그래서 어떤 토큰의 K, V 가 변하지 않는다면 한 번 계산해 저장해 두고 다시 꺼내 쓰면 된다 = KV cache. LLM 이 다음 글자를 빠르게 뽑을 때 쓰는 방법과 같다.

**prefix 가 뭔가** — 시퀀스의 **앞부분** [시스템 프롬프트 + 참조 이미지 + 지시문]. 뒷부분은 만들어 낼 **타깃 이미지**.

**diffusion 에서의 기회**
- 이미지 한 장을 만들려고 같은 DiT 를 **40번** 호출한다.
- prefix 의 입력(지시문, 참조 이미지)은 40번 내내 **똑같다**. 바뀌는 건 노이즈가 걷히는 타깃뿐.
- → prefix 의 K, V 를 첫 스텝에 저장해 두고, 나머지 39스텝은 타깃만 계산.

```
step 0   : [prefix + 타깃] 전부 계산 → 32층마다 prefix K/V 저장   ("extract")
step 1~39: [타깃]만 계산 → K,V = [저장된 prefix K/V ; 타깃 K/V]    ("cached")
```

**문제**: 원래 DiT 에서는 입력이 같아도 prefix 의 K, V 가 스텝마다 **달라진다**. 이유가 두 가지이고, 각각을 나머지 두 장치가 막는다.

#### ② 이유 1: prefix 가 타깃을 본다 → block-causal 이 막음

- full attention(1.x)이면 prefix 토큰도 타깃 토큰을 본다.
- 타깃은 매 스텝 바뀌므로, 그걸 본 prefix 의 계산 결과도 매번 바뀐다.
- block-causal 은 **앞쪽이 뒤쪽을 못 보게** 막는다 (이미지 한 장 안만 예외) → prefix 는 타깃이 어떻게 변하든 영향 없음. 규칙 상세는 → Q4.

#### ③ 이유 2: prefix 도 timestep 을 받는다 → causal_condition 이 막음

**왜 문제인가** — DiT 의 모든 토큰은 블록마다 modulation 을 거친다.

```
h = LayerNorm(x) × (1 + scale(t))
x = x + tanh(gate(t)) × Attention(h)
```

- scale 과 gate 는 **t(현재 노이즈 수준)로 계산**된다. t 는 스텝마다 1.0 → 0.02 로 줄어든다.
- 마스크로 타깃을 못 보게 막아도, prefix 토큰이 **진짜 t 로 modulation 되면** 입력이 같아도 출력이 매번 달라진다 → 캐시 무효.

**해결: 조건 토큰은 t 를 0 으로 고정**

```python
# transformer_qwenimage21.py
timestep = torch.cat([timestep, timestep.new_zeros(1)])   # [진짜 t, 0] 두 행
modulation = self.modulation(time_embed(timestep))        # modulation 도 두 벌

# _select_modulation_rows
타깃 토큰        → 진짜 t 행
텍스트·참조 토큰 → t=0 행   (스텝이 바뀌어도 항상 같은 값)
```

- 이제 prefix 토큰의 modulation 은 40스텝 내내 **상수**. 출구의 `norm_out` 도 같은 방식으로 나뉜다.

**왜 하필 0 인가**
- 이 모델의 규약에서 t=1 은 순수 노이즈, **t=0 은 노이즈가 전혀 없는 깨끗한 상태**.
- 참조 이미지 잠재는 실제로 노이즈를 섞지 않은 깨끗한 값이라 "t=0" 은 **사실 그대로의 라벨**.
- 텍스트에는 노이즈 개념이 없으니 그냥 고정값 하나를 주는 것.

**이름 주의**: `causal_condition` 이라는 이름과 달리 **마스크와는 상관이 없다.** 실제 내용은 "조건 토큰을 t=0 으로 modulation 한다"이고, docstring 도 "Modulate text and condition-image tokens from t = 0. Required for KV caching." 이라고 적혀 있다.

**계보: 1.x Edit-2511 의 `zero_cond_t`**

| | Edit-2511 (`zero_cond_t`) | 2.1 (`causal_condition`) |
|---|---|---|
| 참조 이미지 | t=0 | t=0 |
| 텍스트 | **진짜 t** (텍스트 전용 가중치 `txt_mod` 가 따로 있음) | **t=0** |
| attention | full (참조가 타깃을 봄) | block-causal |
| 캐시 가능? | ✗ | ✓ |

2511 에서는 "참조 이미지는 깨끗하니 t=0 으로 표시하자"라는 **품질 목적**의 장치였다. 2.1 은 이를 텍스트까지 넓히고 마스크를 더해 **속도 목적**(캐시)까지 달성했다.

#### ④ 세 개 정리

| 장치 | 무엇을 하나 | 막는 것 | 코드 |
|---|---|---|---|
| **block-causal** | 앞쪽은 뒤쪽을 못 보게 (이미지 한 장 안은 양방향) | 타깃 변화가 prefix 로 번지는 것 | `build_qwenimage21_block_causal_mask` |
| **causal_condition** | 조건 토큰의 modulation 을 t=0 으로 고정 | 스텝(t) 변화가 prefix 를 흔드는 것 | `_select_modulation_rows` |
| **prefix KV cache** | 첫 스텝에 prefix K/V 저장 → 이후 타깃만 계산 | (앞 두 장치 덕분에 성립) | `QwenImage21KVCache`, `kv_cache_mode` |

- 코드도 이 의존관계를 강제한다: `causal_condition=False` 인데 캐시를 켜면 `ValueError`.

**비유 (요리)**: 재료 손질(prefix)은 한 번만 해 두고, 굽기(타깃)만 40번 반복. 그러려면 두 가지가 보장돼야 한다.
- 굽는 냄비 상태를 보고 재료 손질이 달라지면 안 된다 → block-causal
- 불 세기(t)가 바뀔 때마다 손질한 재료가 변하면 안 된다 → causal_condition

#### ⑤ 효과와 대가

효과 수치(T2I 약 1%, 참조 1장 약 2배, 참조 10장 약 11배 / 캐시 메모리 약 22GB)는 → §6.

**대가**: 조건 토큰이 타깃을 보면서 맞춰 갈 수 없고, 노이즈 수준에 따라 조건을 다르게 읽는 것도 불가능해진다. 이 제약이 품질에 주는 영향은 ablation 이 없어 알 수 없다.

> 한 줄: **prefix KV cache = "조건은 40스텝 내내 같으니 한 번만 계산하자"는 목표. 그게 성립하려면 조건이 타깃을 안 봐야 하고(block-causal), 노이즈 수준에도 안 흔들려야 한다(causal_condition, 조건 토큰은 t=0 고정).**

### Q6. "줄 단위로는 앞만 보고, 이미지 한 장 안에서는 전부 본다" — 이 의미와 매칭되는 실제 코드는?

*Q4 의 말로 된 규칙이 코드 어디에 어떻게 박혀 있는지 줄 단위로 확인하려는 질문.*

코드에서는 이 규칙이 **세 곳**에 구현돼 있다. 모두 diffusers 의 [transformer_qwenimage21.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py) 이고, 줄 번호는 2026-09 main 브랜치 기준.

1. 토큰마다 "어느 이미지 소속인가" 번호 붙이기 (준비 단계)
2. **flex 경로**: 규칙을 한 줄 식으로 직접 쓴 곳 → 의미가 가장 잘 보임
3. **기본(SDPA) 경로**: 같은 규칙을 구간을 나눠 여러 번 계산하는 곳 → 실제로 기본값으로 도는 코드

예시 시퀀스 하나로 끝까지 따라간다.

```
위치:      0  1 | 2  3  4  5 | 6 | 7  8  9  10
토큰:      T  T | A  A  A  A | T | G  G  G  G
           텍스트  참조 이미지   지시문  타깃 이미지
```

#### ① 준비: 토큰마다 소속 번호 붙이기 (`build_token_metadata`, 827~844줄)

```python
image_positions = image_pad_mask.nonzero(as_tuple=True)[0]      # 이미지 토큰 위치들
block_lengths = [math.prod(shape) for shape in img_shapes]       # 이미지별 토큰 수 [4, 4]

image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)   # 전부 -1 (= 텍스트)로 시작
block_ids = torch.repeat_interleave(
    torch.arange(len(block_lengths)),                            # 0, 1, ...
    torch.tensor(block_lengths),                                 # 각 이미지 길이만큼 반복
)
image_ids[image_positions] = block_ids                           # 이미지 자리에 0,0,0,0,1,1,1,1

target_token_mask[image_positions[-block_lengths[-1]:]] = True   # 마지막 이미지 = 타깃
```

예시 적용 결과:

```
위치:      0   1 | 2  3  4  5 | 6  | 7  8  9  10
image_ids: -1  -1 | 0  0  0  0 | -1 | 1  1  1  1
```

- **텍스트 = −1, 이미지 = 0, 1, 2, … (한 장마다 고유 번호)**. "이미지 한 장"을 가리는 기준이 이 번호.
- 이미지 경계는 `True` 가 이어진 구간이 아니라 `img_shapes` 의 길이로 자른다. 코드 주석에 따르면, 참조 이미지 두 장이 사이에 텍스트 없이 붙어 있어도 **서로 다른 블록**으로 남게 하려는 것. 그렇지 않으면 두 장이 서로 양방향으로 보게 된다.

#### ② flex 경로: 규칙이 그대로 적힌 곳 (`build_qwenimage21_block_causal_mask`, 291~296줄)

```python
def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
    is_padding = (q_idx >= seq_len) | (kv_idx >= seq_len)
    q_image_id, kv_image_id = image_ids[q_idx], image_ids[kv_idx]
    same_image_block = (q_image_id == kv_image_id) & (q_image_id >= 0)             # ← "이미지 한 장 안에서는 전부"
    allowed = ((q_idx >= kv_idx) | same_image_block) & key_valid[batch_idx, kv_idx]
    #          ^^^^^^^^^^^^^^^^^ ← "줄 단위로는 앞만"                ^^^^^^^^^ padding 키 제외
    return allowed & ~is_padding
```

| 말 | 코드 조각 | 뜻 |
|---|---|---|
| 줄 단위로는 앞만 본다 | `q_idx >= kv_idx` | 나(q)보다 앞이거나 같은 위치의 토큰(kv)만 허용 |
| 이미지 한 장 안에서는 전부 본다 | `q_image_id == kv_image_id` | 같은 이미지 번호면 위치 순서와 상관없이 허용 |
| 단, 텍스트는 예외에서 제외 | `& (q_image_id >= 0)` | 텍스트끼리는 둘 다 −1 이라 "같은 번호"가 되는데, 이걸 막아서 텍스트는 순수 causal 로 남김 |
| 둘 중 하나면 OK | `\|` (or) | 앞이거나 같은 이미지면 볼 수 있음 |
| 빈칸은 안 봄 | `key_valid[...]` | 프롬프트 padding 자리는 키로 쓰지 않음 |

**예시로 직접 계산해 보기**

| 질문 | q → kv | `q>=kv` | 같은 이미지 | 결과 |
|---|---|---|---|---|
| 참조 첫 토큰이 참조 마지막 토큰을 보나? | 2 → 5 | ✗ | ✓ (0 == 0) | **봄** (한 장 안은 전부) |
| 지시문이 참조 이미지를 보나? | 6 → 3 | ✓ | ✗ | **봄** (앞이니까) |
| 첫 텍스트가 둘째 텍스트를 보나? | 0 → 1 | ✗ | ✗ (−1 이라 제외) | **못 봄** (텍스트는 causal) |
| 참조 이미지가 타깃을 보나? | 4 → 8 | ✗ | ✗ (0 ≠ 1) | **못 봄** → 이게 캐시의 전제 |
| 타깃이 지시문을 보나? | 9 → 6 | ✓ | ✗ | **봄** |
| 타깃 첫 토큰이 타깃 마지막 토큰을 보나? | 7 → 10 | ✗ | ✓ (1 == 1) | **봄** |

- 이 함수는 `create_block_mask` 에 넘겨져 flex_attention 한 번으로 계산된다.
- **컴파일(`transformer.compile()`)을 해야 빠르다.** 안 하면 fp32 점수 행렬을 통째로 만들어 고해상도에서 메모리가 부족해진다고 코드가 경고한다.

#### ③ 기본 경로: 같은 규칙을 구간별로 나눠 계산 (`QwenImage21AttnProcessor`)

기본 processor 는 flex 를 쓰지 않는다. 대신 **prefix 를 "같은 번호가 이어진 구간(segment)"으로 잘라** 구간마다 SDPA 를 한 번씩 호출한다.

**구간 자르기 (`_qwenimage21_prefix_segments`, 309~324줄)**

```python
prefix_ids = image_ids[:prefix_len].tolist()
for index in range(1, prefix_len + 1):
    if index == prefix_len or prefix_ids[index] != prefix_ids[start]:   # 번호가 바뀌는 곳에서 자름
        segments.append((start, index, prefix_ids[start] < 0))           # (시작, 끝, 텍스트인가)
        start = index
```

예시(prefix = 위치 0~6) 적용:

```
segments = [(0, 2, True),    # 텍스트 T T
            (2, 6, False),   # 참조 이미지 A A A A
            (6, 7, True)]    # 지시문 T
```

**구간별 attention (505~543줄)**

```python
for start, end, is_text in segments:
    seg_mask = None
    if is_text:
        seg_len = end - start
        seg_mask = torch.cat([
            torch.ones(seg_len, start),              # ← 앞 구간 전부: "앞은 다 봄"
            torch.tril(torch.ones(seg_len, seg_len)),# ← 자기 구간 안은 아래 삼각형: "텍스트는 줄 단위 causal"
        ], dim=1)
    outputs.append(dispatch_attention_fn(
        query[:, start:end],                         # 이 구간의 query 만
        key[:, :end], value[:, :end],                # key 는 처음부터 "이 구간 끝"까지
        attn_mask=seg_mask, ...))

outputs.append(dispatch_attention_fn(
    query[:, prefix_len:],                           # 타깃 query
    key, value,                                      # ← 전체 key, 마스크 없음 = 전부 봄
    attn_mask=None if key_valid is None else key_valid[:, None, None, :], ...))
```

| 말 | 코드 조각 | 어떻게 구현되나 |
|---|---|---|
| 앞만 본다 | `key[:, :end]` | key 를 **이 구간 끝까지만** 잘라서 넘김 → 뒤쪽은 아예 안 들어감 |
| 이미지 한 장 안에서는 전부 | 이미지 구간은 `seg_mask = None` | `:end` 가 자기 이미지 **끝까지** 포함 → 이미지 안 첫 토큰도 마지막 토큰까지 봄 |
| 텍스트는 줄 단위 causal | `torch.tril(...)` | 텍스트 구간 안에서만 아래 삼각형 마스크를 추가 |
| 타깃은 전부 본다 | `key` 전체, 마스크 없음 | 타깃이 맨 뒤이므로 "앞 전부 + 자기 블록 전부" = 시퀀스 전체 |

예시로 보면:
- 참조 구간(2~6)의 query 는 key 0~5 를 본다 — 앞 텍스트 두 개와 자기 이미지 네 개 전부, 뒤는 안 봄.
- 지시문 구간(6~7)의 query 는 key 0~6 을 보되 자기 구간 안은 tril 적용.
- 타깃(7~10)은 key 0~10 전부.

flex 경로의 한 줄 식과 **결과는 똑같고** 계산 방식만 다르다. 코드 주석도 "exact but slower".

#### ④ 캐시 단계: 타깃만 남으면 마스크가 사라짐 (953~962줄)

```python
if kv_cache_mode == "cached":
    # decode: only the target image's queries are recomputed. The block-causal mask degenerates to full
    # attention for target rows (they see the entire prefix + their own block), so only the padding mask is
    # needed.
    joint_hidden_states = joint_hidden_states[:, prefix_len:]    # 타깃 토큰만 입력
    attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
```

- 2~39스텝의 query 는 타깃뿐. 타깃은 원래 **모든 것을 볼 수 있으므로** block-causal mask 가 필요 없어지고 padding 마스크만 남는다.
- Q4 표에서 "맨 아래 타깃 줄은 전부 ✓" 였던 것이 코드에서는 이렇게 나타난다.

> 한 줄: **`q_idx >= kv_idx` 가 "앞만 본다", `q_image_id == kv_image_id & q_image_id >= 0` 가 "같은 이미지 안은 전부 본다". 기본 경로는 같은 규칙을 "key 를 구간 끝까지만 자르고, 텍스트 구간에만 tril 마스크"로 구현한다. 캐시 단계에서는 타깃만 남아 마스크가 필요 없어진다.**

---

## 9. 한계 / 미확인 사항

*블로그·README 중심 릴리스라 인용·재현 시 주의가 필요한 부분.*

- **기술 리포트 없음** — 학습 데이터·단계·RL·증류 방법은 2.0 리포트로 추정할 뿐.
- **학습 코드 없음** — LoRA 학습은 ModelScope DiffSynth-Studio 쪽 지원에 의존.
- **ablation 0건** — 공유 modulation·block-causal 이 품질에 주는 영향 근거 없음.
- **4-NFE 증류판 미공개** — 40스텝만 제공.
- **속도·편집 정량 수치 없음**.

---

## 10. 한 줄 요약 (전체)

> **Qwen-Image-2.1 = 2.0 리포트 설계(동결 Qwen3-VL + single-stream + f16c64 VAE)의 첫 공개 가중치. 중복 가중치를 걷어내 DiT 20B → 7B (토큰당 계산량은 거의 같음), block-causal + t=0 조건 + prefix KV cache 로 다중 참조 편집 가속, RGBA 통합. 단 자사 벤치 한 장·레시피 비공개·비상업 라이선스라 "무료 업그레이드"는 아니다.**

---

## 11. 관련 링크 / 메모리

- 설계 원논문: [PAPER_Qwen-Image-2.0.md](PAPER_Qwen-Image-2.0.md), 후처리: [PAPER_Qwen-Image-2.0-RL.md](PAPER_Qwen-Image-2.0-RL.md)
- 1.x: [PAPER_Qwen-Image.md](PAPER_Qwen-Image.md), 투명 배경 선행: Qwen-Image-Layered
- 같은 계열(single-stream): [PAPER_Z-Image.md](PAPER_Z-Image.md), [PAPER_Lumina-Image-2.0.md](PAPER_Lumina-Image-2.0.md)
- 관련 메모리: [[paper_qwen_image_2]], [[paper_qwen_image_2_rl]], [[paper_qwen_image]], [[paper_z_image]], [[paper_any2anytryon]]
- 외부: [블로그](https://qwen.ai/blog?id=qwen-image-2.1), [HF](https://huggingface.co/Qwen/Qwen-Image-2.1), [GitHub](https://github.com/QwenLM/Qwen-Image-2.1), [diffusers PR #14804](https://github.com/huggingface/diffusers/pull/14804), [Qwen-Image-Bench (arXiv 2605.28091)](https://arxiv.org/html/2605.28091v1), [2.0 리포트](https://arxiv.org/pdf/2605.10730), [2.0-RL 리포트](https://arxiv.org/pdf/2606.27608)
