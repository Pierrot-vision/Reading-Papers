# PAPER: DMD - One-step Diffusion with Distribution Matching Distillation 쉽게 읽기

## 0. 이 문서를 읽는 법

이 문서는 DMD 논문을 처음 읽는 사람이 흐름을 놓치지 않도록 다시 정리한 리뷰입니다.

핵심 목표는 하나입니다.

> **DMD는 느린 diffusion teacher가 여러 step에 걸쳐 만들던 이미지를, student generator가 1-step으로 만들도록 증류하는 방법이다.**

DMD2 문서와 같이 읽는다면 이렇게 보면 좋습니다.

```text
DMD  = 분포 매칭 증류의 원형. L_reg라는 안전벨트가 있음.
DMD2 = DMD의 L_reg를 제거하고 TTUR, GAN loss, backward simulation으로 확장.
```

이 문서는 아래 순서로 읽으면 편합니다.

1. DMD가 해결하려는 문제
2. 등장인물: `G_θ`, teacher, critic `μ_fake`
3. Figure 2로 보는 전체 학습 구조
4. 핵심 수식: score, KL gradient, surrogate loss, `L_reg`
5. 실제 학습 루프
6. DMD와 CFG distillation 차이
7. 실험 결과와 FAQ

---

## 1. 메타 정보

| 항목 | 내용 |
|---|---|
| 논문 | One-step Diffusion with Distribution Matching Distillation |
| 저자 | Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, Taesung Park |
| 학회 | CVPR 2024 |
| arXiv | https://arxiv.org/abs/2311.18828 |
| 프로젝트 페이지 | https://tianweiy.github.io/dmd/ |
| 후속 논문 | DMD2: Improved Distribution Matching Distillation for Fast Image Synthesis, NeurIPS 2024 |

---

## 2. 한 문장 요약

> **DMD는 student가 만든 이미지 분포 `p_fake_θ`를 teacher/real 이미지 분포 `p_real`에 맞추기 위해, teacher score와 fake score의 차이를 generator에 역전파하는 1-step diffusion distillation 방법이다.**

조금 더 쉽게 말하면:

> **teacher가 "진짜 이미지 분포는 이쪽이야"라고 알려주고, critic이 "지금 student는 이쪽에 몰려 있어"라고 알려주면, DMD는 그 차이를 이용해 student를 real 분포 쪽으로 밀어준다.**

---

## 3. DMD가 해결하려는 문제

일반 diffusion 모델은 이미지를 만들 때 여러 번 모델을 호출합니다.

```text
noise
  -> denoise
  -> denoise
  -> denoise
  -> ...
  -> image
```

좋은 품질을 얻으려면 50 step, 100 step, 때로는 그 이상이 필요합니다. CFG까지 쓰면 한 step에 conditional/unconditional forward를 둘 다 호출하므로 비용이 더 커집니다.

DMD의 목표는 이것입니다.

```text
teacher: many-step diffusion
student: one-step generator
```

즉:

```text
z -> G_θ(z) -> x̂₀
```

한 번의 forward로 이미지를 만들게 하려는 것입니다.

여기서 중요한 점은 DMD가 teacher의 각 step을 그대로 따라 하려는 방법이 아니라는 것입니다.

```text
나쁜 이해:
  teacher의 50-step trajectory를 1-step으로 압축해서 똑같이 따라 한다

더 정확한 이해:
  student가 만들어내는 이미지들의 전체 분포가 teacher/real 분포와 같아지도록 학습한다
```

그래서 이름이 **Distribution Matching Distillation**입니다.

---

## 4. 등장인물 정리

### 4.1 Student generator, `G_θ`

논문에서 student generator는 **`G_θ`**로 씁니다.

여기서 `θ`는 student 신경망의 학습 파라미터, 즉 weights입니다.

DMD의 student는 1-step generator입니다.

```text
G_θ(z) -> x̂₀
```

| 기호 | 뜻 |
|---|---|
| `z` | random noise |
| `G_θ` | 학습되는 student generator |
| `x̂₀` | student가 한 번에 만든 clean image 또는 latent |

### 4.2 Real distribution, `p_real`

`p_real`은 teacher 또는 real dataset이 나타내는 이미지 분포입니다.

쉽게 말하면:

```text
진짜 같고 좋은 이미지들이 이미지 공간에서 어디에 모여 있는가
```

를 나타내는 분포입니다.

### 4.3 Fake distribution, `p_fake_θ`

`p_fake_θ`는 student가 만든 한 장의 이미지가 아닙니다.

정확히는:

> **현재 student `G_θ`에 여러 noise `z`를 넣었을 때 나오는 모든 이미지들의 분포**

예를 들어:

```text
z1 -> G_θ -> dog-like image
z2 -> G_θ -> blurry face
z3 -> G_θ -> landscape
...
```

이 출력들이 이미지 공간 어디에 얼마나 모이는지의 패턴이 `p_fake_θ`입니다.

학습이 진행되면 `θ`가 바뀌므로 `p_fake_θ`도 계속 바뀝니다.

### 4.4 Teacher score, `s_real`

Teacher score는 `p_real`의 방향 신호입니다.

```text
s_real(x_t,t)
= "이 noisy latent 위치에서 real 분포 쪽으로 가려면 어느 방향인가?"
```

Teacher는 이미 학습된 diffusion model이며, DMD 학습 중에는 frozen입니다.

### 4.5 Fake score critic, `μ_fake`

DMD에서 가장 중요한 추가 네트워크입니다.

한 줄 정의:

> **`μ_fake`는 현재 student 분포 `p_fake_θ`의 score를 추정하는 보조 diffusion 모델이다.**

중요한 점:

```text
μ_fake는 GAN discriminator가 아니다.
real/fake 확률을 내는 판별기가 아니라,
student 분포의 score 방향을 알려주는 score estimator다.
```

Teacher가 real 분포의 score를 알려주듯, `μ_fake`는 fake 분포의 score를 알려줍니다.

```text
s_real = teacher/real 분포의 방향
s_fake = student/fake 분포의 방향
```

DMD는 이 둘의 차이로 student를 업데이트합니다.

---

## 5. Diffusion score를 먼저 이해하기

DMD의 핵심 수식은 score에서 시작합니다.

```math
s(x_t,t)
=
\nabla_{x_t}\log p_t(x_t)
```

기호를 풀면:

| 기호 | 뜻 |
|---|---|
| `x_t` | clean image `x₀`에 timestep `t`만큼 noise를 섞은 상태 |
| `p_t(x_t)` | timestep `t`에서 noisy image들이 따르는 분포 |
| `log p_t(x_t)` | 현재 위치 `x_t`가 얼마나 그럴듯한지의 로그 밀도 |
| `s(x_t,t)` | 그 로그 밀도가 가장 빠르게 커지는 방향 |

와닿게 말하면 score는 **이미지 공간 위의 길 안내 화살표**입니다.

```text
현재 위치 = x_t
산의 높이 = log p_t(x_t), 즉 이 위치가 얼마나 그럴듯한지
score = 가장 가파르게 높아지는 방향
```

그래서 score는 정답 이미지를 직접 주는 값이 아닙니다.

```text
정답 이미지 X
그럴듯한 이미지 분포 쪽으로 가는 방향 O
```

DMD에서는 이 score를 두 종류로 씁니다.

```text
s_real:
  teacher가 알려주는 real 분포 방향

s_fake:
  μ_fake가 알려주는 현재 student 분포 방향
```

이 둘의 차이를 보면 student를 어떻게 옮겨야 할지 알 수 있습니다.

```text
student가 너무 자주 만드는 방향에서는 빠져나오고,
teacher/real 분포가 좋아하는 방향으로 이동한다.
```

---

## 6. Figure 2로 보는 DMD 전체 구조

논문 Figure 2가 DMD의 전체 학습 구조를 가장 잘 보여줍니다.

<p align="center">
  <img src="figures/dmd_fig2.png" alt="DMD framework overview" width="900"/>
</p>

그림은 크게 왼쪽과 오른쪽으로 나누어 읽으면 됩니다.

### 6.1 왼쪽: one-step generator와 regression loss

왼쪽은 student generator `G_θ`입니다.

```text
random latent z -> G_θ -> fake image x̂₀
```

DMD는 이 fake image에 두 종류의 학습 신호를 줍니다.

첫 번째가 `L_reg`입니다.

```text
teacher가 미리 만든 paired target image와 student 출력을 MSE로 맞춘다.
```

이 손실은 DMD의 안정화 장치입니다. DMD2에서는 이 장치를 제거하려고 여러 개선을 넣지만, DMD 원논문에서는 매우 중요합니다.

### 6.2 오른쪽: distribution matching gradient

오른쪽은 DMD의 핵심입니다.

Student가 만든 clean image `x̂₀`를 그대로 score network에 넣지 않습니다. 먼저 다시 noise를 섞습니다.

```text
x̂₀ -> forward diffusion -> x_t
```

왜 다시 noise를 섞을까요?

Diffusion score network는 clean image가 아니라 noisy image `x_t`를 입력으로 score를 추정하도록 학습되어 있기 때문입니다.

그다음 같은 `x_t`를 두 네트워크에 넣습니다.

```text
x_t -> teacher   -> s_real
x_t -> μ_fake    -> s_fake
```

같은 위치에서 두 방향을 비교해야 하므로, teacher와 critic 모두 같은 `x_t`를 봅니다.

### 6.3 그림의 핵심 한 줄

Figure 2는 아래 문장을 그림으로 표현한 것입니다.

> **DMD는 student 출력에 다시 noise를 섞은 뒤, 그 위치에서 teacher score와 fake score를 비교해 student generator를 업데이트한다.**

텍스트로 줄이면:

```text
z
 -> G_θ
 -> x̂₀
 -> add noise
 -> x_t
 -> teacher gives s_real
 -> μ_fake gives s_fake
 -> score difference updates G_θ

추가로:
x̂₀와 paired teacher target 사이의 L_reg도 G_θ를 안정화한다.
```

---

## 7. 핵심 수식 1: 목표는 KL 줄이기

DMD의 목표는 student 분포를 real/teacher 분포에 맞추는 것입니다.

```math
\min_\theta
D_{\mathrm{KL}}
\left(
p_{\mathrm{fake},\theta}
\Vert
p_{\mathrm{real}}
\right)
```

기호를 풀면:

| 기호 | 뜻 |
|---|---|
| `p_fake,θ` | 현재 student가 만드는 이미지 분포 |
| `p_real` | real/teacher 이미지 분포 |
| `D_KL(p_fake || p_real)` | 두 분포가 얼마나 다른지 재는 값 |

직관:

```text
student가 만드는 이미지들의 전체 패턴이
real/teacher 이미지들의 전체 패턴과 같아지게 하자.
```

하지만 이 KL 값을 직접 계산하기는 어렵습니다.

왜냐하면 이미지 하나 `x`에 대해:

```math
D_{\mathrm{KL}}(p_{\mathrm{fake}}\Vert p_{\mathrm{real}})
=
\mathbb{E}_{x\sim p_{\mathrm{fake}}}
\left[
\log p_{\mathrm{fake}}(x)
-
\log p_{\mathrm{real}}(x)
\right]
```

이 식에는 `log p_fake(x)`와 `log p_real(x)`가 필요합니다.

그런데 diffusion model은 보통 어떤 이미지의 절대 log probability를 직접 주지 않습니다. 대신 score, 즉 `∇ log p(x)` 방향을 줍니다.

```text
우리가 모르는 것:
  산의 절대 높이 log p(x)

우리가 얻을 수 있는 것:
  산의 경사 방향 ∇ log p(x)
```

DMD는 그래서 KL 값 자체가 아니라 **KL을 줄이는 방향**을 score 차이로 구합니다.

---

## 8. 핵심 수식 2: KL gradient는 score 차이로 근사된다

DMD의 핵심 gradient는 다음 형태입니다.

```math
\nabla_\theta
D_{\mathrm{KL}}
\left(
p_{\mathrm{fake},\theta}
\Vert
p_{\mathrm{real}}
\right)
\approx
\mathbb{E}_{t,z}
\left[
\left(
s_{\mathrm{fake}}(x_t,t)
-
s_{\mathrm{real}}(x_t,t)
\right)
\frac{\partial x_t}{\partial \theta}
\right]
```

여기서:

| 기호 | 뜻 |
|---|---|
| `z` | student 입력 noise |
| `x̂₀ = G_θ(z)` | student가 만든 clean image |
| `x_t = F(x̂₀,t)` | `x̂₀`에 timestep `t`만큼 noise를 다시 섞은 것 |
| `s_real(x_t,t)` | teacher가 알려주는 real 분포 score |
| `s_fake(x_t,t)` | `μ_fake`가 알려주는 fake 분포 score |

이 식을 말로 풀면:

```text
student가 만든 이미지 위치에서
fake 분포가 끌어당기는 방향과
real 분포가 끌어당기는 방향을 비교한다.

그 차이만큼 student를 업데이트한다.
```

왜 `s_fake - s_real`일까요?

```text
s_fake:
  student가 이미 많이 가는 방향
  -> 거기서 빠져나와야 함

s_real:
  real 이미지들이 많은 방향
  -> 그쪽으로 가야 함
```

그래서 결과적으로는:

```text
student가 자기 분포의 과한 모드에서는 빠지고,
real 분포의 모드 쪽으로 이동한다.
```

---

## 9. 핵심 수식 3: autograd surrogate loss

문제는 위 gradient를 그대로 구현하기 어렵다는 점입니다.

그래서 DMD는 surrogate loss를 씁니다.

먼저 score 차이로 gradient 방향을 계산합니다.

```math
g
=
s_{\mathrm{fake}}(x_t,t)
-
s_{\mathrm{real}}(x_t,t)
```

그다음 아래 loss를 만듭니다.

```math
\mathcal{L}_{\mathrm{DMD}}
=
\frac{1}{2}
\left\|
\hat{x}_0
-
\left(
\hat{x}_0
-
g
\right)_{\mathrm{detach}}
\right\|_2^2
```

이 식이 처음 보면 이상합니다.

왜 `x̂₀ - g`를 만들고 detach할까요?

PyTorch autograd 관점에서 보면:

```text
(x̂₀ - g).detach()
```

는 상수입니다. 그러면 위 MSE를 `x̂₀`로 미분하면:

```math
\frac{\partial \mathcal{L}_{\mathrm{DMD}}}{\partial \hat{x}_0}
=
g
```

즉, 복잡한 KL gradient를 직접 미분하지 않고도, 우리가 원하는 방향 `g`가 student 출력에 gradient로 걸립니다.

직관:

```text
원하는 gradient 방향 g를 먼저 계산해 둔다.
그 방향이 autograd로 흘러가도록 MSE 모양의 가짜 loss를 만든다.
```

그래서 surrogate loss입니다.

---

## 10. 핵심 수식 4: `μ_fake`는 어떻게 학습되나

`μ_fake`는 student가 만든 이미지 분포의 score를 추정해야 합니다.

그러려면 student 출력 `x̂₀ = G_θ(z)`를 데이터처럼 사용해 diffusion denoising loss를 학습합니다.

```math
\mathcal{L}_{\mathrm{fake}}
=
\mathbb{E}_{z,t,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_{\mu_{\mathrm{fake}}}
\left(
F(G_\theta(z),t),t
\right)
\right\|_2^2
\right]
```

기호:

| 기호 | 뜻 |
|---|---|
| `ε` | 실제로 섞은 Gaussian noise |
| `F(G_θ(z),t)` | student 출력에 noise를 섞은 noisy image |
| `ε_μ_fake(·)` | `μ_fake`가 예측한 noise |

이건 표준 diffusion training loss와 같습니다.

다만 차이가 있습니다.

```text
일반 diffusion:
  real image에 noise를 섞고 denoising 학습

DMD의 μ_fake:
  student가 만든 fake image에 noise를 섞고 denoising 학습
```

그래서 `μ_fake`는 현재 student 분포의 score를 따라가게 됩니다.

주의할 점:

```text
student가 학습되면 p_fake_θ가 바뀐다.
그러면 μ_fake도 계속 다시 학습되어야 한다.
```

DMD2에서 TTUR가 나온 이유도 이 지점입니다. DMD에서는 `L_reg`가 이 불안정성을 어느 정도 잡아주는 안전벨트였습니다.

---

## 11. 핵심 수식 5: `L_reg`는 안전벨트다

DMD 원논문은 순수 score difference만 쓰지 않고 regression loss를 같이 씁니다.

```math
\mathcal{L}_{\mathrm{reg}}
=
\mathbb{E}_{(z,y)\sim\mathcal{D}}
\left[
\ell(G_\theta(z),y)
\right]
```

기호:

| 기호 | 뜻 |
|---|---|
| `𝒟` | 미리 만든 paired dataset |
| `z` | teacher sampling을 시작한 noise |
| `y` | 같은 `z`에서 teacher를 여러 step 돌려 만든 target image |
| `ℓ` | MSE 또는 LPIPS 같은 regression distance |

쉽게 말하면:

```text
같은 z에서 teacher가 만든 이미지 y와
student가 한 번에 만든 G_θ(z)를
픽셀/feature 기준으로도 비슷하게 맞춘다.
```

이 손실은 DMD에서 매우 중요합니다.

역할:

```text
distribution matching gradient:
  분포를 맞추는 큰 방향

L_reg:
  같은 z에 대해 teacher target 근처에 묶어두는 안정화 장치
```

단점도 있습니다.

```text
paired target y를 미리 만들어야 하므로 비싸다.
student가 teacher 출력을 베끼는 방향으로 묶인다.
teacher를 넘기 어렵다.
```

이 단점을 해결하려고 DMD2가 `L_reg`를 제거하고 TTUR, GAN loss 등을 도입합니다.

---

## 12. 최종 학습 objective

DMD의 generator는 두 손실을 같이 받습니다.

```math
\mathcal{L}_{G}
=
\mathcal{L}_{\mathrm{DMD}}
+
\lambda_{\mathrm{reg}}
\mathcal{L}_{\mathrm{reg}}
```

그리고 `μ_fake`는 별도로 fake denoising loss를 받습니다.

```math
\mathcal{L}_{\mu_{\mathrm{fake}}}
=
\mathcal{L}_{\mathrm{fake}}
```

학습 루프를 단순화하면:

```text
1. z를 뽑는다.
2. student G_θ가 x̂₀를 만든다.
3. x̂₀에 noise를 섞어 x_t를 만든다.
4. teacher가 s_real을 준다.
5. μ_fake가 s_fake를 준다.
6. s_fake - s_real로 L_DMD surrogate를 만든다.
7. paired target y와 비교해 L_reg를 만든다.
8. L_DMD + λ_reg L_reg로 G_θ를 업데이트한다.
9. student 출력에 대한 denoising loss로 μ_fake를 업데이트한다.
```

의사코드:

```python
for step in training:
    z = sample_noise()

    # student output
    x0_hat = G_theta(z)

    # distribution matching
    t = sample_timestep()
    x_t = add_noise(x0_hat, t)
    s_real = teacher_score(x_t, t)      # frozen
    s_fake = fake_score_mu(x_t, t)      # learned critic
    g = s_fake - s_real
    loss_dmd = surrogate_mse(x0_hat, g)

    # regression safety belt
    y = paired_teacher_target(z)
    loss_reg = distance(x0_hat, y)

    update(G_theta, loss_dmd + lambda_reg * loss_reg)

    # train fake score critic
    x0_fake = stop_gradient(x0_hat)
    loss_fake = denoising_loss(mu_fake, x0_fake)
    update(mu_fake, loss_fake)
```

---

## 13. DMD vs CFG distillation

DMD와 CFG distillation은 둘 다 distillation이지만 줄이는 비용이 다릅니다.

| 항목 | DMD | CFG distillation |
|---|---|---|
| 목표 | many-step diffusion을 1-step generator로 압축 | CFG의 2-forward 비용을 1-forward로 압축 |
| student 형태 | `z -> x̂₀` one-step generator | teacher와 비슷한 denoising model |
| 줄이는 것 | step 수 | step당 forward 수 |
| 핵심 loss | score difference + `L_reg` | teacher CFG output과 MSE |
| `μ_fake` critic | 있음 | 없음 |

즉:

```text
CFG distillation:
  한 step 안에서 cond/uncond 두 번 하던 것을 한 번으로 줄임.

DMD:
  여러 denoising step 자체를 한 번으로 줄임.
```

DMD라는 이름을 쓰려면 최소한 아래 요소가 있어야 합니다.

```text
1. student 분포 p_fake_θ
2. fake score critic μ_fake
3. teacher score s_real
4. score difference로 만든 distribution matching gradient
```

---

## 14. 실험 결과 요약

### 14.1 주요 결과

| 데이터셋 | 모델 | NFE | FID 낮을수록 좋음 | 비고 |
|---|---|---:|---:|---|
| ImageNet 64x64 | EDM teacher | 79 | 1.79 | teacher |
| ImageNet 64x64 | DMD | 1 | 2.62 | 1-step |
| COCO-30k / LAION text-to-image | SDv1.5 teacher | 100 | 8.59 | 50 step + CFG |
| COCO-30k / LAION text-to-image | DMD | 1 | 11.49 | 1-step |

의미:

```text
DMD는 teacher보다 조금 품질이 낮지만,
NFE를 극단적으로 줄인다.
```

### 14.2 Ablation

논문에서 중요한 ablation은 다음 흐름입니다.

| 설정 | 결과 해석 |
|---|---|
| Full DMD | 가장 안정적 |
| `L_reg` 제거 | 품질 크게 하락 |
| `μ_fake` 학습 제거 | 학습 불안정 또는 발산 |
| distribution matching 제거 | 단순 regression이 되어 품질 하락 |

즉 DMD에서는:

```text
distribution matching gradient + μ_fake + L_reg
```

세 요소가 같이 필요합니다.

---

## 15. DMD의 한계와 DMD2로 이어지는 이유

DMD는 1-step distillation의 가능성을 보여줬지만 한계도 명확합니다.

### 15.1 `L_reg`가 비싸다

`L_reg`를 쓰려면 paired dataset이 필요합니다.

```text
(noise z, teacher가 여러 step으로 만든 target image y)
```

큰 text-to-image 모델에서는 이 target을 대량으로 만드는 비용이 큽니다.

### 15.2 teacher를 넘기 어렵다

`L_reg`는 student를 teacher output에 묶습니다.

안정적이지만:

```text
student가 teacher를 그대로 베끼는 방향으로 학습됨
```

그래서 teacher의 결함이나 한계도 같이 따라갈 수 있습니다.

### 15.3 pure distribution matching은 불안정하다

`L_reg`를 제거하면 `μ_fake`가 현재 student 분포를 잘 따라가야 합니다.

하지만 student가 계속 바뀌므로 critic이 뒤처질 수 있습니다.

```text
student 분포는 이미 이동했는데
μ_fake는 예전 student 분포의 score를 알려줌
```

이것이 DMD2 문서에서 말한 stale critic 문제입니다.

DMD2는 이 문제를 해결하기 위해:

```text
TTUR:
  critic을 generator보다 더 자주 학습

Real-data GAN loss:
  teacher score만 보지 않고 진짜 이미지도 직접 봄

Backward simulation:
  multi-step student의 train/inference mismatch 해결
```

을 도입합니다.

---

## 16. 자주 헷갈리는 질문

### Q1. DMD는 그냥 teacher output MSE distillation인가?

아닙니다.

DMD에는 `L_reg`가 있지만, 핵심은 distribution matching gradient입니다.

```text
단순 MSE distillation:
  teacher target과 student output을 직접 맞춤

DMD:
  teacher score와 fake score의 차이로 student 분포를 맞춤
  + 안정화를 위해 L_reg도 사용
```

### Q2. 왜 student 출력에 다시 noise를 씌우나?

Score network는 noisy input `x_t`에서 정의됩니다.

Student output `x̂₀`는 clean image에 가깝기 때문에 그대로 넣으면 score network가 기대하는 입력이 아닙니다.

그래서:

```text
x̂₀ -> add noise -> x_t -> score networks
```

를 거칩니다.

또한 KL의 평균이 `x ~ p_fake` 위에서 잡히므로, student가 만든 위치에서 score를 비교해야 합니다.

### Q3. 왜 teacher와 `μ_fake` 둘 다 같은 `x_t`를 보나?

같은 위치에서 두 분포의 방향을 비교해야 하기 때문입니다.

```text
teacher:
  이 위치에서 real 분포는 어느 방향인가?

μ_fake:
  이 위치에서 student 분포는 어느 방향인가?
```

다른 위치에서 비교하면 두 score의 차이가 의미를 잃습니다.

### Q4. `μ_fake`는 discriminator인가?

기본적으로 아닙니다.

`μ_fake`는 real/fake 확률을 출력하는 GAN discriminator가 아니라, student 분포의 score를 추정하는 diffusion-style score estimator입니다.

DMD2에서는 여기에 GAN head가 추가되지만, DMD 원논문의 `μ_fake`는 score critic입니다.

### Q5. 왜 시장의 빠른 모델들은 1-step보다 4-step, 8-step이 많나?

1-step은 가장 빠르지만 가장 어렵습니다.

특히 고해상도 text-to-image에서는:

```text
pure noise -> 완성 이미지
```

를 한 번에 해야 해서 품질, 다양성, prompt alignment가 모두 어려워집니다.

그래서 후속 연구와 실제 배포 모델들은 4-step 또는 8-step 같은 few-step distilled generator로 많이 이동했습니다.

---

## 17. 최종 정리

DMD를 이해할 때 가장 중요한 문장은 이것입니다.

> **DMD는 student가 만든 분포 `p_fake_θ`와 teacher/real 분포 `p_real`의 차이를 KL로 보고, 그 KL gradient를 `s_fake - s_real`이라는 score 차이로 근사해 1-step generator `G_θ`를 학습하는 방법이다.**

구성요소별로 기억하면:

| 구성요소 | 역할 |
|---|---|
| `G_θ` | 1-step student generator |
| `p_fake_θ` | student가 만드는 이미지 분포 |
| `s_real` | teacher/real 분포의 score |
| `μ_fake`, `s_fake` | student 분포의 score를 추정하는 critic |
| `L_DMD` | score 차이로 만든 distribution matching surrogate |
| `L_reg` | paired teacher target에 묶어두는 안정화 손실 |

DMD는 DMD2의 출발점입니다. DMD2를 이해하려면 DMD에서 **왜 `μ_fake`가 필요한지**, **왜 `L_reg`가 안정화에는 좋지만 비용과 품질 상한을 만드는지**를 먼저 이해하는 것이 중요합니다.

---

## 18. 관련 메모

- DMD2 후속 리뷰: [PAPER_DMD2.md](PAPER_DMD2.md)
- DMD 논문: https://arxiv.org/abs/2311.18828
- DMD 프로젝트 페이지: https://tianweiy.github.io/dmd/
