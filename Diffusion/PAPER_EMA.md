# ALG_EMA — Exponential Moving Average of Model Weights

> **한 줄 요약**
> 학습 도중 매 step (또는 N step 마다) 모델 weight 의 **지수이동평균** 을 별도의 EMA copy 로 유지.
> 추론 시 raw weight 가 아닌 EMA copy 를 사용하면 **수렴이 더 매끄럽고 generalization 이 좋아짐** —
> 디퓨전 학습의 표준 관행.

---

## 0. 메타 정보

| 항목 | 값 |
|---|---|
| 알고리즘 명 | EMA (Exponential Moving Average) |
| 출처 | Polyak averaging (1992) 의 deep learning 변종. 디퓨전에서는 [DDPM (Ho et al. 2020)](https://arxiv.org/abs/2006.11239) 부터 표준 |
| 표준 하이퍼파라미터 | decay=0.999, warmup=0, update_interval=10 (디퓨전 학습 통상값) |
| 효과 | inference 품질 ↑, 학습 진동 평활화, FID 점수 안정 |
| 비용 | 모델 사이즈만큼 추가 메모리 (CPU offload 가능), step 당 element-wise mul+add 한 번 |

### 용어 한 줄 설명

- **EMA decay (smoothing)** — `0 < decay < 1`. 클수록 (예: 0.9999) 과거 weight 영향 강함, 작을수록 (예: 0.99) 현재 weight 빨리 따라감.
- **half-life** — EMA 가 과거 weight 의 영향력을 절반으로 줄이는 데 걸리는 step 수. `t_{1/2} = -log(2) / log(decay)`.
- **warmup** — 학습 초반 step 동안 decay 를 0 에서 점진적으로 키워, random-init weight 가 EMA 에 너무 오래 남는 것을 방지.
- **update_interval** — EMA 업데이트 주기. 매 step 이 아니라 10 step 마다 update 하면 비용 1/10.
- **Polyak averaging** — 단순 평균 (`avg = (w_1 + w_2 + ... + w_n) / n`). EMA 는 가중 평균 (최신 weight 가 더 큼).

---

## §1. 한눈에 보는 핵심 — 평균 점수 비유

운동 선수가 매일 경기를 한다고 하자. 컨디션이 들쭉날쭉해서 어떤 날은 90 점, 어떤 날은 70 점.

- **Raw weight** = 오늘 점수. 들쭉날쭉.
- **EMA weight** = "최근 100 일 가중 평균". 매끄럽고 안정적. 어제 점수 99% + 오늘 점수 1%.

학습이 끝났을 때 "어떤 weight 로 추론할까?" 의 답:

- 마지막 step 의 raw weight = 운에 따라 high 일 수도 low 일 수도.
- EMA weight = 최근 N step 의 가중 평균이라 안정.

핵심 통찰 두 가지:

1. **추론은 EMA, 학습은 raw** — EMA 는 grad 를 안 받음 (frozen copy). raw model 은 평소대로 backward.
2. **공짜 정규화** — gradient noise 가 평균화되어 사라지고, sharp minimum 보다 flat minimum 으로 수렴 유도.

---

## §1.5. weight 1 개로 따라가기 — 10 step EMA 트레이스

> **왜 이 절이 있나** — `ema ← decay·ema + (1-decay)·model` 식만 보면 EMA 가 "얼마나 빨리 따라가는지" 직관이 안 옵니다.   **단일 scalar weight 1 개** 의 10 step 변화를 손으로 트레이스해서 decay·warmup·update_interval 의 효과를 직접 봅시다.

### 미니 설정

| 항목 | 값 |
|---|---|
| 모델 weight 개수 | 1 (단일 scalar `w`) |
| 초기값 | w_raw = 0.0 (random init) |
| `decay` | 0.9 (이해하기 쉽게 큰 값.   실제는 0.999) |
| `warmup` | 3 (첫 3 step 동안 decay 점진 증가) |
| `update_interval` | 1 (매 step update.   비교 시 interval=2 도 봄) |
| 학습 step 수 | 10 |

### 등장 인물 — 들쭉날쭉한 학습 weight

학습 중 `w_raw` 가 매 step 변하는 모습 (가상):

```
step :  0     1     2     3     4     5     6     7     8     9
w_raw:  0.0   0.5   0.3   0.8   0.6   1.0   0.9   1.2   1.1   1.3
                                                                  ↑ 학습 진행, w 가 커지면서 진동
```

→ raw 만 보면 step 5 (1.0) 와 step 6 (0.9) 처럼 진동.   inference 에 어떤 step weight 를 쓸지 애매.

### Step-by-step EMA 트레이스 (decay=0.9, warmup=3, interval=1)

effective decay 계산과 update 를 손으로:

```python
def _current_decay(step):
    if warmup <= 0 or step >= warmup:   # step >= 3
        return decay                      # 0.9
    return decay * (step / warmup)        # warmup 동안 선형
```

| step | w_raw | effective decay | one_minus_decay | EMA 식 | w_ema |
|---|---|---|---|---|---|
| 0 | 0.0 | 0.9·(0/3) = 0.0 | 1.0 | 0·0.0 + 0.0·1.0 = 0.0 | **0.000** |
| 1 | 0.5 | 0.9·(1/3) = 0.3 | 0.7 | 0.0·0.3 + 0.5·0.7 | **0.350** |
| 2 | 0.3 | 0.9·(2/3) = 0.6 | 0.4 | 0.350·0.6 + 0.3·0.4 = 0.210 + 0.120 | **0.330** |
| 3 | 0.8 | 0.9 (warmup 끝) | 0.1 | 0.330·0.9 + 0.8·0.1 = 0.297 + 0.080 | **0.377** |
| 4 | 0.6 | 0.9 | 0.1 | 0.377·0.9 + 0.6·0.1 = 0.339 + 0.060 | **0.399** |
| 5 | 1.0 | 0.9 | 0.1 | 0.399·0.9 + 1.0·0.1 = 0.359 + 0.100 | **0.459** |
| 6 | 0.9 | 0.9 | 0.1 | 0.459·0.9 + 0.9·0.1 = 0.413 + 0.090 | **0.503** |
| 7 | 1.2 | 0.9 | 0.1 | 0.503·0.9 + 1.2·0.1 = 0.453 + 0.120 | **0.573** |
| 8 | 1.1 | 0.9 | 0.1 | 0.573·0.9 + 1.1·0.1 = 0.516 + 0.110 | **0.626** |
| 9 | 1.3 | 0.9 | 0.1 | 0.626·0.9 + 1.3·0.1 = 0.563 + 0.130 | **0.693** |

### 시각화 — raw vs EMA

```
       w_raw          w_ema
step    값            값
─────────────────────────────────
  0   ●0.0           ○0.000
  1   ●0.5  ↑↑↑      ○0.350  ↑       ← warmup 중, EMA 가 빨리 따라감
  2   ●0.3  ↓        ○0.330  ↓ (살짝)
  3   ●0.8  ↑↑↑      ○0.377  ↑ (천천히)  ← warmup 끝, decay=0.9 적용
  4   ●0.6  ↓        ○0.399  ↑ (계속 따라감)
  5   ●1.0  ↑↑↑      ○0.459  ↑ (천천히)
  6   ●0.9  ↓        ○0.503  ↑          ← raw 가 ↓ 인데도 EMA 는 ↑ (관성)
  7   ●1.2  ↑↑↑      ○0.573  ↑
  8   ●1.1  ↓        ○0.626  ↑          ← 또 raw ↓, EMA ↑
  9   ●1.3  ↑↑↑      ○0.693  ↑
```

→ **EMA 는 raw 의 진동을 평활화**.   step 6, 8 에서 raw 는 떨어지지만 EMA 는 천천히 올라감 (지난 step 들의 가중 평균).

### Warmup 효과 비교 — warmup=0 vs warmup=3

같은 raw weight 시퀀스에서 warmup=0 (즉시 decay=0.9) 인 경우:

| step | warmup=0 | warmup=3 |
|---|---|---|
| 0 | 0.0·0.9 + 0.0·0.1 = **0.000** | **0.000** |
| 1 | 0.0·0.9 + 0.5·0.1 = **0.050** | **0.350** |
| 2 | 0.050·0.9 + 0.3·0.1 = **0.075** | **0.330** |
| 3 | 0.075·0.9 + 0.8·0.1 = **0.148** | **0.377** |
| 4 | 0.148·0.9 + 0.6·0.1 = **0.193** | **0.399** |

→ **warmup=0 은 random init (w=0) 의 영향이 오래 남음** (step 4 에서도 0.193).   warmup=3 은 첫 3 step 동안 decay 작아서 random init 빨리 씻어내고 step 4 에 0.399 로 따라잡음.

**언제 warmup=0 이 괜찮은가** — 시작 weight 가 이미 합리적일 때 (예: 사전학습/증류 가중치에서 출발).   이 경우 EMA 가 random 값에서 출발하지 않으므로 warmup 이 사실상 불필요.   반대로 **fresh random init 이면 warmup>0 권장** (§9.6 참조).

### update_interval 효과 — interval=1 vs interval=2

interval=2 면 짝수 step 에서만 update:

| step | w_raw | EMA (interval=1) | EMA (interval=2) |
|---|---|---|---|
| 0 | 0.0 | 0.000 | 0.000 (update) |
| 1 | 0.5 | 0.350 | 0.000 (skip) |
| 2 | 0.3 | 0.330 | 0.000·0.9 + 0.3·0.1 = 0.030 (update) |
| 3 | 0.8 | 0.377 | 0.030 (skip) |
| 4 | 0.6 | 0.399 | 0.030·0.9 + 0.6·0.1 = 0.087 (update) |
| 5 | 1.0 | 0.459 | 0.087 (skip) |
| 6 | 0.9 | 0.503 | 0.087·0.9 + 0.9·0.1 = 0.168 (update) |
| 7 | 1.2 | 0.573 | 0.168 (skip) |
| 8 | 1.1 | 0.626 | 0.168·0.9 + 1.1·0.1 = 0.261 (update) |
| 9 | 1.3 | 0.693 | 0.261 (skip) |

→ **interval=2 면 EMA 가 절반 속도로 따라감**. 같은 step 9 에서 interval=1 은 0.693, interval=2 는 0.261.

**effective half-life** 식:

```
interval=1, decay=0.9: half-life = -log(2) / log(0.9) ≈ 6.6 step
interval=2, decay=0.9: half-life ≈ 13.2 step (2 배)
```

→ 실전 표준인 `interval=10, decay=0.999` 조합:
- effective half-life ≈ 6931 step (decay 만)
- interval=10 이라 실제 따라가는 속도 = 6931 / 10 ≈ 693 step (effective).

### 학습 종료 시 — 어떤 weight 로 inference?

step 9 종료 시점:

```
w_raw = 1.3                  ← 마지막 step weight (운에 따라 high/low)
w_ema = 0.693                ← 최근 10 step 의 가중 평균
```

**직관**:
- raw 1.3 은 step 9 의 한순간 진동값 — inference 결과 불안정.
- EMA 0.693 은 step 0 ~ 9 전체의 매끄러운 평균 — inference 결과 안정.

학습 step 수가 더 많으면 EMA 는 raw 를 거의 따라잡지만 (예: step 100 이면 EMA ≈ 1.25), 마지막 step 의 진동 noise 는 평활화 유지.

### scatter plot 직관 (가상)

```
w_raw 분포 (step 0~9):    0.0, 0.3, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3
mean                  =   0.77                                 ← 단순 평균
EMA (decay=0.9)       =   0.693                                ← 최근 step 가중

raw 분산 (std)        =   0.42                                 ← 들쭉날쭉
EMA 의 변화량 std     =   0.10                                 ← 매끄러움
                          ↑ EMA 가 학습 noise 를 ¼ 로 줄임
```

### 큰 decay (0.999) 시뮬레이션 — 1000 step 기준

decay=0.999 면 한 step 당 변화량:

```
EMA 변화 ≈ (w_raw - EMA) · 0.001
        = (1.0 - 0.5) · 0.001 = 0.0005 / step
```

→ 1000 step 후에야 EMA 가 raw 절반 거리만큼 이동.   **half-life = 693 step**.

50k ~ 200k step 규모 학습에서 decay=0.999, interval=10 → effective half-life ≈ 693 step.   학습 종료 시점에 EMA 는 마지막 ~3000 step 의 weight 가중 평균이 됨 → 충분히 평활화 + 충분히 학습 진행 따라잡음.

### state_dict 저장/복원 시 보존되는 것

```python
ema.state_dict() = {
    "ema_model": {...},   ← EMA copy 의 모든 weight (deepcopy 한 모델 state_dict)
    "decay":     0.999,
    "warmup":    0,
}
```

→ 체크포인트로 EMA copy 를 별도 파일로 저장.   resume 시 `load_state_dict` 로 EMA copy + 하이퍼파라미터 복원 → 학습 재개 시 EMA 동역학 정확히 그대로.

### 미니 예제에서 실제 규모로 스케일 업

| 미니 예제 | 실제 학습 |
|---|---|
| weight 1 개 (scalar) | 수억 ~ 수십억 parameters |
| decay=0.9 (이해 쉬운 값) | decay=0.999 (표준) |
| half-life ≈ 6.6 step | half-life ≈ 6931 step (interval=10 → effective 693) |
| 10 step 학습 | 50k ~ 200k step |
| update 비용 0 | 매 10 step 마다 수억 element-wise mul+add ≈ 1ms (interval=10 이라 0.1ms/step) |

### 미니 예제 한 줄 요약

> **EMA 한 step** = "ema_p ← decay · ema_p + (1-decay) · p" 의 in-place mul+add.   decay 가 클수록 천천히 따라가고 평활 효과 ↑, warmup 은 random init 영향을 빨리 씻어냄, interval 은 비용/속도 trade-off.   학습 종료 시 raw (마지막 step 진동) 가 아닌 EMA (가중 평균) 로 inference 하면 안정적인 결과.

---

## §2. 단계별 분해 — 학습 1 step 의 흐름

### 2.1 Step 0 — 학습 시작 시 EMA copy 생성

```python
ema = EMAModel(model, decay=0.999, warmup=0, device=None)
# 내부:
self.ema_model = deepcopy(model).requires_grad_(False)   # ★ 핵심
                                                          # - deepcopy: 별도 메모리
                                                          # - requires_grad=False: backward 안 함
```

**메모리** — 모델 한 벌 더 (수억 ~ 수십억 params 추가). GPU 메모리 부담 시 `device='cpu'` 로 두어 CPU 에 보관 가능.

### 2.2 Step n — Optimizer step 후 EMA update

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()

if (
    ema is not None
    and sync_gradients                                # grad accumulation 시 실제 step 일 때만
    and (step + 1) % ema_update_interval == 0         # ★ 10 step 마다
):
    ema.update(model, step)
```

EMA `update()`:

```python
@torch.no_grad()
def update(self, model, step):
    decay           = self._current_decay(step)        # warmup 처리된 effective decay
    one_minus_decay = 1.0 - decay

    for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
        p_data = p.data.to(ema_p.device, non_blocking=True)
        ema_p.data.mul_(decay).add_(p_data, alpha=one_minus_decay)
        # 등가: ema_p ← decay·ema_p + (1-decay)·p
```

**in-place 연산** (`mul_`, `add_`) 으로 메모리 추가 X, GPU kernel 한 번에 융합 가능.

### 2.3 Warmup 처리 — 초반 step 에서 decay 점진 증가

```python
def _current_decay(self, step):
    if self.warmup <= 0 or step >= self.warmup:
        return self.decay
    return self.decay * (step / max(1, self.warmup))   # 선형 증가
```

`warmup=1000` 인 경우:
- step 0: decay = 0.0 → 첫 step weight 가 EMA 의 100%
- step 500: decay = 0.999 × 0.5 = 0.4995
- step 1000: decay = 0.999 (warmup 끝)
- step 5000: decay = 0.999 (그대로)

**왜 warmup 필요?** random-init weight 가 EMA 에 0.999 weight 로 남으면 매우 천천히 사라짐. half-life ≈ 693 step. warmup 으로 random init 영향을 빨리 씻어냄.

### 2.4 학습 종료 시 — EMA model 추론용 사용

```python
ema_model = ema.get_model()     # frozen, requires_grad=False
ema_model.eval()
out = ema_model(noisy_x, t, ...)   # 매끄러운 weight 로 inference
```

### 2.5 체크포인트 저장/복원

```python
def state_dict(self, *args, **kwargs):
    return {
        "ema_model": self.ema_model.state_dict(*args, **kwargs),
        "decay":     self.decay,
        "warmup":    self.warmup,
    }

def load_state_dict(self, state_dict, *args, **kwargs):
    self.ema_model.load_state_dict(state_dict["ema_model"], *args, **kwargs)
    self.decay  = state_dict.get("decay",  self.decay)
    self.warmup = state_dict.get("warmup", self.warmup)
```

→ EMA weight + 하이퍼파라미터 (decay, warmup) 함께 저장. resume 시 정확히 같은 동역학 재개.

---

## §3. 왜 이렇게 했나 — 대안 비교

### (A) EMA vs Polyak averaging (단순 평균)

| metric | 특성 |
|---|---|
| ❌ Polyak averaging | `avg = mean(w_1, ..., w_n)` — 모든 step 동등 가중. 초반 random weight 가 영원히 영향. |
| ✅ EMA | 가중 평균, 오래된 weight 가 기하급수적으로 감쇠. 학습이 진행되면서 자연스럽게 잊음. |

EMA 는 **stationary distribution** 을 따라가므로 학습 후반부의 weight 분포에 점근적으로 수렴.

### (B) decay 0.999 vs 0.9999 vs 0.99999

half-life 표:

| decay | half-life (step) | 의미 |
|---|---|---|
| 0.99 | 69 | 매우 빠르게 따라감 (사실상 raw 와 비슷) |
| 0.999 | 693 | 적당 (디퓨전 표준) |
| 0.9999 | 6,931 | 천천히, 매우 안정 (짧은 학습엔 너무 길 수도) |
| 0.99999 | 69,314 | ~50k step 분량 안에선 거의 못 따라옴 |

decay=0.999 가 표준인 이유: 50k ~ 200k step 규모 학습에서 EMA 가 충분히 학습 분포를 따라잡으면서도 평활화 효과를 가짐.

### (C) update_interval 1 vs 10 vs 100

매 step update 하면 비용 ↑. 10 step 마다 update 하면:

- **비용** — element-wise mul+add 한 번 = `O(num_params)`. 수억 params 면 ~1ms (GPU). 10 step 마다면 0.1ms/step.
- **신호 손실** — 10 step 모은 후 한 번 update. EMA 가 보는 weight 는 띄엄띄엄. half-life 가 사실상 1/10 로 짧아짐 (decay=0.999, interval=10 → effective half-life ≈ 69 step).

→ `interval=10` 이 비용/효과 trade-off 의 통상 지점.

### (D) deepcopy vs lazy state_dict

| 방식 | 메모리 | 속도 | 단점 |
|---|---|---|---|
| ❌ state_dict 매번 비교 | 적음 | 느림 (clone 필요) | step 마다 메모리 alloc |
| ✅ deepcopy + in-place | 모델 한 벌 더 | 빠름 (in-place) | 메모리 부담 |

대부분 구현이 deepcopy 방식. 메모리 부담은 `device='cpu'` 옵션으로 완화 가능.

### (E) 경량 함수형 구현 vs 프레임워크 통합형 구현

EMA 구현은 크게 두 갈래로 나뉜다.

**프레임워크 통합형** (예: 학습 프레임워크의 `Algorithm` 인터페이스 위에 구축):

1. **Event hook 기반** — BATCH_END, EPOCH_END 등 학습 이벤트에 자동으로 EMA update 가 걸림.
2. **분산 학습 reshard 처리** — FSDP 처럼 weight 가 GPU 마다 쪼개진(sharded) 경우, full param 으로 모았다가 EMA update 후 다시 shard.
3. **half_life ↔ smoothing 변환** — `"1000ba"` 같은 시간 단위 문자열을 받아 내부 decay 로 변환.
4. **ema_start** — 학습 진행률 / step 단위로 EMA 시작 시점을 지연.

**경량 함수형 구현** (단일 `EMAModel` 클래스):

1. `update()` 를 학습 루프에서 직접 호출 — 프레임워크 의존 없음.
2. 분산 환경의 unwrap 은 학습 가속 라이브러리(accelerator 등)에 위임.
3. **단순 decay (float)** 만 — 시간 단위 변환 없음.
4. **warmup 만** 지원 (ema_start 는 train loop 에서 step 비교로 직접 구현 가능).

→ 통합형이 기능은 풍부하지만 프레임워크 의존이 큼. 경량형은 ~90 줄로 압축돼 이식성이 좋음.   본 문서의 코드 예시는 **경량 함수형** 기준.

---

## §4. 구체 수치 예시

### 4.1 EMA 업데이트 한 step 수치 추적

decay=0.999, 단일 weight w 가정:

| step | w_raw | one_minus_decay | EMA 변화 | w_ema |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 (warmup=0 first step) | 0.000 → 1.000 | 1.000 |
| 1 | 1.005 | 0.001 | 1.000·0.999 + 1.005·0.001 | 1.000005 |
| 100 | 1.040 | 0.001 | (이전 EMA)·0.999 + 1.040·0.001 | ~1.020 |
| 1000 | 1.150 | 0.001 | ... | ~1.085 |
| 5000 | 1.200 | 0.001 | ... | ~1.190 (수렴) |

→ EMA 는 raw weight 가 변할 때 천천히 따라감. **5×half-life = 5×693 ≈ 3500 step** 후 사실상 수렴.

### 4.2 warmup 효과 (warmup=1000)

| step | effective decay | EMA weight 의 random init 비중 |
|---|---|---|
| 0 | 0.000 | 0% (random init 즉시 덮임) |
| 100 | 0.0999 | 0.999^0 ≈ 100% × 0.0001 ≈ 0.01% |
| 500 | 0.4995 | 빠르게 0 으로 |
| 1000 | 0.999 (warmup 끝) | 거의 0% (random init 영향 사라짐) |

→ warmup=0 이면 step 0 의 random weight 가 0.999 weight 로 EMA 에 남아 half-life 693 step 동안 영향.   warmup=1000 이면 1000 step 후엔 그 영향 무시 가능.

**언제 warmup=0 / warmup>0 을 쓰나** — 사전학습·증류 가중치에서 출발하면 시작 weight 가 합리적이라 warmup=0 으로 충분.   fresh random init 이면 첫 ~700 step 동안 random weight 가 EMA 에 남으므로 warmup>0 (예: 1000) 권장.

### 4.3 update_interval=10 의 effective half-life

decay=0.999, interval=10 → 10 step 한 번 update:
- 100 step 학습 = 10 번 update
- effective decay per step ≈ 0.999^(1/10) ≈ 0.9999
- effective half-life ≈ 6931 / 10 ≈ 693 step

→ interval=1 (decay=0.999) 와 같은 step 단위 half-life 를 갖되, 비용은 1/10. 둘이 동등.

### 4.4 메모리 / 비용

1.17B params (fp32) 모델 가정:

| 항목 | 메모리 | 추가 비용 |
|---|---|---|
| EMA copy (GPU) | 4.7 GB | 0 (보관만) |
| EMA copy (CPU offload) | 4.7 GB CPU | + Host↔GPU 전송 (interval=10 이면 step 당 0.5GB/10) |
| update step | 0 (in-place) | 1.17B × (mul+add) ≈ 1ms / 10 step = 0.1ms/step |

→ 학습 비용 부담 미미. 메모리 부담이 큰 경우 CPU offload.

---

## §5. 코드 구조 — 클래스/함수와 책임

전형적인 경량 EMA 클래스 (`EMAModel`) 의 멤버 구성:

| # | 멤버 | 책임 |
|---|---|---|
| 1 | `__init__` | model deepcopy, requires_grad=False, device 옵션 |
| 2 | `_current_decay` | warmup 적용한 effective decay 계산 |
| 3 | `update` | 매 step 호출 — `ema ← decay·ema + (1-decay)·model` in-place |
| 4 | `get_model` | 추론용 EMA 모델 반환 |
| 5 | `state_dict` / `load_state_dict` | 체크포인트 저장/복원 (decay, warmup 포함) |

프레임워크 통합형은 여기에 더해 ① smoothing 식 계산 + 분산 reshard 처리, ② `summon_full_params` 류 context manager, ③ Event hook 등록 + half_life↔smoothing 변환 + ema_start 지연, ④ FSDP-aware EMA copy 컨테이너 등을 추가로 갖춘다 (§3-(E) 참조).

---

## §6. 학습 루프 통합 위치

### 6.1 옵션 기본값

```python
'ema'                : True,             # 표준 디폴트
'ema_decay'          : 0.999,
'ema_warmup'         : 0,
'ema_update_interval': 10,
'init_ema'           : None,             # 사전학습/증류 단계의 EMA weight path (선택)
```

### 6.2 build

```python
ema = EMAModel(model, decay=ema_decay, warmup=ema_warmup) if use_ema else None
```

→ `use_ema=False` 면 `ema is None`, 이후 모든 update/save 블록이 건너뜀.

### 6.3 사전학습 EMA 로드 (선택)

```python
if init_ema:
    ema_sd = torch.load(init_ema, map_location='cpu', weights_only=False)
    if isinstance(ema_sd, dict) and any(k.startswith('module.') for k in ema_sd.keys()):
        ema_sd = {k.replace('module.', ''): v for k, v in ema_sd.items()}
    ema.load_state_dict({"ema_model": ema_sd}, strict=False)
```

→ 사전 단계의 EMA decay history 를 본 학습 시작점으로 보존.

### 6.4 학습 step

```python
if (
    ema is not None
    and sync_gradients                          # ★ grad accumulation 시 실제 step 일 때만
    and (step + 1) % ema_update_interval == 0
):
    ema.update(model, step)
```

`sync_gradients` 체크 — grad accumulation 으로 micro-step 여러 번 모아서 한 번 step 하는 경우, 실제 optimizer.step 일어난 시점에만 EMA update. 안 그러면 EMA 가 micro-step 마다 update 되어 의미가 흐려짐.

### 6.5 체크포인트

```python
save(ema.state_dict(), save_path / "ema.pt")     # 주기적
save(ema.state_dict(), final_path / "ema.pt")    # 학습 종료
```

→ EMA 체크포인트 별도 저장 (`ema.pt`). model checkpoint (`model.pt`) 와 분리.

---

## §7. 권장 하이퍼파라미터 조합

| 항목 | 값 | 이유 |
|---|---|---|
| `ema` | True | inference 품질 (표준) |
| `ema_decay` | 0.999 | 50k ~ 200k step 규모에서 충분히 따라잡으면서 평활화 |
| `ema_warmup` | 0 (또는 fresh init 이면 >0) | 사전학습/증류 init 가정 시 0, random init 이면 1000 권장 |
| `ema_update_interval` | 10 | 비용 1/10, effective half-life 동일 |

**왜 EMA 가 inference 품질의 핵심인가** — 학습 마지막 step 의 raw weight 는 grad noise 에 민감. EMA 는 최근 N step 의 가중 평균이라 inference 시 더 안정. 측정상 EMA 켜면 FID 점수가 일관되게 더 좋음.

---

## §8. 구현 변형 정리 — 경량형 vs 통합형

| 항목 | 프레임워크 통합형 | 경량 함수형 |
|---|---|---|
| 통합 방식 | `Algorithm` Event hook | 학습 루프에서 직접 `update()` |
| 분산 reshard | 자동 (`summon_full_params`) | 가속 라이브러리가 처리 |
| smoothing 표현 | `half_life="1000ba"` 문자열 + `smoothing` float | `decay` float 만 |
| `ema_start` | 학습 진행률/step 으로 EMA 시작 지연 | 미지원 (warmup 으로 비슷한 효과) |
| `update_interval` | "1ba" / "1ep" 시간 단위 | int (step 단위) |
| named_buffers | 함께 update | 미포함 (named_parameters 만) |
| device 옵션 | 보통 없음 | 있음 (CPU offload) |
| state_dict 저장 항목 | ema_started flag 등 | decay, warmup |

---

## §9. 실패 모드 — 자주 발생하는 함정

### 9.1 EMA 추론 잊고 raw weight 로 inference

→ 학습 종료 후 raw weight 로 inference. EMA 의 평활화 효과 안 받음. FID 점수 ↓.   해결: inference pipeline 이 `ema_model = ema.get_model()` 사용.

### 9.2 grad accumulation 환경에서 micro-step 마다 update

→ EMA 가 grad accumulation 의 micro-step 마다 update 되면 의미 흐려짐.   해결: `sync_gradients` 체크.

### 9.3 EMA copy 와 raw model 의 device mismatch

→ DDP/FSDP 환경에서 model 은 GPU, EMA 가 CPU 에 있으면 매 step `to(device)` 비용.   해결: `ema_p.device` 캐싱, non_blocking 전송.

### 9.4 체크포인트 EMA 빠뜨림

→ resume 시 EMA copy 가 없으면 random-init 으로 다시 시작해 EMA half-life 만큼 학습 손실.   해결: `ema.pt` 별도 저장 + load_state_dict.

### 9.5 update_interval 너무 큼

→ interval=100 + decay=0.999 → effective decay 가 0.999^(1/100) ≈ 0.99999. 짧은 학습에서 거의 안 따라옴.   해결: decay × interval ≈ const 유지 (0.999 × 10 = 표준).

### 9.6 warmup=0 + fresh init

→ random-init weight 가 EMA 에 0.999 weight 로 ~3500 step 영향. 학습 초반 EMA inference 가 random.   해결: fresh init 이면 warmup>0 (예: 1000) 권장. 사전학습/증류 init 일 때만 warmup=0 안전.

---

## §10. 한 줄 정리

> **EMA 는 학습 weight 의 지수이동평균을 별도 copy 로 유지해 inference 시 raw 보다 안정적인 weight 를 제공.**
> 학습 루프에서 `update()` 를 직접 호출하는 경량 `EMAModel` 단일 클래스 ~90 줄로 구현 가능하며,
> inference 품질 향상의 표준 축이다.

---

## §11. 향후 확장 가능한 부분

- **named_buffers update** — RMSNorm 의 running_mean 같은 buffer 도 EMA 가 따라가면 inference 일관성 ↑ (`itertools.chain(named_parameters, named_buffers)`).
- **half_life 식 인자** — `decay = exp(-log(2) / half_life)` 변환 helper. 직관적인 hyperparam tuning.
- **ema_start** — 학습 초반 N step 동안 EMA update skip. random init 영향 차단.
- **CPU offload 자동화** — GPU 메모리 부족 시 자동으로 CPU 로 보내는 fallback.
- **Polyak averaging 옵션** — 학습 마지막 N step 만 단순 평균 (SWA, Stochastic Weight Averaging) 변종.
- **다중 EMA (multi-decay)** — 0.999 + 0.9999 두 decay 동시 유지 → inference 시 best 선택. 더 나아가 학습 후 임의 길이의 EMA 를 사후 재구성하는 **Post-hoc EMA** (EDM2, Karras et al. 2024, [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)) 로 발전.
