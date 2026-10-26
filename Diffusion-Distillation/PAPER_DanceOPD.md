# DanceOPD: On-Policy Generative Field Distillation

## 0. 메타 정보

| 항목 | 내용 |
|---|---|
| 제목 | DanceOPD: On-Policy Generative Field Distillation |
| 저자 | Wei Zhou, Xiongwei Zhu, Zelin Xu, Bo Dong, Lixue Gong, Yongyuan Liang, Meng Chu, Leigang Qu, Lingdong Kong, Wei Liu†, Tat-Seng Chua (ByteDance Seed, NUS, UMD, HKUST) |
| 게재 | arXiv 2606.27377 (v1 2026-06-25, v3 2026-08-15), 기술 리포트 42쪽·그림 14·표 9, CC BY-SA 4.0 |
| 분야 | cs.CV / cs.CL / cs.LG — flow matching 다능력 통합(여러 능력을 한 모델에 합치기), on-policy distillation |
| 링크 | https://arxiv.org/abs/2606.27377 · https://arxiv.org/pdf/2606.27377 · 프로젝트 https://danceopd.github.io/ |
| 코드 | 논문 저장소 https://github.com/worldbench/DanceOPD (직접 클론해 분석, §5-2; 커밋 f856e87, 2026-10-04, Apache-2.0) / DiffSynth-Studio `examples/*/model_training/special/opd/` (OPD 변형으로 포함, 본문 §5) |
| 베이스/데이터 | Z-Image(주력), SD3.5-M(실사 흡수만). 편집 데이터 OmniEdit. 평가 GEditBench-EN, GenEval. 실사 보상 모델은 **비공개** |
| 검증 범위 | PDF 전문 + 부록(구현·하이퍼파라미터) 직접 읽음. 선행 DiffusionOPD(2605.15055)는 별도 문서 [[PAPER_DiffusionOPD.md]]. DiffSynth 문서·스크립트·`flow_match.py`, 공개 저장소(core/·백엔드 2종·설정·테스트) 직접 읽음. 학습은 돌려 보지 않음 |

---

## 0-1. 이 논문은 무엇인가? (What)

*이 절을 두는 이유: 용어와 수식에 들어가기 전에 "이 논문이 무슨 일을 하고, 무슨 일은 하지 않는지"를 먼저 잡아 두어야, 뒤의 DiffusionOPD·DiffSynth 스크립트 비교가 헷갈리지 않는다.*

### 한 문장
**여러 가지를 따로 잘하는 전문가 모델들의 실력을, 학생 모델 하나에 물려주는 학습법이다.** 논문은 이를 "능력(capability)을 합친다(compose)"고 부른다. (스텝 수를 줄이는 압축법이 아니다 → 아래 "하지 않는 일")

### 풀어 쓰면
이미지 생성 모델에게 요구하는 일은 여러 종류다.
- **T2I**: 글을 보고 그림 그리기
- **로컬 편집**: 사진의 일부만 바꾸기 (예: "컵을 빨간색으로")
- **글로벌 편집**: 사진 전체를 바꾸기 (예: "수채화 스타일로", "밤 풍경으로")

문제는 한 모델에 이걸 다 시키면 **서로 방해**한다는 점이다. 편집을 가르치면 그림 그리기가 나빠지고, 로컬 편집("나머지는 그대로 둬")과 글로벌 편집("전부 바꿔")도 원하는 방향이 반대라 충돌한다.

DanceOPD의 해법:
1. 일마다 **전문가(선생)**를 먼저 둔다. 그림 전문가, 편집 전문가 등.
2. 학생 하나가 이들의 실력을 **하나로 모아 물려받는다.**
3. 이때 샘플마다 선생을 **하나만** 정해서 따라 하게 한다. 여러 선생의 의견을 평균 내지 않는다(평균하면 어느 능력의 질의도 아닌 방향이 되므로).
4. 따라 하는 위치는 **학생이 직접 걸은 길 위의 한 지점**이다(on-policy). 선생이 정한 길이 아니라 학생이 실제로 가는 길에서 "여기선 어느 쪽?"을 묻는다.

### 비유
학생 한 명이 수학·영어·과학을 다 잘해야 한다. 세 과목을 섞어 가르치면 헷갈리고, 세 선생님 의견을 평균 내면 무슨 풀이인지 모르게 된다. 그래서 문제마다 담당 선생님 한 분을 정하고, 학생이 직접 푼 풀이 지점에서 그 선생님의 안내를 따라 하게 한다.

### 하지 않는 일 — 스텝 감소가 목적이 아니다
- 논문 부록: DanceOPD는 "trajectory-compression distillation"이 아니라 **지점별 velocity field 매칭**을 하므로, 학습 스텝 격자와 평가 스텝 격자가 같을 필요가 없다.
- 실제로 학습 rollout은 **16스텝**, 평가는 **28스텝**이다(GEditBench CFG 7.0, GenEval CFG 3.5). 스텝을 줄이는 것이 목적이었다면 평가 스텝이 학습보다 많을 이유가 없다.
- 헷갈리기 쉬운 이유 세 가지:
  1. 같은 OPD 계열이라 틀이 스텝 증류에도 쓰일 수 있고, DiffSynth 문서가 OPD를 "증류 가속 학습"으로 소개하면서 DanceOPD를 인용한다.
  2. 논문에 **CFG 흡수** 실험(D)이 있다. 가이던스를 학생에 내장하면 추론 forward가 2회에서 1회로 줄지만, 이것은 응용 중 하나이고 이득도 작다(eval-only CFG 대비 +1.4%). 스텝 수는 그대로다.
  3. 공개 설정에 Z-Image-Turbo가 있으나 README상 코드 경로 점검용 대체물이다.

| | 목적 | 스텝 수 |
|---|---|---|
| DanceOPD | 여러 능력을 한 모델에 합치기 | 학습 16, 평가 28 (줄이지 않음) |
| DiffSynth 스크립트 | 50스텝·CFG 4를 8스텝·CFG 1로 압축 | 50 → 8 |

### 선행 연구 DiffusionOPD는 무엇을 통합하나
DiffusionOPD(2605.15055)의 "과제별 RL teacher들을 학생 하나로 통합"은 같은 발상이다. 원본 모델을 보상 하나씩(GenEval, OCR, 미학)으로 RL해 **과제별 전문가**를 따로 키운 뒤, 원본에서 출발한 학생이 자기 생성 궤적에서 각 과제의 전문가를 돌아가며 따라 하게 한다. 보상 여러 개를 한 모델에 동시에 RL하면 서로 방해하는 문제를 "따로 키우고 나중에 합친다"로 푸는 것이다. DanceOPD는 teacher가 RL 전문가가 아니라 SFT 편집 모델 등이라는 점, 질의를 전 스텝에서 한 지점으로 줄였다는 점이 다르다(Q2 표).

---

## 1. 주요 용어 사전 (Glossary)

*이 절을 두는 이유: 논문이 "field", "routing", "query" 같은 말을 새로 쓰므로, 본문 전에 뜻을 맞춰 두면 이후 설명이 쉽다.*

### 핵심 개념
| 용어 | 쉬운 풀이 |
|---|---|
| OPD (On-Policy Distillation, 온폴리시 증류) | 선생이 정해 준 길을 따라 걷는 게 아니라, **학생이 직접 걸은 위치**에서 선생에게 "여기선 어느 쪽?"을 묻는 증류 |
| velocity field (속도장, 벡터장) | 각 지점(latent 상태)에서 "노이즈→이미지로 어느 방향으로 가라"를 알려 주는 화살표 지도. 모델 하나 = 지도 하나 |
| 능력 = field | T2I 모델, 편집 모델, 실사 모델, CFG까지 전부 "같은 공간 위의 화살표 지도"로 본다는 관점 |
| 질의(query) | 학생이 있는 한 지점에서 선생(field)의 화살표를 한 번 읽어 오는 일 |
| hard routing | 샘플마다 선생을 **딱 하나만** 지정. 여러 선생을 평균 내지 않음 |
| soft mixing | 여러 선생의 화살표를 평균 내어 하나의 정답으로 쓰는 방식(논문이 피하려는 것) |
| 저노이즈 / 고노이즈 | 이미지가 거의 완성된 쪽(low-t) / 거의 노이즈인 쪽(high-t) |
| stop-gradient (sg) | 그 값을 **상수로 취급**, 그쪽으로는 미분하지 않음 |
| K | 한 번의 rollout에서 질의(손실 계산)하는 상태의 수 |

### 비교 대상
| 용어 | 쉬운 풀이 |
|---|---|
| Joint Training | 여러 능력의 데이터를 섞어 한꺼번에 학습 |
| Weight Merge | 전문가들의 가중치를 보간해 합침 |
| Off-Policy Distill | 같은 hard routing + MSE지만, 질의 상태를 학생 rollout이 아니라 정답 이미지에 노이즈를 입힌 상태로 잡음 |
| DiffusionOPD | 학생 rollout 전 스텝(K=N)에서 teacher 전이 평균을 맞추는 선행 연구. → [[PAPER_DiffusionOPD.md]] |
| Flow-OPD | SDE rollout + PPO식 clip + GRPO 그룹으로 teacher KL을 보상처럼 쓰는 선행 연구 |

### 평가
| 용어 | 쉬운 풀이 |
|---|---|
| GEditBench-EN | 편집 능력 벤치. 6개 범주(주체 추가·교체·배경·스타일·색상·제거) 평균 |
| GenEval | 객체·개수·색·위치를 보는 T2I 구성 능력 벤치 |

---

## 2. 논문 요약 (TL;DR)

- **한 줄**: 능력(T2I/편집/실사/CFG)을 각각 "속도장"으로 보고, **샘플마다 선생 하나**를 골라 **학생이 직접 걸은 저노이즈 상태 한 곳**에서 속도를 MSE로 맞춘다.
- **문제**: T2I·로컬 편집·글로벌 편집을 한 모델에 합치면 서로 간섭한다(편집이 T2I를 깎고, 로컬과 글로벌 편집끼리 충돌).
- **해결**: 세 질의 설계(어느 선생 / 어느 상태 / 몇 개)를 각각의 실패 모드에 대응시켜 고정.
- **검증**: Z-Image에서 GEditBench +8.1%(최고 OPD baseline 대비) 등. 단 baseline이 전부 자체 재현이고 통계가 없음(§7).

## 3. 핵심 기여 (논문 주장)

1. 여러 능력을 한 모델에 합치는 문제를 "on-policy generative field distillation"으로 정식화.
2. 세 가지 질의 정렬 문제(target-field ambiguity, state-distribution mismatch, trajectory-query correlation)를 진단하고 각각에 대한 설계를 제시.
3. T2I+편집, 로컬+글로벌 편집, 실사 흡수, CFG 흡수 4개 설정 + 폭넓은 ablation.

---

## 4. 주요 알고리즘 설명

### 4.1 세 가지 설계 (각각 실패 모드 하나에 대응)

*이 절을 두는 이유: 이 논문의 전부가 "세 질문에 어떻게 답했나"이므로, 실패 모드와 처방을 짝지어 한 번에 보여 준다.*

| 실패 모드 | 처방 | 알고리즘 |
|---|---|---|
| **target-field ambiguity**: 여러 teacher를 평균 내면 어떤 능력의 질의도 아닌 방향이 됨 | **Hard routing**: 샘플마다 teacher를 정확히 하나만 배정(확률 균등, 튜닝 안 함) | 샘플마다 teacher 하나 선택 |
| **state mismatch**: 데이터나 teacher 궤적 상태에서 물으면 학생이 실제 방문하는 상태가 빠짐 | **On-policy 질의**: 학생 자신의 rollout 상태(stop-gradient)에서 teacher에게 물음 | 학생이 16스텝 Euler ODE로 rollout |
| **trajectory correlation**: 같은 rollout의 여러 상태는 prompt·seed·경로가 같아 gradient가 상관됨 | **저노이즈 쪽 단일 질의**: 샘플당 K=1, Beta(5,2)로 깨끗한 쪽에 치우쳐 샘플링 | 상태 1개에서 velocity MSE |

### 4.2 한 스텝 (Algorithm 1)

1. 라우팅: 능력 버킷 m 하나를 뽑고 그 버킷의 (입력, 조건)을 뽑는다.
2. 학생이 초기 노이즈에서 16스텝 rollout(stop-gradient).
3. 저노이즈 쪽 좌표 s를 하나 뽑아 시간 t를 정하고, 그 상태를 질의점으로 쓴다.
4. 동결된 teacher의 속도 u를 그 상태에서 읽는다.
5. 학생 속도와 u의 MSE로 한 번 업데이트.

손실(식 6): 라우팅된 teacher 속도와 학생 속도의 차이 제곱을, 학생 rollout 상태에서 평균. 가중치 없음.

CFG도 "guided velocity field"(무조건 속도 + 가이던스 배수 × (조건 − 무조건))로 보고 같은 손실로 teacher 자리에 넣을 수 있다(식 7).

### 4.3 기본 하이퍼파라미터 (Table 5)

| 항목 | 값 |
|---|---|
| 학습 대상 | DiT LoRA, rank 128 |
| rollout | 16스텝 Euler ODE |
| K | 1 |
| 시간 샘플링 | Beta(5,2)로 저노이즈 편향 |
| 라우팅 | 활성 버킷 균등 (2버킷 1:1, 3버킷 1:1:1) |
| 옵티마이저 | AdamW, lr 2e-4, gradient accumulation 4 |
| 초기화 | 속성 편집 SFT LoRA (가장 가까운 전문가) |
| 평가 | GEditBench 28스텝·CFG 7.0, GenEval 28스텝·CFG 3.5 |

### 4.4 이론 부록 요약

*이 절을 두는 이유: "왜 하필 MSE인가"에 대한 논문의 근거를 한 곳에 모아 둔다.*

- **KL = MSE**: 학생과 teacher가 같은 공분산의 가우시안 한 스텝 전이를 가진다고 가정하면, KL은 속도 차이 제곱에 스텝 길이 제곱을 곱한 가중 MSE가 된다(식 8~10). 본문은 teacher 타깃이 결정론적이라 가중 없는 MSE가 더 안정적이라고 주장.
- **온폴리시 불일치 한계**: teacher field가 Lipschitz라면, off-policy 상태에서 읽은 값과 학생 상태에서 읽은 값의 차이가 상태 거리에 비례해 커진다(§7.2). 사실상 자명한 부등식.
- **gradient**: 롤아웃 솔버를 통한 미분은 하지 않고 국소 속도 예측만 미분하는 1차 추정량(식 13).

---

## 5. DiffSynth 스크립트와의 관계

*이 절을 두는 이유: 사용자가 함께 보라고 한 스크립트가 논문 방법을 얼마나 실제로 구현했는지 가려 두기 위해서다. 스크립트 전체 분석(x0 점프형 목표, 시그마 하드코딩 등)은 [[PAPER_DiffusionOPD.md]] §4.2·§5.2에 있으므로 여기서는 DanceOPD와 관련된 부분만 둔다.*

### 5.1 DiffSynth 문서가 말하는 것
- 문서(Training/Distillation)는 OPD를 "학생이 자기 궤적을 만들고 teacher의 **다단계 출력**이 학습 타깃이 되는 온라인 증류", 판별기 불필요라고 소개하고 **"OPD 사용을 강력 권장"**한다(Direct Distill, DMD2 대비).
- 인용: DiffusionOPD(2605.15055), DanceOPD(2606.27377).
- 변형은 파라미터로 구분:
  - `--opd_num_distill_steps`: DiffusionOPD 스타일(궤적 여러 지점)
  - `--opd_min_distill_idx`: DanceOPD 스타일(문서 문구: "`input` is sampled only in the low-noise-content stage")
- 스크립트는 `examples/<model>/model_training/special/opd/`에 모델별로 있다: FLUX.2-klein-base-4B, Qwen-Image-2.1(+Turbo 3스텝), MOSS-Image-SFT.

### 5.2 코드 확인 결과 — 옵션이 꺼져 있음
`train_opd_lora.py` 91~94행:
```
distill_idx = torch.randperm(len(student_sigmas) - min_distill_idx) + min_distill_idx
for idx in distill_idx[:num_distill_steps].tolist():
    ...
    idx = torch.randint(0, len(student_sigmas), (1,)).item()   # 94행: 위에서 고른 idx를 덮어씀
```
- 92행에서 뽑은 idx를 94행이 `[0, 전체)` 균등 무작위로 덮어쓴다. 따라서 `--opd_min_distill_idx`는 어떤 값을 줘도 효과가 없고, `--opd_num_distill_steps`는 반복 횟수만 늘린다(중복 가능).
- Qwen-Image-2.1, Flux2, Moss-Image용 `train_opd_lora.py`에 같은 줄이 있다(Moss는 102행). 최신 커밋 "Support OPD Training (#1737)", 2026-10-13. 세 파일 모두 복사된 코드로 보인다.
- sh에 `--opd_min_distill_idx`를 줘도 인자는 `OPDTrainingModule`까지 정상 전달(156행)되지만 94행 때문에 무시된다. **sh만으로는 고칠 수 없고 py의 94행 삭제가 필요**하다.
- 이 스크립트의 sh는 두 옵션을 지정하지 않아 기본값(1, 0)으로 돈다 → 결과적으로 "전 스텝 균등 무작위 1개 질의".

### 5.3 구현 여부 정리
| 구분 | 상태 |
|---|---|
| 옵션 정의·인자 전달·선택 로직(91~92행) | 구현됨 |
| 옵션이 실제 동작에 반영 | **안 됨**(94행 덮어쓰기) |
| hard routing, 다중 teacher | 구현 안 됨(teacher = LoRA 끈 원본 1개) |
| Beta(5,2) 저노이즈 분포 | 구현 안 됨(`randperm` 균등 → 94행으로 다시 균등) |
| teacher 타깃 | 논문의 "즉석 velocity 1회"가 아니라 **teacher가 구간을 CFG로 여러 스텝 내려간 뒤 x0를 velocity로 환산**(문서의 "teacher's multi-step output"과 일치, DiffSynth 고유 설계) |

94행을 지우면 "min_distill_idx 이상에서 균등 선택"이라는 근사적 DanceOPD 스타일이 된다. 라우팅·Beta 분포는 여전히 없다.

### 5.4 checkpointing + LoRA 스케일 전환 (스크립트 공통 문제)
- forward 끝에서 `switch_model("teacher")`로 LoRA 스케일이 0이 된 채 backward가 일어나고, `--use_gradient_checkpointing`(non-reentrant)이 켜져 있으면 재계산이 스케일 0 상태로 일어나 기울기가 어긋난다.
- 장난감 검증(peft 0.17.1, torch 2.2.2, 단층 선형+tanh, LoRA B를 N(0,0.5)로 초기화):

| 조건 | LoRA 기울기 합 |
|---|---|
| checkpoint 없음 | 69.74 |
| checkpoint 있음, 스케일 전환 없음 | 69.74 |
| checkpoint 있음, 스케일 0 전환 | **286.67 (다름)** |

- [[PAPER_DiffusionOPD.md]] §5.2(3)의 검증은 다른 장난감 설정이라 수치가 다르다(0.7121 vs 0.7384). 어긋남이 생긴다는 결론은 같다. 실제 Qwen 모델에서의 크기는 미측정.

---

## 5-2. DanceOPD 공개 저장소 분석 (worldbench/DanceOPD)

*이 절을 두는 이유: 논문이 말한 알고리즘이 공개 코드에서 의도대로 구현돼 있는지, 그리고 코드만으로 논문 결과를 재현할 수 있는지 가려 두기 위해서다.*

저장소 전체(코드 약 5,400줄)를 클론해 `core/`, 두 백엔드(SD3.5, Z-Image), 설정, 테스트를 읽었다. 학습은 돌려 보지 않았다. 클론한 저장소에는 커밋이 1개 있고(f856e87, 2026-10-04) 제목은 "Accelerate DanceOPD with query-prefix rollouts"이다.

### 5-2.1 저장소의 정체: 알고리즘 코드만 있고 재현 자산은 없음
README가 직접 밝히는 내용이다.
- **논문 teacher LoRA와 student 체크포인트는 공개하지 않는다.** "does not release the paper teacher LoRAs or student checkpoints".
- `configs/paper/`의 Z-Image 설정은 student init이 `/path/to/unspecified_main_table_student_init.safetensors`로 비어 있다. 주석: "The exact Table-2 main-run initialization is not stated in the manuscript".
- 이는 §7의 **초기화 의존도(T2I 초기화 대비 +204%)**와 직결된다. 가장 민감한 변수가 논문에도 코드에도 명시돼 있지 않다. 공개 코드로는 메인 표를 재현할 수 없다.
- 공개 설정(`configs/public/zimage.yaml`)은 Z-Image-Turbo + Ostris 학습 어댑터를 student로, 원본을 teacher로 쓴다. README도 "스모크용 대체물"이라고 인정한다. 능력 통합이 아니라 코드 경로 점검용이다.
- 라이선스는 Apache-2.0(코드만), 모델 가중치는 각자의 약관을 따른다.

### 5-2.2 핵심 알고리즘 구현 (논문과의 대조)

| 논문 요소 | 코드 위치 | 구현 상태 |
|---|---|---|
| hard routing | `core/routing.py`, `engine.py` | 구현됨 |
| 온폴리시 rollout | `zimage_diffsynth.py::_rollout` | 구현됨 |
| 저노이즈 단일 질의 Beta(5,2) | `core/timestep.py::sample_query_indices` | 구현됨 |
| velocity MSE | `core/loss.py` | 구현됨 (한 줄) |
| 기본값 | `config.py` | 논문 Table 5와 일치 (rank 128, lr 2e-4, grad_accum 4, 16스텝, K=1) |

- **라우팅** (`engine.py`): 최적화 스텝마다 route 하나를 뽑고(가중치 기반, 기본 균등), 그 route의 데이터에서만 `grad_accum`개 마이크로 배치를 뽑는다. 분산 학습에서는 rank 0이 뽑아 전체에 broadcast. 논문의 "한 업데이트 = 한 능력(G=1)"과 정확히 맞다.
- **손실 한 스텝** (`compute_loss`): 학생이 16스텝 Euler ODE로 rollout(no_grad) → 인덱스 하나를 Beta(5,2)로 뽑음 → 그 상태에서 teacher 속도(no_grad)와 학생 속도(grad)를 구함 → 속도 MSE. 논문 Algorithm 1을 그대로 옮긴 구조다.

### 5-2.3 눈여겨볼 발견

**(1) query-prefix 최적화: 논문에 없는 가속**
- `_rollout(selected_indices=...)`는 질의 인덱스까지만 rollout하고 중단한다(`if pos == stop_index: break`). ODE가 결정론적이라 결과 분포는 동일하다(테스트 `test_selected_states_match_full_rollout...`가 확인).
- 논문 부록 8.1은 "온폴리시 방법은 N스텝 rollout 전체를 먼저 수행"이라는 비용 모델이다. 이 커밋은 논문 v3(2026-08-15) 이후(2026-10-04)라 논문의 비용 비교에는 반영되지 않았다.
- Beta(5,2) 분포상(N=16, 직접 계산) 평균 약 11.9스텝만 굴리면 된다. 약 25% 절감. 마지막 스텝이 뽑힐 확률은 4.9%, 질의의 89%가 후반 절반(인덱스 8 이상)에 몰린다.

**(2) "저노이즈"는 인덱스 기준이지 노이즈 수준 기준이 아님**
- Beta 샘플은 rollout 인덱스로 변환되므로 스케줄이 시프트된 경우 실제 σ와 어긋난다.
- 중앙 질의(u≈0.736, 인덱스로 약 12/16)는 Z-Image 스케줄 시프트가 3이라면(**확인 못 함**) σ≈0.52 정도로 "깨끗한 쪽"이 아니라 중간 노이즈다. 논문의 "semantic-side low-noise" 서술과 실제 샘플링 분포가 다를 수 있어 시프트 값 확인이 필요하다.

**(3) 재현한 baseline의 세부**
- **DiffusionOPD** (`compute_loss`): 가중치를 `0.5 / rollout_steps²`(고정)로 두고 K=16개 MSE를 합산한다. 주석에 "uniform grid 가정". 시프트된 스케줄의 실제 간격과 다르지만, AdamW라 전체 크기는 상쇄되고 스텝 간 상대 가중이 사실상 균등이다. → §7.2의 "손실 가중이 밀집 질의 열세와 섞였을 수 있다"는 우려는 **약해졌다**(남는 차이는 스텝 균등 가중).
- **Flow-OPD** (`flowopd.py`): 보상 = −(teacher와의 전이 평균 차이 / 분산), ±5 클립, PPO식 비율 클립 1e-4. 이 1e-4는 Flow-GRPO의 기본값이다. → 앞서 "사실상 업데이트를 막는 값"이라 한 비판은 **철회**. 다만 advantage에 그룹 평균 기준선이 없다(원 Flow-OPD가 어떻게 하는지는 확인 못 함). 분산이 큰 추정량일 수 있다.
- 세 방법이 같은 하이퍼파라미터(lr 2e-4, 3000스텝)를 쓴다. baseline 튜닝 여부는 코드에 흔적이 없다.

**(4) 메모리 낭비**
- 각 teacher를 DiT 전체 복사본으로 GPU에 올린다(`copy.deepcopy`). T2I+편집 설정에서 DiT가 3벌(학생, T2I teacher, 편집 teacher).
- `t2i` teacher는 LoRA 없는 원본이라 "학생 LoRA를 끈 상태"로 대체할 수 있다(DiffSynth 스크립트가 쓰는 방식). Z-Image 6B 기준 약 12GB씩이라 의미 있는 낭비.
- 마이크로 배치는 1샘플, 유효 배치는 `grad_accum × GPU 수`.

**(5) DiffSynth 내부 API 의존**
- `_encode_prompt`에 폴백이 4단계, 예외도 "버전 확인" 안내뿐이다. README가 검증한 커밋을 명시(2.1.2, `6343ded`)한 것은 좋지만 버전이 달라지면 깨지기 쉽다.
- 편집 경로는 DiT의 다중 이미지 인터페이스를 직접 호출하고 출력 부호를 뒤집는다(`_conditioned_velocity`의 `-output`). 부호 규약이 틀리면 에러 없이 나쁜 학습이 될 수 있는 지점이라 기준 모델 출력과 대조하는 편이 안전하다.

### 5-2.4 좋은 점
- 알고리즘 핵심이 `loss.py` 한 줄, `timestep.py`, `routing.py`로 작고 읽기 쉽다.
- 설정 검증이 꼼꼼하다(라우트가 가리키는 teacher 존재, `accumulation_groups`를 다른 방법에서 쓰면 거부, 존재하지 않는 로컬 경로 사전 점검).
- 논문 설정 템플릿과 공개 스모크 설정을 분리해 대체 가중치를 논문 재현 자산처럼 보이게 하지 않는다.
- 단위 테스트 6개 파일(약 760줄). query-prefix가 전체 rollout과 동치인지 손실과 gradient까지 검증한다.

### 5-2.5 종합
- **논문 알고리즘의 구현으로는 충실하다.** 라우팅, 온폴리시 상태, 저노이즈 단일 질의, MSE가 전부 코드에 있다. DiffSynth 스크립트의 `min_distill_idx` 덮어쓰기 버그와 대조적이다.
- **하지만 논문 결과의 재현은 불가능하다.** teacher와 student 초기화(결과를 좌우하는 변수)가 비공개이고 논문에도 명시돼 있지 않다. 공개 코드가 보증하는 것은 "알고리즘이 의도대로 동작한다"까지다.

---

## 6. 실험 요약

*이 절을 두는 이유: §7의 비판이 어떤 숫자에 근거하는지 표로 남긴다.*

| 설정 | 주요 결과 (논문 서술, 모두 상대 증가율) |
|---|---|
| A. T2I+편집 합성 | GEditBench: 최고 재현 OPD baseline +8.1%, 편집 source +8.5%. GenEval: T2I source +2.0%, 최강 합성 baseline +1.6%. DiffusionOPD 대비 배경변경 +21.9%, 스타일 +21.3%, 색상 +5.5% |
| B. 로컬+글로벌 합성 | GEditBench: 최고 경쟁 합성 baseline +16.1%, 로컬 source +7.9%. GenEval: 최강 합성 baseline +2.5% |
| C. 실사 흡수 (SD3.5-M) | 보상: off-policy 대비 +9.9%, teacher-student 격차의 85.3% 메움. T2I 점수는 학생 anchor 대비 +7.6%, off-policy와 0.1% 이내 |
| D. CFG 흡수 | 최적 조합이 train-only 흡수 대비 +7.6%, eval-only CFG 대비 **+1.4%**. 과다 조합은 최적 대비 −31.2% |

### 진단·ablation

| 비교 | 결과 |
|---|---|
| hard routing vs soft mixing | MSE에서 +15.2%, KL에서 +10.6% (배경변경 +20.8%, 제거 +26.8%) |
| step 교대 vs 같은 스텝 누적(G=3) | 누적이 −4.6% (K=2,G=3은 −22.8%) |
| ODE vs SDE (밀집 K=2,G=3) | SDE가 +18.4% 회복하나 단일 질의 기본값보다 −8.6% |
| 저노이즈 vs 중간·고노이즈 질의 | 각각 +23.7%, +19.5% (2k 스텝 시점) |
| 단일 질의 vs 가중 밀집 K=2,4,8,16 | 기본값이 +16.6%, +7.9%, +10.2%, +12.2% |
| 손실 | 가중 MSE·DMD-EMA +2.8%, consistency +4.1%, KL-σ² +4.5%, DMD2·AuxFeat +15.6~21.1% |
| rollout 스텝 | 16스텝이 8/20/28스텝 대비 GEdit +0.2/+3.0/+0.9% |
| 초기화 | 로컬 편집 초기화가 합성(merge) +37.2%, 글로벌 +112.8%, **T2I +204.4%** |

### 비용 (부록 8.1, N=16, 물리 배치 8, GRPO 그룹 16)
- DanceOPD: rollout N스텝 + K=1 질의.
- DiffusionOPD: rollout + K=16 질의.
- Flow-OPD: 거기에 GRPO 그룹 16이 물리 배치 8을 넘어 마이크로 배치 2배 + SDE·PPO 오버헤드.

---

## 7. 비판적 평가

### 7.1 좋은 점
*이 절을 두는 이유: 숫자를 못 믿는다는 비판과 별개로, 가져다 쓸 만한 부분을 분리해 둔다.*

1. **ablation 설계가 깔끔하다.** 라우팅(hard/soft), 질의 상태, 질의 개수, 손실을 하나씩 분리했다. "밀집 질의가 나쁘다"를 SDE로 상관을 줄이면 18.4% 회복된다는 **예측 검증**으로 뒷받침한 점이 좋다.
2. **CFG 흡수의 αβ 상호작용 지적.** 학습 CFG(α)와 추론 CFG(β)가 곱으로 쌓인다. 증류된 모델을 기본 CFG로 평가하면 과유도가 되어 점수가 31.2% 폭락한다. 실무에서 실제로 밟는 함정.
3. **rollout 스텝 수를 평가 스텝 수와 맞출 필요가 없다**는 관찰(16스텝 학습이 8/20/28스텝보다 좋거나 비슷).
4. 한계(공통 field 가정, 사전 정의 라우팅)를 스스로 적었다.

### 7.2 비판할 점
1. **베이스라인이 전부 자체 재현이다.** DiffusionOPD는 코드가 없어서 직접 재현. Flow-OPD는 MAR을 꺼서(β=0) 공식 레시피와 다르다. (PPO clip 1e-4는 처음에 "업데이트를 막는 값"이라 적었으나, 공개 저장소(worldbench/DanceOPD)를 읽어 보니 Flow-GRPO 계열의 기본값을 따른 것으로 보여 철회한다.) 공정성을 검증할 방법이 없어 "+8.1%"의 신뢰도가 낮다.
2. **초기화 의존도가 극단적이다.** 로컬 편집 초기화가 T2I 초기화보다 204%, 글로벌 초기화보다 112.8% 좋다. 방법의 이득보다 "어디서 시작하느냐"가 훨씬 크다. 합성이 성공했다기보다 "가장 가까운 전문가에서 출발해 조금 보정한다"에 가까울 수 있다.
3. **seed·신뢰구간·반복 실험이 없다.** 퍼센트는 전부 상대 증가율이고 2k 스텝 시점 단일 체크포인트 비교가 많다. low-t vs median-t 같은 타이밍 ablation이 수렴 후에도 유지되는지 모른다. GenEval +2.0% 같은 작은 차이는 노이즈일 수 있다.
4. **비용 비교가 유리하게 짜였다.** 비교 대상 DiffusionOPD는 K=N=16이라 비용이 16배로 잡힌다. K=1 vs K=16의 속도 우위는 구조상 당연하다.
5. **"밀집 질의가 나쁘다"는 손실 가중과 섞였을 수 있다.** 재현한 DiffusionOPD는 Δt² 가중이 붙은 손실(식 49)이다. "상관된 상태 탓"이라는 설명은 SDE 회복 실험이 간접 증거일 뿐, 가중·학습률 상호작용은 배제되지 않았다.
6. **보상 모델이 비공개다.** 실사 흡수의 핵심 지표를 재현할 수 없다.
7. **이론부가 약하다.** KL-MSE 동치는 같은 분산의 가우시안 커널을 가정하고, 온폴리시 부등식은 Lipschitz 상수 하나로 끝나는 자명한 수준이다. 설계를 정당화하기보다 사후 정리에 가깝다.
8. **"능력 창발"(4.4절)은 정성 사례 5개다.** 비교 대상 방법들이 편집 브랜치를 복사한다는 설명은 가설일 뿐 검증이 없다.
9. **CFG 흡수 이득이 작다.** 최적 조합이 eval-only CFG 대비 +1.4%뿐이다. 품질을 올리기보다 **NFE를 절반으로 줄이는 계산 절약**이 실제 가치다.

**한 줄 평**: 설계 원리는 명확하고 ablation 구조도 좋다. 다만 재현 가능한 baseline·통계·공개 보상이 없어 숫자를 그대로 믿기 어렵다. 가장 재사용 가치가 높은 것은 hard routing과 αβ 관찰이다.

---

## 8. 💬 Q&A

### Q1. DiffusionOPD가 베이스 논문이야?
*이 절을 두는 이유: 선행 연구와 베이스를 혼동하면 기여를 잘못 평가하게 된다.*

**아니요. 논문이 "베이스"라고 밝히지는 않고, 가장 가까운 선행 연구이자 비교 baseline이다.**

- 논문은 DiffusionOPD[53], Flow-OPD[24], D-OPSD[46]을 같은 줄에 "최근 flow 모델 OPD 방법"으로 나열하고, DanceOPD는 이들과 "complementary(상호 보완)"라고 쓴다. 확장했다는 표현은 없다.
- 기반 모델은 Z-Image(실사 흡수만 SD3.5-M). 방법의 뿌리로는 LLM의 on-policy distillation(GKD, MiniLLM 등)과 flow matching의 velocity field 관점을 인용한다.
- 공통점: 학생이 직접 rollout한 상태에서 teacher의 velocity를 맞추는 on-policy velocity matching. 뼈대가 가장 비슷하다.
- 차이점은 Q2의 표. DanceOPD는 "DiffusionOPD의 밀집 질의와 teacher 혼합이 문제"라는 진단 위에 서 있다. 개선판으로 읽을 수는 있지만 저자가 그렇게 선언하지는 않았다.
- 주의: DiffusionOPD는 코드가 공개되지 않아 DanceOPD 저자들이 직접 재현했다(부록 8.2). DanceOPD 논문 속 DiffusionOPD 모습은 그 재현판이다.

### Q2. DiffusionOPD와의 차이 표

*이 절을 두는 이유: 세 방법이 어디서 갈리는지 한눈에 보기 위해서다. (DiffusionOPD 원논문은 이후 직접 읽어 확인했고, 상세는 [[PAPER_DiffusionOPD.md]].)*

| 항목 | DiffusionOPD | DanceOPD | DiffSynth 스크립트 |
|---|---|---|---|
| 목적 | 멀티 보상 RL 통합 (GenEval·OCR·미학 등) | 여러 능력을 한 모델에 합치기/흡수 (T2I+편집, 실사, CFG) | CFG 4→1, 50→8스텝 증류 |
| Teacher | 과제별 RL로 학습한 전문가 (Stage 1) | 동결 전문가 (SFT 편집 모델 등) | 원본 모델 자신 (LoRA 스케일 0) |
| Student 초기화 | 사전학습 원본 | 가장 가까운 전문가 (효과 극단적) | 원본 + 빈 LoRA |
| 기반 모델 | SD3.5-M 512² | Z-Image (실사만 SD3.5-M) | Qwen-Image-2.1 |
| Teacher 선택 | 매 라운드 모든 과제 순회 | **샘플별 hard routing** | 해당 없음 (1개) |
| 한 번의 업데이트 | 과제 M개 손실을 누적해 1회 업데이트 | 라우팅된 과제 1개만 | 샘플 1개 |
| Rollout | SDE 또는 ODE | 16스텝 Euler ODE | 8스텝 (student 스케줄) |
| 질의 상태 수 | **궤적 전체** (K=N) | **1개**, 저노이즈 편향 Beta(5,2) | 1개, **균등 무작위** |
| 손실 | ODE 기본: transition-mean L2 (식 12), SDE: 닫힌 형태 KL (식 11) | 가중 없는 velocity MSE | teacher 다단계 x0 환산 velocity와의 MSE |
| PPO 사용 | 안 씀 (PPO와 기댓값 동일하지만 분산 높다고 논증) | 안 씀 | 안 씀 |
| 스텝 증류 의도 | **명시적으로 아님** | 아님 (rollout과 평가 스텝 달라도 됨) | **핵심 목적** |
| 코드 공개 | DanceOPD 논문에 따르면 당시 미공개 | GitHub 있음 | DiffSynth 내 있음 |

**읽을 때 중요한 점**
1. DiffusionOPD는 OPD를 flow matching에 처음 옮긴 쪽에 가깝다. LLM의 토큰별 reverse KL을 연속 상태 마르코프 체인으로 올리고, 같은 공분산 가우시안이라 KL이 평균 차이 제곱 하나로 떨어진다는 유도가 핵심이며, DanceOPD도 부록 7.1에서 사실상 재사용한다.
2. DanceOPD가 DiffusionOPD에서 바꾼 것은 세 가지: 모든 스텝 질의 → 저노이즈 1회 질의, 같은 스텝 과제 누적 → 샘플별 라우팅, 손실 가중 제거.
3. teacher가 서로 다르다. DiffusionOPD는 RL 전문가, DanceOPD는 SFT 편집 전문가. 비교 대상도 달라 **두 논문의 숫자는 직접 비교할 수 없다.**
4. "밀집 질의가 나쁘다"는 결과는 DiffusionOPD 원저자 설정이 아니라 DanceOPD 저자가 자기 설정(Z-Image, 편집)에서 재현한 DiffusionOPD를 비교한 것이다. baseline 자체 재현 문제가 그대로 해당된다.
5. 스크립트는 DiffusionOPD와 목적이 반대다. DiffusionOPD는 "step distillation에는 쓰지 않는다"고 선언했고 스크립트는 바로 그 용도다.

### Q3. 문서에 DanceOPD가 참조돼 있는데 "구현이 없다"고? (정정 이력)
*이 절을 두는 이유: 어떤 판단이 어떤 근거로 바뀌었는지 남겨 같은 실수를 막는다.*

- 처음에 "스크립트는 DiffusionOPD도 DanceOPD도 아닌 독자 구현으로 보는 편이 안전하다"고 했던 것은 **틀렸다.** DiffSynth 문서가 두 논문을 인용하고, 같은 OPD 코드를 파라미터(`opd_num_distill_steps` / `opd_min_distill_idx`)로 두 스타일에 대응시킨다(§5.1).
- 다만 "논문 방법과 일치하지 않는다"는 판단은 유지된다: 옵션이 94행 때문에 꺼져 있고, 라우팅·다중 teacher·Beta 분포는 없다(§5.2~5.3).

### Q4. 코드 확인해봐 / 구현은 되어 있다는 소리야?
- 다시 받아 확인했다(파일 동일). **구현하려는 코드는 있지만 지금은 작동하지 않는다.** 91~92행의 선택 로직과 옵션 정의는 있으나 94행이 덮어써서 옵션 값이 결과에 영향을 주지 않는다.
- 스크립트 기본 설정은 DanceOPD도 DiffusionOPD도 아닌 "균등 무작위 단일 질의" 증류.
- 세 모델(Qwen-Image-2.1, Flux2, Moss-Image)의 `train_opd_lora.py`에 같은 줄이 있다. 상세는 §5.2~5.3.

### Q5. sh에서 수정하면 되잖아?
**아니요.** sh에 `--opd_min_distill_idx 4`나 `--opd_num_distill_steps 2`를 추가해도 인자는 정상 전달되지만 94행 `randint`가 덮어써 무효다. 옵션은 sh에서 설정하는데 코드가 읽어 쓰지 않는 상태. DanceOPD 스타일을 쓰려면 **py의 94행 삭제가 필요**하고, 삭제 후에 sh의 `--opd_min_distill_idx`가 동작한다. (§5.2)

### Q6. DMD2 / DiffusionOPD / DanceOPD / DiffSynth 스크립트 4자 비교
*이 절을 두는 이유: 같은 "증류"라도 DMD2(분포 매칭)와 OPD 계열(점별 회귀)은 맞추는 대상이 달라, 상황별 선택 기준을 보기 위해서다. DMD2 열은 기존 지식(메모리 요약 포함) 기준이며 원문을 다시 열어 확인하지 않았다.*

| 항목 | DMD2 | DiffusionOPD | DanceOPD | DiffSynth 스크립트 |
|---|---|---|---|---|
| 목적 | 다스텝 모델을 1~4스텝으로 압축 | 과제별 RL teacher들을 학생 하나로 통합 | 여러 능력 합치기/흡수(T2I+편집, 실사, CFG) | 50스텝·CFG 4를 8스텝·CFG 1로 압축 |
| 맞추는 것 | 분포(학생 출력 분포와 teacher 분포) | 한 스텝 전이 평균(점별 회귀) | 한 지점의 velocity(점별 회귀) | teacher x0 추정을 velocity로 환산한 점별 회귀 |
| teacher | 고정 사전학습 확산 모델(CFG 포함) | 과제별 RL 전문가 여러 개 | 동결 전문가 여러 개(SFT 편집 모델 등) | LoRA를 끈 원본(학습 안 됨, 1개) |
| teacher 선택 | 해당 없음 | 매 라운드 모든 과제를 순회, 손실 누적 | **샘플별 hard routing** | 해당 없음 |
| 손실 | (가짜 score − 진짜 score) 방향 gradient + GAN | 전이 평균 제곱거리(ODE 기본 L2, SDE는 닫힌 형태 KL) | 가중 없는 velocity MSE | velocity MSE 한 개 |
| 보조 네트워크 | fake score 네트워크 + 판별기 | 없음 | 없음 | 없음 |
| 실제 데이터 | 있음(GAN이 실제 이미지를 봄) | 없음(프롬프트+보상) | 편집 데이터 일부 사용(라우팅된 버킷의 조건) | 없음(프롬프트만) |
| L_reg | 제거 | 해당 없음 | 해당 없음 | 해당 없음 |
| 학습 안정화 | TTUR(critic 5회 : generator 1회), GAN | 라운드로빈 누적 | 없음 | 없음 |
| 다스텝 처리 | backward simulation으로 학습·추론 불일치 해소 | 롤아웃 전 스텝 손실 합산 (K=N) | 롤아웃 후 **저노이즈 1지점**(K=1, Beta(5,2)) | 롤아웃 후 1스텝 균등 무작위 (94행 버그) |
| 롤아웃 | 학생의 다스텝 출력 | SDE 또는 ODE | 16스텝 Euler ODE | 8스텝 student 스케줄 |
| on-policy | 있음 | 있음 | 있음 | 있음 |
| teacher 호출 | 학생 샘플에 노이즈를 입혀 score 1회 | 학생 상태에서 1스텝 | 학생 상태에서 1회 | 서브스텝 1~9회(평균 약 6) × CFG 2배 |
| gradient 경로 | score 차이를 생성기 출력으로 역전파 | 평균 차이를 학생 예측으로 역전파 | velocity 차이를 학생 예측으로 역전파 | 속도 MSE를 학생 1스텝으로 역전파 |
| 모드 거동 | 분포 매칭이라 mode-seeking(다양성 감소 위험) | teacher 궤적 추종 | teacher 궤적 추종 | teacher 궤적 추종(결정론적) |
| 품질 천장 | GAN 덕에 teacher 초과 가능 | RL baseline 대비 우위 주장 | 개별 teacher 초과 주장(자체 재현 baseline 기준) | 원본 teacher 수준이 상한 |
| 학습 비용 | 학생 + fake score + 판별기(부담 큼) | 단일 모델, 전 스텝 gradient | 단일 모델, gradient는 1지점 | 단일 모델(LoRA 토글로 공유) |
| 기반/규모 | ImageNet, SDXL | SD3.5-M 512² | Z-Image(실사만 SD3.5-M) | Qwen-Image-2.1 |
| 코드/이슈 | 공개 | 공개 안 됨(DanceOPD 논문 기준) | GitHub 공개 | 94행이 옵션을 무효화, checkpoint 충돌 의심 |

**표에서 고친 곳(외부에서 받은 초안 대비)**
1. DiffusionOPD "품질 천장: teacher 수준이 상한"은 상한이 아니다. 원논문은 멀티태스크 RL baseline을 웃돈다고 주장한다. 그 비교가 학생 vs teacher인지는 읽은 범위에서 확인 못 했다.
2. DiffusionOPD "하이퍼파라미터: 부록 없음(미기재)"은 단정할 수 없어서 행을 제외했다(논문 PDF에는 부록이 없다는 [[PAPER_DiffusionOPD.md]]의 기술과 일치하나, 이 4자 표에서는 근거 부족).
3. 스크립트 teacher 호출 "약 6회"는 구간마다 다르다(첫 구간 약 8~9회, 마지막 구간 1회, 평균 약 6).

**표에서 짚을 점**
- 세 OPD 계열의 공통점은 점별 회귀이고 on-policy라는 것. DMD2는 분포 매칭이라 구조가 다르다.
- 스크립트만 teacher 타깃을 다단계로 만든다. 8스텝은 스텝이 커서 즉석 velocity 일치로는 부족하기 때문에 teacher가 구간을 세밀하게 내려간 x0를 쓰는 것이다. 논문들에는 없는 설계.
- DMD2만 GAN으로 teacher를 넘길 수 있다는 점이 구조적 차이. 스크립트 방식은 원본 teacher보다 좋아질 경로가 없다.

---

### Q7. DanceOPD 공개 저장소와 DiffSynth `train_Qwen-Image-2.1-8steps_lora.sh`의 차이는?
*이 절을 두는 이유: 이름이 비슷한 두 코드가 실제로는 푸는 문제가 달라, 어느 쪽을 어떤 목적에 쓸지 가려야 하기 때문이다.*

두 코드는 이름이 비슷하지만 **푸는 문제가 다르다.** DanceOPD는 여러 능력을 한 모델에 합치기, DiffSynth 스크립트는 스텝·CFG 압축이다. (저장소 분석은 §5-2)

| 항목 | DanceOPD 저장소 | DiffSynth 스크립트 |
|---|---|---|
| 목적 | 여러 능력(T2I, 편집, 실사)을 학생 하나에 합치기. CFG 흡수는 옵션 | 50스텝·CFG 4를 8스텝·CFG 1로 압축 |
| 지원 모델 | SD3.5-M, Z-Image, toy (Qwen-Image-2.1 없음) | Qwen-Image-2.1 (FLUX.2, MOSS용 별도 스크립트) |
| teacher | 설정에 정의한 여러 개. 각각 별도 DiT 복사본(LoRA 병합 또는 전체 체크포인트) | 1개. 같은 DiT에서 LoRA 스케일 0으로 둔 원본 |
| 라우팅 | 있음. 최적화 스텝마다 route 하나, 해당 데이터 버킷에서만 샘플링 | 없음 |
| student rollout | 16스텝 Euler ODE. 배포 스텝 수와 무관 | **배포 스케줄(8스텝)**과 동일 |
| 질의 위치 | Beta(5,2)로 후반부 편향, 의도대로 동작 | 전 스텝 균등 무작위. `min_distill_idx`는 94행 덮어쓰기로 무효 |
| rollout 가속 | query-prefix: 질의 인덱스까지만 굴리고 중단(평균 약 12/16) | 항상 전체 8스텝 |
| teacher 타깃 | 같은 상태에서 teacher **속도 1회 평가**(CFG 시 2회) | **teacher가 구간을 서브스텝(평균 약 6회, CFG 2배)으로 내려간 뒤 x0 추정을 속도로 환산** |
| 손실 | 속도 MSE | 속도 MSE (환산된 타깃) |
| CFG 처리 | `teacher_cfg_scale`로 teacher 속도에 CFG 합성, 학생 1.0. 기본값 둘 다 1.0 | teacher 4 / 학생 1 고정 |
| 스케줄 | 스케줄러에서 읽음 | `compute_sigmas.py` 출력을 `.sh`에 손으로 붙여 넣음, 해상도 고정 |
| 프롬프트 인코딩 | 샘플당 1회, rollout 밖 | `unit_runner`를 호출마다 재실행 |
| LoRA 대상 | 어텐션 + FFN(w1/w2/w3), rank 128 | 어텐션 + img_mlp + **timestep embedder + modulation + norm_out**, rank 128 |
| 학습률/규모 | lr 2e-4, grad_accum 4, 3000스텝 | lr 2e-5, 5 에폭 × dataset_repeat 40 |
| student 초기화 | LoRA 웜스타트 옵션(논문 설정은 비공개) | 없음(빈 LoRA = 원본) |
| 데이터 | prompt + task + source/target 이미지 CSV | 예제 데이터셋. 정답 이미지는 손실에 안 쓰임 |
| 지원 방법 | danceopd, diffusionopd, flowopd, offpolicy | OPD 한 종류 |
| 체크포인트 재개 | 있음 | 이 스크립트에는 없음 |

**차이가 학습 결과에 미치는 영향**
1. **teacher 타깃 설계가 가장 근본적인 차이.** DanceOPD는 "지금 이 지점의 teacher 속도"를 따라 하며 학생이 16스텝으로 가는 작은 Δt를 전제한다. DiffSynth는 8스텝(큰 Δt)이라 순간 속도 일치로는 부족해서 teacher가 구간을 정제한 x0를 목표로 삼는다. 따라서 DanceOPD 코드에서 `rollout_steps: 8`, `teacher_cfg_scale: 4`만 바꿔도 스텝 증류가 되는 게 아니라 **큰 Δt에서 순간 속도를 맞추는 다른 방법**이 된다.
2. **rollout 스케줄의 의미.** DiffSynth는 학습 때 방문 상태가 배포 때 방문 상태와 같다. DanceOPD는 평가 스텝 수와 다른 16스텝 rollout을 논문이 의도적으로 허용한다(Fig. 4c).
3. **질의 분포가 정반대.** DanceOPD는 후반 편향, DiffSynth는 균등(옵션이 무효여서). 스텝 증류에서는 고노이즈 스텝이 레이아웃을 결정하므로 균등이 반드시 나쁘지는 않다. 두 코드의 결과를 서로의 근거로 쓸 수 없다.
4. **구조적 버그 유무.** DanceOPD는 teacher가 별도 모듈이라 LoRA 스케일 전환이 없다. DiffSynth에서 의심된 **checkpointing + 스케일 0 재계산 불일치는 DanceOPD 쪽에서는 생기지 않는 구조**다(대가로 teacher 복사본만큼 GPU 메모리를 더 쓴다). DanceOPD의 선택 로직은 의도대로 동작하고 DiffSynth의 `min_distill_idx`는 무효다.
5. **LoRA 범위.** DiffSynth는 시간 조건 모듈까지 열어 둔다. 스텝 스케줄이 바뀌는 증류에서는 합리적이다. DanceOPD는 어텐션과 FFN만.
6. **학습률 10배 차이**(2e-4 vs 2e-5). DiffSynth 값은 예제 데이터용으로 보이고 실전 조정이 필요할 가능성이 크다.

**정리**: DanceOPD 저장소는 다능력 통합 연구 코드로, 구조가 깔끔하고 논문 알고리즘에 충실하지만 Qwen-Image-2.1은 지원하지 않고 스텝 증류용도 아니다. DiffSynth 스크립트는 문서가 DanceOPD를 인용하지만 실제로는 CFG·스텝 압축 레시피이고 teacher 타깃 설계는 목적에 더 맞지만 옵션 무효, 스케일 전환, 스케줄 하드코딩이 있다. Qwen-Image-2.1을 8스텝으로 줄이려면 DiffSynth 스크립트를 고쳐 쓰는 편이 맞다.

### Q8. DanceOPD는 step을 줄이기 위한 목적이 아냐?
**아니다.** 논문 부록: DanceOPD는 "trajectory-compression distillation"이 아니라 지점별 velocity field 매칭을 하므로 학습 스텝 격자와 평가 스텝 격자가 같을 필요가 없다. 실제로 학습 rollout 16스텝, 평가 28스텝(GEditBench CFG 7.0, GenEval 3.5)이다. 헷갈리는 이유는 §0-1에 정리(같은 OPD 계열, CFG 흡수 실험, Z-Image-Turbo 공개 설정).

### Q9. DiffSynth 스크립트에는 왜 이 알고리즘을 적용했나? (방향성이 다른데)
*이 절을 두는 이유: 목적이 다른 두 논문의 알고리즘이 왜 스텝 증류 스크립트에 쓰였는지, 그 근거가 어디까지인지 가르기 위해서다.*

DiffSynth 개발진의 속마음은 알 수 없고, 문서(Training/Distillation)에 적힌 근거만 정리한다.

문서는 스텝 증류 방법을 네 기준(범용성, 안정성, 단순함, 품질)으로 비교한다.

| 방법 | 범용성 | 안정성 | 단순함 | 품질 |
|---|---|---|---|---|
| Direct Distill | ★★★ | ★★★ | ★★★ | ★ |
| DMD2 | ★ | ★ | ★ | ★★★ |
| **OPD (권장)** | ★★★ | ★★ | ★★★ | ★★★ |

OPD를 권장하는 이유 두 가지:
- Direct Distill은 teacher가 미리 만든 궤적으로 학습하는데, OPD는 학생 자신의 궤적에서 시작하므로 결과가 더 좋다.
- DMD2는 모델마다 판별기를 만들어야 하는데, OPD는 필요 없어 범용적이고 튜닝이 적다.

즉 두 논문의 목적(능력 통합)이 아니라 **"학생이 자기 궤적 위에서 teacher를 따라 한다"는 뼈대**만 가져와서 스텝 증류에 썼다.

**DiffSynth가 직접 보탠 것**
- **teacher 타깃을 "다단계 출력"으로 바꿨다.** 8스텝은 Δt가 커서 두 논문의 즉석 속도 일치로는 안 되기 때문. 문서도 알고리즘을 "teacher의 multi-step output과의 MSE"라고 적는다.
- **DanceOPD 변형의 이유를 다르게 설명한다.** 문서는 저노이즈 구간 샘플링이 "학습 후 출력이 크게 벗어나는 것을 막는다"고 하나, 논문의 이유는 능력 특화 정보가 저노이즈에 모여 있다는 것이다. 문서 설명의 실험 근거는 읽은 범위에서 찾지 못했다.

**한계도 문서가 인정한다**: "OPD is not fully stable"(오래 학습한다고 좋아지지 않음), 50 → 4처럼 격차가 크면 결과가 나쁘니 50 → 8 → 4 점진 증류를 권장, 정량 평가는 문서에서 보이지 않음(별점 표뿐).

**현실**: 지금 코드에서는 이 근거마저 실현되지 않는다. 스크립트가 두 변형 옵션을 쓰지 않고, 쓰더라도 94행 덮어쓰기로 무효다.

**정리**: 적용 이유는 "범용적이고 단순한 증류 레시피가 필요했는데 OPD의 구조가 맞아 보여서"다. 두 논문이 스텝 증류를 검증한 것은 아니므로 이 레시피의 효과는 DiffSynth가 직접 입증해야 하는 별개 주장이다. 스크립트를 쓸 때는 두 논문의 결과를 근거로 삼지 말고 실제 8스텝 출력의 품질을 직접 확인한다.

### Q10. OPD는 few-step 증류에 적합하지 않다는 말인가? (정정)
**아니다.** 앞서 Q9 초안의 "논문들은 오히려 반대로 말합니다"는 과한 표현이었다.
- **DiffusionOPD**: "스텝 축소 증류에는 쓰지 않는다. 우리는 다과제 통합을 연구한다"는 **연구 범위 선언**이지 부적합 평가가 아니다.
- **DanceOPD**: "trajectory-compression distillation이 아니므로 스텝 격자가 같을 필요 없다"는 **자기 방법의 성격 설명**이다.
- 두 논문 모두 스텝 증류를 시험하지 않았을 뿐이다. "검증되지 않음"과 "안 된다"는 다르다.
- OPD 발상(학생 상태에서 supervise)은 few-step에서 특히 의미가 있다. 학생의 큰 스텝이 만드는 오차가 쌓이는 곳이 그 학생 궤적이기 때문이다. DMD2의 backward simulation이 같은 문제를 푸는 같은 계열의 해법이다. DanceOPD 논문이 참고문헌으로 단 D-OPSD(2605.05204)는 제목이 "On-policy self-distillation for continuously tuning step-distilled diffusion models"로 스텝 증류 모델에 OPD를 적용하는 연구다(제목만 확인, 내용은 읽지 않음).
- 정확한 표현: **"적합할 수 있으나 이 스크립트의 결과는 직접 확인해야 한다."** 구현 쪽 문제(옵션 무효, 스케일 전환)는 알고리즘 적합성과 별개다.

### Q11. 서로 모순처럼 들리는데? — 세 층위로 정리
헷갈린 이유는 **서로 다른 층위를 "OPD"라는 한 단어로 섞어 말했기** 때문이다.

| 층 | 무엇인가 | 목적 |
|---|---|---|
| ① 뼈대 (OPD 일반) | 학생이 직접 생성하고, 그 상태에서 teacher를 불러 맞추는 방식 | 목적에 중립적. 어디에 쓸지는 정하지 않음 |
| ② 두 논문 (DiffusionOPD, DanceOPD) | ① + 능력 통합용 선택들 (여러 teacher, 라우팅, 질의 지점 선택 등) | **여러 능력을 한 모델에 합치기** |
| ③ DiffSynth 스크립트 | ① + 스텝 압축용 선택들 (teacher 서브스텝 x0 타깃, 8스텝 학생 스케줄) | **50스텝을 8스텝으로 줄이기** |

- "이 알고리즘은 다른 목적이 있다" → **②를 가리킨 말**. 두 논문이 풀려던 문제는 능력 통합이다.
- "few-step에 맞을 수 있다" → **①을 가리킨 말**. 뼈대의 발상은 스텝 증류에도 쓸 만한 합리적 가설이다.
- ③은 ②의 방법을 그대로 쓴 것이 아니라 ①의 뼈대만 빌리고 스텝 증류에 필요한 부분을 DiffSynth가 새로 설계한 것이다. 그 설계의 효과는 두 논문이 보증하지 않는다.

**한 문장**: 두 논문은 능력 통합용이고, DiffSynth 스크립트는 그 뼈대를 스텝 증류에 전용한 별개의 레시피이며, 그 효과는 두 논문이 아닌 스크립트 자체로 검증돼야 한다.

**요리 학원 비유**: OPD의 기본 방식은 "제자가 직접 요리하다가 중간중간 스승에게 '지금 이 상태에서 어떻게 해요?'라고 묻고 그대로 따라 하기"라는 목적 중립적인 연습 방법이다.
- 두 논문: 스승이 여러 분(한식·양식·디저트)이고, 제자 한 명이 세 분야를 다 하게 만드는 것이 목표.
- DiffSynth 스크립트: 스승은 한 분(예전의 제자 자신)이고, 50단계로 정성껏 만드는 요리를 8단계로 비슷한 맛을 내게 하는 것이 목표.
- 공통점은 "직접 해 보며 스승에게 묻는" 연습 방식뿐.

### Q12. 여러 배우기 수업이면 선생이 여러 명이란 소리야?
**그렇다. 두 논문 모두 선생(teacher)이 여러 명이다.**

- **DanceOPD**: 선생이 능력별로 따로 있다(T2I 선생, 편집 선생, 스타일 선생 등). 샘플 하나마다 **선생을 한 명만 지정**한다("이 문제는 편집 선생님께 물어봐"). 여러 선생의 답을 평균 내지 않는데, 평균하면 누구의 답도 아니게 되기 때문이다. 학습 전체로 보면 선생 여러 명이 돌아가며 가르치지만 한 문제는 한 선생이 담당한다.
- **DiffusionOPD**: 선생은 과제별로 따로 RL로 키운 전문가(GenEval, OCR, 미학). 학습 한 바퀴에 과제를 모두 한 번씩 돌고 세 선생의 손실을 모아 한 번에 업데이트한다.
- **DiffSynth 스크립트**: 선생은 **자기 자신의 원본 한 명**뿐(LoRA를 끈 상태). 그래서 선생을 고르는 기능(라우팅)이 아예 없다.

| | 선생 수 | 선생 고르기 |
|---|---|---|
| DanceOPD | 여러 명 | 샘플마다 한 명 지정 |
| DiffusionOPD | 여러 명 | 매 라운드 전부 순회 |
| DiffSynth 스크립트 | 1명 | 없음 |

이 점이 두 논문과 스크립트의 가장 큰 구조적 차이다.

---

## 9. 한 줄 요약 (전체)

DanceOPD는 "여러 능력을 속도장으로 보고, 샘플마다 선생 하나를 골라(hard routing) 학생이 직접 걸은 저노이즈 상태 한 곳에서 속도를 MSE로 맞춘다"는 단순한 설계로 DiffusionOPD의 밀집 질의·teacher 혼합을 대체했지만, 자체 재현 baseline과 초기화 의존·통계 부재 때문에 숫자는 보수적으로 읽어야 하며, DiffSynth의 DanceOPD 변형 옵션은 94행 덮어쓰기 때문에 지금 꺼져 있다.

## 10. 관련 메모리 링크

- [[paper_diffusionopd]] — 선행 연구 + 같은 스크립트의 x0 점프형 목표·checkpoint 문제 분석
- [[paper_dmd2]], [[paper_dmd]] — 분포 매칭 증류(Q6 비교 대상)
- [[paper_z_image]] — DanceOPD의 주력 기반 모델
- [[paper_qwen_image_2_1]] — 스크립트의 대상 모델
- [[paper_qwen_image_2_rl]] — 작업별 교사 OPD를 쓴 또 다른 사례
