# CSA-Net — attention 의 조건을 "낮춰서" 검색을 되찾은 논문

## 📌 메타 정보

| 항목 | 내용 |
|---|---|
| **논문 제목** | Fashion Outfit Complementary Item Retrieval |
| **저자** | Yen-Liang Lin, Son Tran, Larry S. Davis |
| **소속** | **Amazon** |
| **공개일** | 2020-06 (CVPR 2020 본회의) |
| **학회** | **CVPR 2020** (pp. 3311–3317) |
| **분야** | Fashion Compatibility(패션 호환성), Complementary Item Retrieval(보완 아이템 검색), Metric Learning(거리 학습) |
| **논문 PDF** | https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Fashion_Outfit_Complementary_Item_Retrieval_CVPR_2020_paper.pdf |
| **약칭** | **CSA-Net** (Category-based Subspace Attention Network) |
| **코드** | ❌ **미공개** (Amazon 사내 논문) |
| **비공식 구현** | [Jungjaewon/Fashion_Outfit...](https://github.com/Jungjaewon/Fashion_Outfit_Complementary_Item_Retrieval) (`a447b66`), [bigohofone/csa-net](https://github.com/bigohofone/csa-net) (`07db8e4`) — **둘 다 실행 불가·논문 미재현**, 상세는 Q3 |
| **데이터셋** | Polyvore Outfits [16] (Vasileva et al., ECCV 2018) — disjoint / non-disjoint 두 세트 |
| **베이스 모델** | ResNet18 (ImageNet 사전학습), embedding size 64 |
| **ANN 라이브러리** | hnswlib (https://github.com/nmslib/hnswlib) |
| **직접 계승한 방법론** | CSN [17] (Conditional Similarity Networks) → Type-aware [16] → SCE-Net [15] |

---

## 📖 주요 용어 사전 (Glossary)

*이 논문은 새 용어를 거의 만들지 않습니다. 패션 검색 분야의 기존 용어 몇 개만 알면 본문이 다 읽힙니다.*

### 문제 정의 관련

| 용어 | 풀이 |
|---|---|
| **outfit(코디, 착장)** | 상의·하의·신발·가방 등 여러 아이템이 하나로 묶인 세트. 이 논문의 기본 단위. |
| **complementary item(보완 아이템)** | 코디를 완성시켜 주는 **다른 카테고리**의 아이템. 상의+하의가 있을 때의 "신발" 같은 것. |
| **compatibility(호환성, 어울림)** | 아이템끼리 스타일이 맞는 정도. **similarity(닮음)와 정반대 개념**이라는 점이 핵심 — 보완 아이템은 일부러 시각적으로 안 닮은 걸 찾아야 한다. |
| **compatibility prediction(호환성 예측)** | "이 코디 조합이 어울리나?"를 점수로 **채점**하는 작업. 기존 연구 대부분이 여기에 머물렀다. |
| **complementary item retrieval(보완 아이템 검색)** | 채점이 아니라 카탈로그 수백만 개 중에서 **찾아오는** 작업. 이 논문이 새로 정의한 과제. |
| **FITB (Fill-In-The-Blank, 빈칸 채우기)** | 코디에서 아이템 하나를 지우고, **4지선다**(정답 1 + 오답 3) 중에서 맞는 걸 고르는 평가 방식. Han et al. [5]가 도입. |

### 방법론 관련

| 용어 | 풀이 |
|---|---|
| **subspace(부분공간)** | 특징 벡터(feature vector) 전체를 쓰지 않고, 일부 차원만 골라 쓰는 "작은 공간". 색으로 볼 때 / 소재로 볼 때 / 격식으로 볼 때처럼 **여러 관점(aspect)의 어울림**을 따로 재기 위해 쓴다. |
| **mask(마스크)** | 특징 벡터와 같은 길이의 학습 가능한 가중치 벡터. 원소별 곱(Hadamard product)으로 곱해서 subspace 를 만들어낸다. |
| **Hadamard product(원소별 곱, ⊙)** | 같은 길이 벡터 두 개를 자리마다 곱하는 연산. `[1,2,3] ⊙ [0,1,1] = [0,2,3]` — 즉 마스크가 0인 자리는 꺼진다. |
| **conditional similarity(조건부 유사도)** | "무엇을 기준으로 닮았다고 볼지"를 **조건에 따라 바꾸는** 거리. CSN [17]이 원조. |
| **attention weight(주의 가중치)** | 여러 subspace 중 어느 것을 얼마나 쓸지 정하는 소수 5개(합이 1). 이 논문에서는 **오직 카테고리 쌍에만** 의존한다. |
| **triplet loss(삼중항 손실)** | anchor(기준)·positive(정답)·negative(오답) 세 장을 놓고 "정답은 가깝게, 오답은 margin 이상 멀게" 학습시키는 고전적 손실 함수. |
| **outfit ranking loss(코디 랭킹 손실)** | 이 논문이 제안. anchor 한 장이 아니라 **코디 전체**와의 평균 거리를 쓴다. |
| **hard negative mining(어려운 오답 채굴)** | 오답 중에서도 모델이 헷갈려 하는(=거리가 가까운) 것만 골라 학습에 쓰는 기법. 학습 신호가 강해진다. |
| **semi-hard negative(준-어려운 오답)** | 정답보다는 멀지만 margin 안에는 들어와 있는 오답. 너무 어려운 오답만 쓰면 학습이 붕괴하므로 쓰는 절충안. |

### 인덱싱·평가 관련

| 용어 | 풀이 |
|---|---|
| **indexing(색인)** | DB 아이템들의 임베딩(embedding)을 **미리 계산해서** 검색 자료구조에 저장해 두는 것. 이게 가능해야 실서비스가 된다. |
| **KNN search (k-Nearest Neighbor, k-최근접 이웃 탐색)** | 질의 벡터와 가장 가까운 k개를 인덱스에서 찾아오는 것. |
| **ANN (Approximate Nearest Neighbor, 근사 최근접 이웃)** | 정확도를 조금 포기하고 KNN 을 훨씬 빠르게 하는 방식. hnswlib 가 대표. |
| **recall@top k** | 상위 k개 안에 정답이 들어 있었던 비율. 이 논문의 검색 지표. |
| **AUC (Area Under the ROC Curve)** | 호환성 예측의 표준 지표. 1에 가까울수록 좋음. |
| **disjoint / non-disjoint set** | Polyvore Outfits 의 두 분할. **disjoint** = 학습과 테스트가 아이템을 **하나도 공유하지 않음**(어려움, 작음). **non-disjoint** = 완전한 코디는 안 겹치지만 개별 아이템은 겹칠 수 있음(쉬움, 큼). |
| **distractor(방해꾼)** | 검색 평가에서 정답이 아닌 후보로 섞어 넣는 이미지들. |

---

## 🎯 논문 요약 (TL;DR)

**한 줄**: 기존 outfit 호환성 모델들은 "이 조합이 어울리나?"를 **채점**만 할 수 있고 **검색**은 못 한다. 이 논문은 attention 의 조건을 *이미지 쌍*에서 *카테고리 쌍*으로 낮춰서, 아이템 임베딩을 미리 계산해 KNN 인덱스에 넣을 수 있게 만든다. 성능은 덤으로 올랐다.

| 구분 | 내용 |
|---|---|
| **핵심 문제** | 호환성 예측 SOTA 모델(SCE-Net)은 subspace 를 고르는 데 **이미지 쌍**이 필요하다 → DB 아이템의 임베딩을 미리 만들 수 없다 → 질의마다 카탈로그 전수 비교 → 실서비스 불가 |
| **해결책 (1)** | **CSA-Net** — attention 을 이미지가 아니라 **카테고리 one-hot 두 개**에만 조건부여. 이미지 한 장 + 카테고리 두 개면 임베딩이 나오므로 인덱싱 가능 |
| **해결책 (2)** | **Outfit ranking loss** — anchor 한 장이 아니라 코디 전체와의 평균 거리로 랭킹 학습 |
| **검증** | Polyvore Outfits 에서 FITB 63.73 / AUC 0.91 (텍스트 미사용인데도 텍스트 쓰는 baseline 전부 격파), 검색 recall@10 은 SCE-Net 대비 상대 +62% |
| **가장 약한 곳** | Table 2·3 을 겹쳐 보면 **기여 (2)의 이득 전부가 hard negative mining 에서 나온다** (7장 참조) |

---

## ⭐ 핵심 기여 (Contributions)

논문이 스스로 밝힌 기여는 둘입니다.

1. **Category-based attention selection mechanism** — attention 을 **아이템 카테고리에만** 의존하게 만들어, 대규모 색인(indexing)과 탐색이 가능해짐
2. **Outfit ranking loss** — 아이템 쌍이 아니라 **코디 전체**에 작용하는 새 랭킹 손실. 호환성 예측과 검색 정확도를 함께 개선

그리고 논문이 명시하진 않았지만 실질적 기여가 하나 더 있습니다.

3. **검색 벤치마크 신설** — Polyvore Outfits 위에 recall@top k 로 재는 검색 평가 프로토콜을 처음 정의 (단, 공개 여부 언급 없음)

---

## 1️⃣ 문제 정의 — 왜 기존 방법은 검색에 못 쓰이나

*이 절을 두는 이유: 이 논문의 모든 설계 판단이 "인덱싱이 되느냐"라는 단 하나의 제약에서 나오기 때문에, 그 제약을 먼저 이해해야 나머지가 자연스럽게 읽힙니다.*

### 작업 정의

상의·하의·가방이 있는 미완성 코디에 어울리는 **신발 하나를 카탈로그에서 찾아라.**

![CSA-Net Figure 1 — 보완 아이템 검색 예시](figures/csa_net_fig1.png)
*Figure 1. 위: 신발이 빠진 코디. 아래: 제안 방법이 찾아온 상위 결과들.*

일반 이미지 검색과 결정적으로 다른 점이 있습니다. 찾으려는 아이템은 질의 아이템들과 **일부러 다른 카테고리**입니다. 즉 **시각적으로 안 닮은 걸 찾아야** 합니다. 기준이 similarity(닮음)가 아니라 **compatibility(어울림)** 입니다.

### 기존 방법이 검색에 못 쓰이는 이유

| 기존 방법 | 검색 불가 사유 |
|---|---|
| **SCE-Net [15]** | subspace 선택에 **이미지 쌍**이 필요 → DB 아이템 임베딩을 미리 못 만든다. 질의마다 N개 전수 비교 |
| **GCN [2]** (Cucurull et al.) | 분류(classification) 모델 + 테스트 시점 그래프 연결을 활용 → **신상품은 연결선이 없어** 임베딩 자체가 정의 안 됨 |
| **Type-aware [16]** | 카테고리 쌍마다 독립 subspace 66개 → 인덱싱은 가능하지만 파라미터가 흩어져 학습 비효율 |

> ⚠️ 이 표는 나중에 7장의 급소 (3)과 이어집니다 — 실제로 "인덱싱 불가"가 엄밀히 성립하는 건 SCE-Net 과 GCN 뿐입니다.

---

## 2️⃣ 방법 (1) — CSA-Net

*이 절을 두는 이유: 인덱싱을 가능하게 만든 구조적 장치가 여기이고, 논문이 "attention"이라 부르는 것의 실체를 정확히 파악해야 성능 향상의 원인을 오해하지 않습니다.*

### 2.1 구조

![CSA-Net Figure 2 — 전체 구조](figures/csa_net_fig2.png)
*Figure 2. 소스 이미지 + 소스 카테고리 + 타깃 카테고리 → CNN 특징에 마스크 k개를 곱해 subspace 임베딩을 만들고, 카테고리 두 개로 예측한 attention 가중치로 섞어 최종 임베딩 f 를 만든다.*

입력이 셋입니다: **소스 이미지 + 소스 카테고리 + 타깃 카테고리.**

네트워크는 다음 함수를 학습합니다.

```
f = ψ(I_s, c_s, c_t)
```
*(즉 이미지 한 장과 카테고리 one-hot 벡터 두 개를 넣으면 임베딩 한 개가 나온다)*

단계별로:

1. 이미지를 CNN(ResNet18)에 통과 → 특징 벡터 **x**
2. 학습 가능한 마스크(mask) 5개 **m₁ ~ m₅** 를 x 에 원소별 곱 → subspace 임베딩 5개
3. 두 카테고리 one-hot 을 이어붙여(concatenation) 작은 MLP(fc → relu → fc → softmax)에 넣어 가중치 **w₁ ~ w₅** 산출
4. 최종 임베딩 = 각 subspace 임베딩에 w 를 곱해 더한 값

$$\mathbf{f} = \sum_{i=1}^{k} (\mathbf{x} \odot \mathbf{m}_i) * w_i \qquad (1)$$

*(k = subspace 개수 = 5, ⊙ 는 원소별 곱, wᵢ 는 스칼라)*

### 2.2 ⭐ 여기서 반드시 짚어야 할 점 — 이건 사실 attention 이 아니다

*이 소절을 두는 이유: 논문의 포장(attention)과 실제 수학이 다르고, 그 차이를 알아야 "왜 성능이 올랐는가"에 대한 답이 바뀌기 때문입니다.*

wᵢ 는 **스칼라**이므로 식 (1)을 정리하면 이렇게 됩니다.

$$\mathbf{f} = \mathbf{x} \odot \left( w_1 \mathbf{m}_1 + \cdots + w_5 \mathbf{m}_5 \right)$$

즉 subspace 5개를 섞는 게 아니라 **마스크 5개를 먼저 섞어서 유효 마스크(effective mask) 한 장을 만드는 것**과 완전히 동일합니다.

그리고 w 는 오직 카테고리 쌍에만 의존합니다. 카테고리 11개 기준 조합은 최대 121개(순서 무관이면 66개)뿐입니다.

> **결론: 이 "attention"은 입력 이미지에 전혀 반응하지 않는, 66칸짜리 학습된 룩업 테이블(lookup table)입니다.**

그러면 Type-aware [16]의 "66개 독립 마스크"와 뭐가 다른가? 딱 하나입니다.

| | 마스크 개수 | 제약 |
|---|---|---|
| Type-aware [16] | 66개 | 서로 **독립** (자유도 66 × 특징차원) |
| **CSA-Net** | 66개 (유효 마스크) | **5개 기저(basis) 마스크의 convex 조합으로 제한** (자유도 5 × 특징차원 + 66 × 5) |

표현력(expressive power)은 오히려 **줄었는데** 성능은 올랐습니다. 그러므로 이득의 정체는 새로 생긴 능력이 아니라 **파라미터 공유에 의한 regularization(정규화) 효과**입니다. 논문이 "disjoint set 성능이 전부 낮은 건 학습 데이터가 적어서"라고 스스로 설명한 것과 정확히 맞아떨어집니다.

논문은 이걸 "attention mechanism" 이라 부르지만, 정직한 이름은 **마스크 테이블의 low-rank(저랭크) 파라미터화**입니다. 아이디어의 가치가 떨어지진 않지만 포장이 실제보다 화려합니다.

---

## 3️⃣ 방법 (2) — Outfit Ranking Loss

*이 절을 두는 이유: 기존 triplet loss 가 코디에서 아이템 한 장만 anchor 로 뽑아 쓰는 것이 정보 낭비라는 문제의식에서 출발하기 때문입니다.*

![CSA-Net Figure 3 — outfit ranking loss](figures/csa_net_fig3.png)
*Figure 3. 코디 안 모든 아이템(왼쪽)에서 positive(위)와 negative 여러 개(아래)로 거리를 각각 재고, 그것을 하나의 Dpositive / Dnegative 로 합친다.*

### 3.1 기존 triplet loss 와의 차이

기존 triplet loss 는 코디에서 **아무 아이템 하나**를 anchor 로 뽑아 씁니다. 이 논문은 코디 **전체**를 씁니다.

### 3.2 수식

학습 샘플 하나는 Υ = {O, p, N} — 코디 O = {I₁ᵒ, …, Iₙᵒ}, 정답 p, 오답 집합 N = {I₁ⁿ, …, I_mⁿ} 입니다.

**코디 아이템들의 임베딩** (타깃 카테고리 슬롯에 정답 카테고리를 넣음):

$$\mathbf{f}_i^o = \psi(\mathbf{I}_i^o,\; c(\mathbf{I}_i^o),\; c(\mathbf{I}^p)); \quad i = 1 \to n \qquad (2)$$

**정답 이미지의 임베딩** (소스 카테고리 슬롯에 코디 아이템 카테고리를 넣음):

$$\mathbf{f}_i^p = \psi(\mathbf{I}^p,\; c(\mathbf{I}_i^o),\; c(\mathbf{I}^p)); \quad i = 1 \to n \qquad (3)$$

> 정답 이미지 한 장이 임베딩을 **n개** 갖습니다. 코디 아이템마다 소스 카테고리가 달라 attention 이 달라지기 때문입니다. 오답도 마찬가지 (식 4).

**코디와 아이템 s 의 거리** = 코디 안 모든 아이템과의 쌍거리 평균:

$$D_{outfit}(O, s) = \frac{1}{n}\sum_{i=1}^{n} d(\mathbf{f}_i^o, \mathbf{f}_i^s) \qquad (5)$$

**오답 m개의 거리를 하나로 집계** (φ = min 또는 average):

$$D_N = \varphi(D_{n_1}, \ldots, D_{n_m}) \qquad (6)$$

**최종 손실** (margin m = 0.3):

$$l(O, p, N) = \max(0,\; D_p - D_N + m) \qquad (7)$$

### 3.3 대칭성 — 카테고리 순서를 뒤집어도 같은 이유

식 (2)와 (3)을 비교하면, 질의 쪽과 후보 쪽이 **완전히 같은 카테고리 쌍** (c(Iᵢᵒ), c(Iᵖ)) 을 씁니다. 그러므로 둘은 같은 유효 마스크를 쓰고, 거리는 대칭입니다.

논문이 4.4절에서 "카테고리 순서를 뒤집어(order flipping) 학습해도 성능 변화가 없더라"고 보고한 것은 이 대칭성의 자연스러운 귀결입니다. 그리고 이는 곧 실제 카테고리 조합이 121개가 아니라 **66개**뿐임을 확인해 줍니다.

---

## 4️⃣ 인덱싱·검색 파이프라인

*이 절을 두는 이유: 이 논문의 존재 이유가 "실제로 인덱스에 넣을 수 있다"이므로, 그 절차와 비용을 구체적으로 봐야 주장의 진위를 판정할 수 있습니다.*

![CSA-Net Figure 4 — 색인과 검색](figures/csa_net_fig4.png)
*Figure 4. 위(Indexing): DB 아이템을 타깃 카테고리마다 임베딩해 인덱스에 넣어둔다. 아래(Testing): 질의 코디의 각 아이템으로 KNN 검색 후 점수를 융합한다.*

| 단계 | 하는 일 | 비용 |
|---|---|---|
| **색인 (오프라인)** | DB 아이템 하나를 타깃 카테고리 **전부**와 짝지어 임베딩을 여러 벌 생성 | 저장 비용이 카테고리 수에 비례 → **11배** (논문도 "embedding size is linear in the number of high-level categories"라고 솔직히 인정) |
| **검색 (온라인)** | 질의 코디의 **아이템마다** KNN 검색 → 각 랭킹 점수를 average fusion 으로 융합 | 코디 크기 n 에 비례하는 n번의 ANN 질의 |

ANN 라이브러리는 hnswlib 를 사용했다고만 언급되어 있습니다.

> 📌 실제 추론 때 무엇이 입력으로 들어가는지(아이템 1개 vs 아웃핏 전체), 구체적인 카테고리 열거 예시는 **Q1** 에서 자세히 다룹니다.

---

## 5️⃣ 구현 세부 (Implementation Details)

*이 절을 두는 이유: 재현 가능성을 판정하려면 무엇이 공개되고 무엇이 빠졌는지 한눈에 봐야 하기 때문입니다.*

| 항목 | 값 |
|---|---|
| backbone CNN | ResNet18 (ImageNet 사전학습) |
| embedding size | 64 (baseline 과 공정 비교 위해 통일) |
| subspace 개수 k | **5** (SCE-Net [15]과 동일, ablation 은 그 논문 것을 인용) |
| optimizer | ADAM |
| mini-batch | 96 |
| margin | 0.3 |
| learning rate | 5e-5, **선형 감소**(linear decay to zero), warmup ratio = 0 (초기 lr 이 이미 작아서) |
| negative 선택 | 정답과 **같은 카테고리**에서 랜덤 샘플 후 **semi-hard** 선택 |
| 최적화 트릭 | online mining — 이미지당 subspace 임베딩을 **한 번만** 계산해 여러 쌍거리에 재사용 |
| 텍스트 특징 | ❌ **사용 안 함** (baseline 들은 사용) |
| **미공개 항목** | 오답 집합 크기 m, 마스크 초기화 방식, 선택된 fine-grained 카테고리 목록, 코드 전체 |

---

## 6️⃣ 실험 요약

*이 절을 두는 이유: 두 기여가 각각 얼마나 벌었는지를 숫자로 분리해야 7장의 급소 분석이 가능해집니다.*

### 6.1 호환성 예측 / FITB

*모든 방법이 ResNet18 + embedding 64 로 통일된 공정 비교입니다.*

| 방법 | 특징 | FITB (D / non-D) | AUC (D / non-D) |
|---|---|---|---|
| Siamese-Net [16] | 이미지만 | 51.80 / 52.90 | 0.81 / 0.81 |
| Type-aware [16] | 이미지 + 텍스트 | 55.65 / 57.83 | 0.84 / 0.87 |
| SCE-Net average [15] | 이미지 + 텍스트 | 53.67 / 59.07 | 0.82 / 0.88 |
| **CSA-Net + outfit ranking loss (제안)** | **이미지만** | **59.26 / 63.73** | **0.87 / 0.91** |

*(D = disjoint set, non-D = non-disjoint set)*

텍스트를 안 쓰고도 이겼고, **검색이 불가능한 원본 SCE-Net(61.6 / 0.91)까지 FITB 에서 넘어섰습니다.**

### 6.2 Ablation — 손실 함수

| 손실 함수 | FITB (D / non-D) | AUC (D / non-D) |
|---|---|---|
| Triplet loss | 56.17 / 60.91 | 0.85 / 0.90 |
| **Outfit ranking loss** | **59.26 / 63.73** | **0.87 / 0.91** |

### 6.3 Ablation — 집계 함수 φ

| 집계 함수 | FITB (D / non-D) | AUC (D / non-D) |
|---|---|---|
| Average | 56.19 / 60.0 | 0.84 / 0.89 |
| **Min** | **59.26 / 63.73** | **0.87 / 0.91** |

논문의 설명: min 이 더 좋은 이유는 **"더 어려운 오답(hard negative)을 랭킹 손실 학습에 넣기 때문"**. → 이 문장이 7장 (1)의 출발점입니다.

### 6.4 검색 (recall@top k)

*카테고리당 3000장, non-disjoint 는 27개 / disjoint 는 16개 fine-grained 카테고리의 평균.*

| 방법 | D top10 / 30 / 50 | non-D top10 / 30 / 50 |
|---|---|---|
| Type-aware [16] | 3.66 / 8.26 / 11.98 | 3.50 / 8.56 / 12.66 |
| SCE-Net average [15] | 4.41 / 9.85 / 13.87 | 5.10 / 11.20 / 15.93 |
| **CSA-Net (제안)** | **5.93 / 12.31 / 17.85** | **8.27 / 15.67 / 20.91** |

절대 수치가 낮아 보이지만 무작위 기준선 대비로는 큰 값입니다 (→ Q6 참조).

---

## 7️⃣ 급소 — 이 논문의 가장 약한 곳

*이 절을 두는 이유: 논문이 내세운 두 기여 중 하나는 실험 표를 겹쳐 보면 설명이 뒤집히기 때문에, 인용하거나 후속 연구의 baseline 으로 쓸 때 반드시 알고 있어야 합니다.*

### (1) ⭐ Ablation 두 개를 겹쳐 보면 두 번째 기여가 사라진다

논문의 Table 2(6.2절)와 Table 3(6.3절)을 **같이** 놓아보세요. non-disjoint FITB 기준입니다.

| 설정 | FITB |
|---|---|
| CSA-Net + **triplet loss** | **60.91** |
| CSA-Net + outfit ranking loss, **average** 집계 | **60.0** ⬅ triplet보다 **낮다** |
| CSA-Net + outfit ranking loss, **min** 집계 | **63.73** |

즉 "코디 전체를 본다"는 아이디어를 정직하게(average) 구현하면 **평범한 triplet loss 보다 오히려 낮습니다.** +2.8%p 이득 전부가 min 집계에서 나오는데, min 은 논문 자신도 인정하듯 그냥 **hardest negative mining** 입니다. 게다가 semi-hard negative 선택은 이미 별도로 쓰고 있습니다.

> **정리: 기여 #2("outfit ranking loss")의 실질은 "코디 전체를 고려한 랭킹"이 아니라 "negative 를 더 어렵게 뽑는 법"입니다.** 논문의 서사와 데이터가 어긋나는 지점이고, 이 조합 분석은 논문 본문에 없습니다.

### (2) 아키텍처 기여를 뜯어보면

non-disjoint FITB 기준 기여 분해:

| 기여 | 계산 | 값 |
|---|---|---|
| 아키텍처 (CSA-Net) | 60.91 − 59.07 (SCE-Net average) | ≈ **+1.8%p** |
| 손실 (outfit ranking) | 63.73 − 60.91 | ≈ **+2.8%p** |

그런데 비교 대상 SCE-Net average 는 **텍스트를 쓰는 조건**이라 이 1.8%p 도 완전 통제된 수치가 아닙니다. **같은 outfit ranking loss 로 SCE-Net average 를 학습시킨 대조군**이 없어서, 아키텍처 자체의 순수 기여는 끝까지 미확정입니다.

### (3) "대규모 검색용"이라면서 효율 수치가 0건

논문의 최대 셀링 포인트가 scalability(확장성)인데, **인덱스 크기·질의 지연시간(latency)·QPS·메모리 어느 것도 보고하지 않습니다.** 검색 실험도 카테고리당 3000장 — 대규모라 부르기 어렵습니다. hnswlib 를 썼다는 언급 한 줄이 전부입니다. 게다가 저장 비용은 카테고리 수만큼(11배) 늘어나므로, "확장성"은 절대적 성질이 아니라 SCE-Net 대비 **상대적** 우위입니다.

또한 Type-aware 는 66개 subspace 투영을 미리 계산하면 인덱싱이 되고, 실제로 논문도 baseline 실험에서 그렇게 돌렸습니다(4.3절). 따라서 **"기존 방법은 인덱싱 불가"라는 주장이 엄밀히 성립하는 대상은 SCE-Net 과 GCN 뿐입니다.**

### (4) 검색 벤치마크 설계의 인위성

| 문제 | 내용 |
|---|---|
| 정답이 1장뿐 | 어울리는 다른 아이템은 전부 오답 처리 → 지표가 성능을 **구조적으로 과소평가**. 논문도 인정하지만 인간 평가 같은 보완이 전혀 없음 |
| 후보를 미리 좁힘 | 후보를 **정답과 같은 fine-grained 카테고리**로 제한 → 실제 서비스보다 훨씬 쉬운 세팅 |
| 학습 이미지를 distractor 로 사용 | non-disjoint 에서는 모델이 **본 적 있는** 이미지가 방해꾼으로 들어감 → disjoint(5.93) 대 non-disjoint(8.27) 격차에 오염 요인 |
| 두 열이 비교 불가 | 평가 카테고리 수가 다름 (non-D 27개 / D 16개) |

### (5) 재현성

Amazon 논문이라 코드 미공개. 오답 집합 크기 m, 마스크 초기화, 선택된 fine-grained 카테고리 목록이 전부 빠져 있고, 새로 만들었다는 검색 벤치마크의 공개 언급도 없습니다. **후속 연구가 이 검색 수치를 그대로 재현하기는 어렵습니다.**

### (6) AUC 는 이미 포화

0.91 은 원본 SCE-Net 과 **동일**합니다. 실제로 앞선 건 FITB 와 검색뿐입니다.

---

## 8️⃣ 그래도 좋은 점

*이 절을 두는 이유: 급소가 많다고 해서 논문의 설계 판단까지 틀린 건 아니며, 실제로 계승할 가치가 있는 통찰이 분명히 있기 때문입니다.*

- **문제 설정의 전환이 정확합니다.** "분류 점수로 랭킹 매기면 되지 않나"를 **"인덱싱 가능해야 실서비스다"**로 다시 정의한 것이 이 논문의 진짜 기여입니다
- **덜어냄으로써 얻었습니다.** attention 의 조건을 이미지 쌍에서 카테고리 쌍으로 **약화시킨 것 자체가** 곧 인덱싱 가능성입니다. 성능까지 오른 건 low-rank 제약의 정규화 효과라는 보너스
- 텍스트 없이, 텍스트 쓰는 baseline 을 이긴 건 공정성 측면에서 유리한 결과
- 실패 사례(Figure 6)를 "색·질감·스타일이 너무 비슷해서" / "정답이 아닐 뿐 실제로는 어울리는 아이템이 여럿이라서"로 분류해 제시한 건 정직합니다

---

## 9️⃣ 계보 안에서의 위치

*이 절을 두는 이유: 이 논문의 기여는 독립적 발명이 아니라 앞선 세 논문의 제약을 하나씩 교환한 결과라서, 계보를 보면 무엇이 새로운지가 즉시 드러납니다.*

```
CSN [17]              조건부 유사도 마스크 (conditional similarity mask) 개념 도입
   ↓
Type-aware [16]       카테고리 쌍마다 독립 마스크 66개 — 인덱싱 가능, 학습 비효율
   ↓
SCE-Net [15]          마스크 공유 + 이미지 쌍 조건 — 성능 ↑, 검색 ↓ (인덱싱 불가)
   ↓
CSA-Net (본 논문)     마스크 공유 + 카테고리 조건 — 성능 ↑, 검색 복원
   ↓
(이후) Transformer 계열   코디 전체를 시퀀스로 인코딩 + "타깃 아이템 토큰"으로 검색 임베딩 추출
```

| 모델 | 마스크 공유? | 조건부여 대상 | 인덱싱 |
|---|---|---|---|
| Type-aware [16] | ❌ (66개 독립) | 카테고리 쌍 | ✅ |
| SCE-Net [15] | ✅ | **이미지 쌍** | ❌ |
| **CSA-Net** | ✅ | **카테고리 쌍** | ✅ |

이후 흐름은 Transformer 로 넘어가서, 코디 전체를 시퀀스로 인코딩하고 "타깃 아이템 토큰"으로 검색 임베딩을 뽑는 방향(OutfitTransformer 계열)이 CSA-Net 을 표준 baseline 으로 인용하며 FITB 를 60% 후반대까지 올립니다.

---

## 💬 Q&A

### Q1. 실제 추론 때 입력은 뭔가? 옷 아이템 하나인가, 아웃핏(여러 벌)인가?

*모델을 실제로 붙여 쓰려면 무엇을 넣어야 하는지가 가장 먼저 걸리는 부분입니다.*

**둘 다 맞고, 층이 다릅니다.**

- **네트워크(CSA-Net) 입력 = 옷 아이템 딱 1장**
- **시스템 전체 입력 = 아웃핏(여러 벌) + 찾을 카테고리**

CSA-Net 은 아웃핏을 통째로 먹지 못합니다. 아이템 하나씩 n번 돌리고, 결과를 **바깥에서 평균으로 합칩니다.**

#### 네트워크 한 번의 입력 (3개)

| 입력 | 예시 | 형태 |
|---|---|---|
| 소스 이미지 | 청바지 사진 1장 | 이미지 |
| 소스 카테고리 | bottoms | one-hot 11차원 |
| 타깃 카테고리 | shoes | one-hot 11차원 |

출력은 64차원 임베딩 1개. **아웃핏이라는 개념이 여기엔 아예 없습니다.**

#### 색인할 때 (오프라인, DB 쪽)

DB 에 신발 사진 한 장이 들어왔다고 합시다. 이 신발이 나중에 **어떤 아이템과 짝지어 비교될지 지금은 모릅니다.** 그래서 가능한 상대 카테고리를 전부 열거해 임베딩을 미리 다 만들어 둡니다.

```
신발 이미지 → ψ(신발, bags,       shoes) → 임베딩 #1
            → ψ(신발, tops,       shoes) → 임베딩 #2
            → ψ(신발, bottoms,    shoes) → 임베딩 #3
            → ψ(신발, sunglasses, shoes) → 임베딩 #4
            ...  (카테고리 11개 전부)
```

신발 한 켤레가 임베딩 **11개**를 갖게 되고, 이게 저장 비용 11배의 정체입니다 (→ Q5).

#### 검색할 때 (온라인, 질의 쪽)

질의 아웃핏이 가방·선글라스·원피스·아우터 **4개**이고, 빠진 게 신발이라고 합시다.

> ⚠️ **중요**: 빠진 카테고리(신발)는 **모델이 맞히는 게 아니라 사람/시스템이 알려줘야 합니다.** 이건 입력이지 출력이 아닙니다.

| 단계 | 하는 일 |
|---|---|
| ① | 가방 → ψ(가방, bags, **shoes**) → 임베딩 |
| ② | 선글라스 → ψ(선글라스, sunglasses, **shoes**) → 임베딩 |
| ③ | 원피스 → ψ(원피스, all-body, **shoes**) → 임베딩 |
| ④ | 아우터 → ψ(아우터, outerwear, **shoes**) → 임베딩 |

즉 **CNN 을 4번 돌립니다.** 그다음 4번의 KNN 검색을 합니다.

| 질의 | 검색 대상 인덱스 |
|---|---|
| 가방 임베딩 | 신발들의 **(bags, shoes)** 임베딩 모음 |
| 선글라스 임베딩 | 신발들의 **(sunglasses, shoes)** 임베딩 모음 |
| 원피스 임베딩 | 신발들의 **(all-body, shoes)** 임베딩 모음 |
| 아우터 임베딩 | 신발들의 **(outerwear, shoes)** 임베딩 모음 |

**질의 쪽과 후보 쪽이 반드시 같은 카테고리 쌍을 써야 합니다.** 그래야 같은 마스크(같은 공간)에서 거리를 재니까요 (→ 3.3절 대칭성). 다른 쌍끼리 비교하면 애초에 의미가 없습니다.

마지막으로 랭킹 4개를 **average fusion** 으로 합쳐서 최종 순위를 냅니다.

#### 정리

```
아웃핏 4개 + "신발 찾아줘"
        │
        ├─ CNN+CSA-Net 4번 (아이템별 독립, 서로 안 봄)
        │
        ├─ KNN 검색 4번 (서로 다른 4개 인덱스에서)
        │
        └─ 랭킹 4개 평균 → 최종 신발 목록
```

| 층위 | 입력 |
|---|---|
| 네트워크 ψ | **아이템 1개** + 카테고리 2개 |
| 검색 시스템 | **아웃핏 n개** + 타깃 카테고리 1개 |
| 계산량 | CNN n번, KNN n번 |

#### ⭐ 여기서 드러나는 구조적 한계

**아이템끼리는 서로를 절대 보지 못합니다.** 가방 임베딩을 만들 때 원피스가 뭔지 모릅니다.

그래서 "아웃핏 전체를 고려한다"는 말이 실제로 작동하는 지점은 딱 두 군데뿐입니다.

1. **학습 때** — 손실 함수에서 거리 n개를 평균 (식 5)
2. **추론 때** — 랭킹 n개를 평균 (score fusion)

둘 다 **바깥에서 평균 내는 것**이지, 네트워크 안에서 상호작용이 일어나는 게 아닙니다. 파란 원피스 + 검은 아우터라는 **조합**에서만 생기는 스타일 같은 건 원리상 포착 못 합니다.

이후 OutfitTransformer 계열이 아웃핏을 하나의 시퀀스로 넣어 self-attention 으로 아이템끼리 서로 보게 만든 게, 정확히 이 한계를 겨냥한 후속 작업입니다 (→ 9장 계보).

---

### Q2. 그럼 이미지 한 장만 갖고 어울리는 바지·신발·아우터·가방을 다 찾을 수 있나?

*실서비스에서 가장 흔한 시나리오("이 옷에 뭐 매치하지?")가 이것이라 별도로 확인해 둘 가치가 있습니다.*

**됩니다.** 아웃핏 크기 n=1 은 식 5에서 평균 낼 항이 하나뿐인 경우라 수식상 아무 문제 없습니다. 애초에 네트워크가 아이템을 하나씩만 보니까요 (→ Q1).

#### 되는 것 — CNN 은 딱 1번

```
셔츠 이미지 → CNN 1번 → 특징 x (재사용!)
                          │
    ├─ 마스크(tops→bottoms)   → 임베딩 → 바지 인덱스 검색   → 어울리는 바지 목록
    ├─ 마스크(tops→shoes)     → 임베딩 → 신발 인덱스 검색   → 어울리는 신발 목록
    ├─ 마스크(tops→outerwear) → 임베딩 → 아우터 인덱스 검색 → 어울리는 아우터 목록
    └─ 마스크(tops→bags)      → 임베딩 → 가방 인덱스 검색   → 어울리는 가방 목록
```

**비용이 거의 안 듭니다.** CNN 은 **딱 1번**만 돌리고, 그 특징 x 를 재사용해서 마스크만 바꿔 끼우면 됩니다. 마스크 곱하기는 벡터 원소별 곱 한 번이라 사실상 공짜입니다. 카테고리 10개를 다 뒤져도 CNN 비용은 1회분입니다.

> 단, 찾을 카테고리는 매번 지정해줘야 합니다. "이 셔츠에 뭐가 빠졌는지 알아서 찾아줘"는 안 되고, "바지 찾아줘" "신발 찾아줘"를 각각 시켜야 합니다.

#### ⚠️ 조심할 것 — 4개가 서로 어울린다는 보장은 없다

찾아온 바지·신발·아우터·가방은 **각자 셔츠하고만** 어울립니다. **자기들끼리는 한 번도 비교된 적이 없습니다.**

| 결과 | 셔츠와의 관계 | 문제 |
|---|---|---|
| 정장 바지 | ✅ 흰 셔츠와 잘 어울림 | |
| 러닝화 | ✅ 흰 셔츠와 잘 어울림 | |
| **정장 바지 + 러닝화** | — | ❌ **서로는 안 어울림** |

각 검색이 완전히 독립이라 이런 조합이 그냥 통과합니다.

#### 해결 — 하나씩 붙여가며 굴리기 (greedy 방식)

논문이 명시하진 않았지만, 구조상 이렇게 쓰는 게 맞습니다.

| 라운드 | 질의 아웃핏 | CNN 호출 | 찾는 것 |
|---|---|---|---|
| 1 | 셔츠 (1개) | 1번 | 바지 → 하나 확정 |
| 2 | 셔츠 + 바지 (2개) | +1번 | 신발 → 하나 확정 |
| 3 | 셔츠 + 바지 + 신발 (3개) | +1번 | 아우터 → 하나 확정 |
| 4 | 4개 | +1번 | 가방 |

라운드가 갈수록 질의 아이템이 늘어나서 **평균 낼 랭킹이 많아지고**, 결과가 안정됩니다. 앞서 확정한 아이템들이 다음 검색의 제약으로 작동하니까 전체가 하나로 묶입니다.

이전 라운드 아이템들의 CNN 특징은 캐시해두면 되므로, 라운드당 CNN 은 새 아이템 1개분만 추가됩니다.

#### 성능은 어떨까 — 논문에 답이 없다

FITB 도 검색 실험도 전부 아이템이 여러 개 있는 아웃핏으로만 평가했고, **아이템 1장짜리 질의는 한 번도 측정하지 않았습니다.**

원리상으로는 이렇게 예상됩니다.

| 질의 아이템 수 | 예상 |
|---|---|
| 1장 | 랭킹 평균 효과 없음 → **노이즈가 큼** |
| 4~5장 | 랭킹 여러 개 평균 → **안정적** |

논문 수치(recall@10 8.27%)는 여러 장 조건에서 나온 것이니, 1장 조건에서는 **더 낮을 것으로 봐야 합니다.** 검증되지 않은 영역이라 직접 재봐야 합니다.

#### 한 줄

**된다. CNN 1번으로 카테고리별 추천 목록을 전부 뽑을 수 있다. 다만 목록끼리는 서로 조율되지 않으므로, 완성된 한 벌을 만들려면 하나씩 붙여가며 재검색해야 한다.**

---

### Q3. 공개된 비공식 구현 두 개는 논문과 일치하게 작성됐나?

*원저자 코드가 없으므로(Amazon 사내), 재현하려면 서드파티 구현에 의존하게 됩니다. 그 구현들을 믿을 수 있는지 먼저 확인해야 합니다.*

**결론부터: 둘 다 논문 재현이 아니고, 둘 다 실행조차 되지 않습니다.**

| 저장소 | 커밋 | 저자 선언 | 결과 보고 |
|---|---|---|---|
| [Jungjaewon/Fashion_Outfit_Complementary_Item_Retrieval](https://github.com/Jungjaewon/Fashion_Outfit_Complementary_Item_Retrieval) | `a447b66` | "the code is not tested yet", "testing: Not implemented yet" | "Empty for now" |
| [bigohofone/csa-net](https://github.com/bigohofone/csa-net) (= owj0421/csa-net) | `07db8e4` | "🚧 This project is under construction 🚧", Train/Test 사용법 칸이 비어 있음 | 없음 |

---

#### A. Jungjaewon 저장소

##### A-1. 실행 자체가 불가능 — 크래시 지점 4곳

| # | 위치 | 문제 |
|---|---|---|
| ① | **solver.py 27번째 줄** | `config['TRAINING_CONFIG']['BASE_LR']` 를 읽는데 config.yml 에는 `LR` 만 있음 → **KeyError, 생성자에서 즉사** |
| ② | **data_loader.py 44·67번째 줄** | one-hot 길이를 `num_conditions`(=5)로 만듦. 두 개 이어붙여 **10차원**인데 model.py 27번째 줄의 cate_net 입력은 `num_category*2` = **20차원** → shape mismatch. 게다가 카테고리 번호가 5 이상이면 IndexError |
| ③ | **solver.py `dict2gpu`** | `data_dict[key].to(self.gpu)` — 반환값을 **버림**. `.to()` 는 텐서에서 in-place 가 아니므로 데이터는 CPU에 남고 모델은 GPU → device mismatch |
| ④ | **data_loader.py 47번째 줄** | `torch.LongTensor(positive_cate)` — 정수를 넣으면 스칼라가 아니라 **길이 N짜리 미초기화 텐서**가 생성됨 |

##### A-2. 조용히 틀리는 논리 오류 (더 심각)

**ⓐ 브로드캐스팅으로 거리 행렬이 배치×배치가 됨**

```python
D_p = torch.zeros((batch, 1))          # (96, 1)
D_p += torch.mean((f_o-f_p)**2, dim=1) # (96,)
```

실측: `(4,1) + (4,)` → 결과가 **(4,4)**. 스칼라 거리 벡터여야 할 D_p 가 배치 전체의 교차 행렬로 부풀어 오릅니다.

**ⓑ 오답 집계가 논문의 min 도 average 도 아님**

```python
for m in range(self.num_negative):
    D_n_m = 0
    for n in range(self.num_outfit):
        f_n = CSA(negative_image_{n}, ...)   # ← m이 아니라 n으로 인덱싱!
    D_n += D_n_m
D_n = D_n / self.batch_size                  # ← 배치 크기로 나눔?
```

세 가지가 동시에 틀렸습니다.

1. 안쪽에서 오답 이미지를 **m 이 아니라 n 으로** 꺼냅니다. m 루프를 5번 돌아도 **매번 똑같은 값**이 나옵니다. 결국 D_n = 5 × (동일값)
2. 논문 식 6의 φ 는 **min**(핵심 기여!)인데 코드는 **합**입니다. min 도 average 도 없습니다
3. 마지막에 오답 개수(5)가 아니라 **배치 크기(96)로 나눕니다**. 근거 없는 스케일

7장 (1)에서 짚었듯 이 논문 기여 #2의 실질은 min 집계인데, **그 min 이 코드에 아예 없습니다.**

**ⓒ 학습 목표의 부호가 50% 확률로 뒤집힘**

data_loader.py 39~42번째 줄:

```python
if random.random() > 0.5 and self.get_negative_changing(...):
    loss_tensor = -1
else:
    loss_tensor = +1
```

`MarginRankingLoss(x1, x2, y)` 는 `max(0, -y·(x1-x2) + margin)` 입니다. 논문 식 7을 얻으려면 **y 는 항상 -1** 이어야 합니다. 여기서는 절반이 +1 이 되어, **그 절반은 정답을 오답보다 멀게 밀어내도록** 학습됩니다.

덤으로 `get_negative_changing` 함수는 `positive[0] = positive_path` 처럼 **자기 자신을 자기에게 대입**할 뿐 아무것도 바꾸지 않습니다.

**ⓓ 질의와 후보가 서로 다른 공간에서 비교됨**

Q1 에서 설명한 "질의 쪽과 후보 쪽이 같은 카테고리 쌍을 써야 한다"는 원칙이 오답에서 깨집니다.

| 임베딩 | 사용하는 카테고리 쌍 |
|---|---|
| f_o (코디 아이템) | (정답카테고리, **코디아이템카테고리**) |
| f_p (정답) | (정답카테고리, **코디아이템카테고리**) ✅ 일치 |
| f_n (오답) | (정답카테고리, **오답카테고리**) ❌ **불일치** |

오답만 다른 마스크를 거칩니다. 서로 다른 subspace 에서 잰 거리를 margin 으로 비교하게 됩니다.

**ⓔ 학습률이 즉시 붕괴**

`ExponentialLR(gamma=0.95)` 의 `.step()` 을 **에폭이 아니라 매 이터레이션**마다 호출합니다. 100 이터레이션이면 0.95^100 ≈ 원래의 0.6%. 논문의 "선형 감소, warmup 0"과 완전히 다릅니다.

##### A-3. 인덱싱·평가 — 논문 3.3절이 사실상 미구현

| 논문 | 코드 |
|---|---|
| DB 아이템을 **타깃 카테고리 전부 열거**해 색인 | 열거 없음. 색인 대상도 DB 후보가 아니라 **코디 아이템**(f_o) |
| 아이템별 KNN 후 **average fusion** | 융합 없음 |
| **recall@top k** | 없음 |
| FITB 정확도 / 호환성 AUC | 없음 |

게다가 `p.add_items(data, data_labels)` 에서 `data_labels[:] = 1` — **모든 아이템 라벨이 1**입니다. 그 뒤 `labels == positive_cate` 로 정답을 판정하니, 평가 코드 자체가 무의미합니다.

##### A-4. 논문과 맞는 부분

ResNet18 / embedding 64 / subspace 5 / batch 96 / margin 0.3 / ADAM ✅, 식 1 구현 ✅, hnswlib 사용 ✅(사용법은 틀렸지만).

**모델 구조(model.py)만 논문에 부합하고 나머지는 전부 어긋납니다** (→ Q4 에서 그 model.py 를 따로 검증). 학습률 1e-4(config 의 `10e-5`)도 논문 5e-5 의 2배, 카테고리 수도 10(논문 11)입니다.

---

#### B. bigohofone / owj0421 저장소

코드 품질은 확실히 낫습니다. 하지만 **논문의 두 기여 중 하나가 통째로 비어 있습니다.**

##### B-1. 결정적 문제 — Outfit Ranking Loss 가 `pass`

`src/models/loss.py`:

```python
class OutfitRankingLoss(nn.Module):
    pass          # ← 논문 기여 #2, 본문이 없음
```

실제로 쓰이는 건 같은 파일의 `InBatchTripletMarginLoss` 와, train.py 안에 따로 정의된 `InBatchOutfitRankingLoss` 입니다. 후자를 뜯어보면 논문 식 5~7 과 **연산 순서가 다릅니다.**

| | 논문 | 코드 |
|---|---|---|
| 순서 | ① 코디 전체 평균 → ② 오답들에 min → ③ hinge 1번 | ① **아이템별** min → ② **아이템별** hinge → ③ 마지막에 평균 |
| 결과 | 아웃핏 단위 랭킹 | 사실상 **아이템 단위 triplet 의 평균** |

논문이 "단일 아이템이 아니라 아웃핏 전체로 계산한다"고 강조한 바로 그 지점(식 5의 평균이 hinge **안쪽**에 있어야 함)이 뒤집혔습니다.

##### B-2. 오답을 in-batch 로 뽑아 카테고리 제약이 사라짐

논문은 **정답과 같은 카테고리**에서 오답을 뽑고 semi-hard 를 선택합니다. 코드는 배치 안 다른 샘플의 정답을 오답으로 씁니다.

- 가방이 정답인데 오답으로 신발·모자가 들어옴 → **카테고리만 봐도 구분 가능한 쉬운 오답**
- semi-hard 가 아니라 **hardest**(argmin)

FITB 의 오답이 같은 카테고리에서 뽑힌다는 점을 생각하면, 학습 분포와 평가 분포가 어긋납니다.

##### B-3. 패딩이 손실에 그대로 들어감

`preprocess` 가 짧은 아웃핏을 **검은 이미지 + 'unknown' 카테고리**로 채우고 `mask` 를 만들어 손실 함수에 넘깁니다. 그런데 `InBatchOutfitRankingLoss.forward` 는 **인자로 mask 를 받기만 하고 본문에서 한 번도 쓰지 않습니다.** 검은 이미지와의 거리가 손실에 그대로 누적됩니다.

##### B-4. import 부터 실패

| 문제 | 내용 |
|---|---|
| `from ..data.datasets import polyvore` | **`src/data/datasets/` 디렉토리 자체가 없음** |
| `from ..utils.utils import seed_everything` | **`src/utils/utils.py` 없음** (있는 건 distributed_utils, logger_utils) |
| `__init__.py` | 저장소 전체에 **0개** → 상대 import 불가 |

##### B-5. 학습 루프의 나머지 버그

| # | 문제 |
|---|---|
| ① | `save_checkpoint` 와 `save_results` 가 **같은 경로**(`epoch_N_score.json`)에 씀. JSON 이 체크포인트를 덮어씀 |
| ② | `save_checkpoint(args, ...)` 에 cfg 대신 argparse 객체를 넘김 → 나중에 `CSANetConfig(**{...'polyvore_dir'...})` → TypeError |
| ③ | `load_checkpoint` 가 `enc.load_state_dict(...)` 의 반환값(`_IncompatibleKeys`)을 모델로 되돌려줌 → 2에폭째 크래시 |
| ④ | `valid_logs = None` 인데 `{**train_log, **valid_log}` → 1에폭 끝에서 TypeError |
| ⑤ | `is_distributed = torch.distributed.is_initialized()` 를 **setup() 호출 전에** 확인 → 항상 False → setup 안 됨 → 뒤의 `dist.barrier()` 크래시 |
| ⑥ | valid_step 전체가 **주석 처리** → 학습 중 FITB 검증 없음 |
| ⑦ | `torch.autograd.set_detect_anomaly(True)` 가 전역에 켜진 채 |

##### B-6. test.py / 2_test_compatibility.py 는 다른 프로젝트 코드

두 스크립트 모두 `load_model(model_type=..., checkpoint=...)` 이 **모델 하나**를 반환한다고 가정하고 `model(query, use_precomputed_embedding=True)` 를 호출합니다. 이 저장소의 `load_model` 은 `(cfg, enc, sp_attn)` **3-튜플**을 반환하고, `CSANetEncoder.forward` 에는 그런 인자가 없습니다.

`--model_type choices=['original','clip']`, `wandb.init(project='outfit-transformer-cir')` 같은 흔적으로 볼 때 저자 본인의 **OutfitTransformer 저장소에서 복사해 온 뒤 손대지 않은 파일**입니다. 실행 불가한 죽은 코드입니다.

##### B-7. 잘한 점

| 항목 | 평가 |
|---|---|
| 카테고리 11개 | ✅ 논문과 일치 (`all-body, bottoms, tops, outerwear, bags, shoes, accessories, scarves, hats, sunglasses, jewellery`) |
| 인코더/어텐션 분리 | ✅ subspace 임베딩을 **이미지당 한 번만** 계산해 재사용 — 논문 4.2절의 online mining 취지와 일치 |
| ⭐ `comb_of_cat` | 카테고리 쌍을 **전부 열거해 룩업 딕셔너리**로 만들어 씀. 2.2절의 "이 attention 은 사실 카테고리 쌍 룩업 테이블"이라는 분석이 **코드로 그대로 확인**됨 |
| 식 1 | ✅ `x * m` 후 w 로 가중합 — 정확 |

한 가지 흥미로운 부작용: 카테고리 임베딩 두 개를 정규화한 뒤 **더하기**(`embs + target_embs`) 때문에 (A,B)와 (B,A)가 **구조적으로 동일**합니다. 논문은 concat 을 쓰고 순서 무관은 실험으로 확인했는데(4.4절), 여기서는 아예 대칭이 강제됩니다.

---

#### C. 논문 대비 항목별 대조

| 항목 | 논문 | Jungjaewon | bigohofone |
|---|---|---|---|
| Backbone | ResNet18 | ✅ | ✅ |
| Embedding 차원 | 64 | ✅ 64 | ❌ 128 |
| Subspace 개수 | 5 | ✅ 5 | ✅ 5 |
| 식 1 (가중합) | — | ✅ | ✅ |
| attention 입력 | one-hot 2개 concat | ❌ 차원 불일치 | ❌ 학습 임베딩 덧셈 |
| 서브넷 구조 | fc-relu-fc-softmax | ✅ | ✅ |
| 카테고리 수 | 11 | ❌ 10 | ✅ 11 |
| 식 5 (코디 평균 거리) | 평균 | ❌ 합 + shape 버그 | △ 평균 위치가 다름 |
| **식 6 (min 집계)** | **min** | ❌ **없음(합)** | △ 아이템별 min |
| 식 7 margin | 0.3 | ✅ 0.3 | ❌ 2.0 |
| Negative | 같은 카테고리 + semi-hard | ❌ 부호 랜덤 반전 | ❌ in-batch, hardest |
| Batch | 96 | ✅ 96 | ❌ 20 |
| Learning rate | 5e-5 | ❌ 1e-4 | ❌ 2e-5 |
| LR 스케줄 | 선형감소, warmup 0 | ❌ 지수감쇠(매 스텝) | ❌ OneCycle, warmup 30% |
| Optimizer | ADAM | ✅ | ❌ AdamW |
| 색인(카테고리 열거) | 필수 | ❌ 대상 자체가 틀림 | ❌ 없음 |
| recall@top k | 주 지표 | ❌ | ❌ |
| FITB 정확도 | 63.73 | ❌ | ❌ 주석 처리 |
| 호환성 AUC | 0.91 | ❌ | △ 함수만 존재, 경로 단절 |
| **실행 가능 여부** | — | ❌ | ❌ |

---

#### D. 종합 판정과 직접 구현 시 체크리스트

| | Jungjaewon | bigohofone |
|---|---|---|
| 성격 | **미완성 초안** — 논리 오류가 광범위 | **다른 프로젝트의 뼈대 이식** — 핵심 손실이 빈 껍데기 |
| 살릴 부분 | `model.py` (구조는 정확) | `csa_net.py` + 카테고리 쌍 열거 방식 |
| 버릴 부분 | solver.py 학습 루프 전체, 인덱싱/평가 전체 | loss.py, test.py 2종, 학습 루프의 체크포인트 처리 |
| 기여 #1 (CSA-Net) | ⭕ 구조는 재현됨 | ⭕ 구조는 재현됨 |
| 기여 #2 (Outfit Ranking Loss) | ❌ min 없음, 부호 뒤집힘 | ❌ 클래스가 `pass`, 대체 구현도 순서 다름 |
| 기여 #3 (검색 파이프라인) | ❌ | ❌ |

구조(기여 #1)는 두 저장소 어느 쪽을 봐도 30줄이면 끝나니 그대로 참고해도 됩니다. **실제로 만들어야 하는 건 나머지 둘입니다.**

1. **손실 함수** — 순서를 반드시 지킬 것: 코디 평균(식 5) → 오답들에 min(식 6) → hinge 한 번(식 7). 7장 (1)에서 짚었듯 **min 을 빼면 성능이 triplet 이하로 떨어집니다.** 이 논문에서 min 은 옵션이 아니라 성능의 전부입니다
2. **오답 샘플링** — 정답과 같은 카테고리에서 뽑고 semi-hard 선택. in-batch 로 대충 하면 안 됩니다
3. **색인** — DB 아이템마다 타깃 카테고리를 열거해 임베딩을 여러 벌 만들고, 질의와 **같은 카테고리 쌍 인덱스**에서만 검색. 그다음 아이템별 랭킹을 average fusion (→ Q1)

논문에 원저자 코드가 없으니, 이 세 가지는 어느 저장소에서도 가져올 게 없습니다.

---

### Q4. Jungjaewon 구현의 `model.py` forward 는 정확한가?

*Q3 에서 "모델 구조만 맞다"고 판정했으므로, 그 한 부분이 정말 식 1과 같은지 따로 검증합니다.*

문제의 코드:

```python
def forward(self, image, concat_categories):
    feature_x = self.backbone(image)                          # (B, 64)
    feature_x = feature_x.unsqueeze(dim=1)                    # (B, 1, 64)
    b, _, _ = feature_x.size()
    feature_x = feature_x.expand((b, self.num_conditions, self.embedding_size))

    index = Variable(torch.LongTensor(range(self.num_conditions)))
    if image.is_cuda:
        index = index.cuda()
    index = index.unsqueeze(dim=0).expand((b, self.num_conditions))

    embed = self.masks(index)                                 # (B, 5, 64)
    embed_feature = embed * feature_x

    attention_weight = self.cate_net(concat_categories)       # (B, 5)
    attention_weight = attention_weight.unsqueeze(dim=2)
    attention_weight = attention_weight.expand((b, self.num_conditions, self.embedding_size))

    weighted_feature = embed_feature * attention_weight
    final_feature = torch.sum(weighted_feature, dim=1)        # (B, 64)
    return final_feature
```

#### 수치 검증

코드가 하는 계산과, 2.2절에서 유도한 닫힌 형태 `f = x ⊙ (Σ wᵢmᵢ)` 를 같은 입력으로 비교했습니다.

| 항목 | 결과 |
|---|---|
| 최대 절대 오차 | **4.8e-07** (float32 반올림 수준) |
| 출력 shape | (B, 64) ✅ |
| softmax 합 | 1.0 ✅ |

**완전히 동일합니다.**

#### ✅ 맞는 부분

| 논문 | 코드 | |
|---|---|---|
| x = CNN(I) | `feature_x = self.backbone(image)` | ✅ |
| x ⊙ mᵢ (Hadamard product) | `embed * feature_x` | ✅ |
| wᵢ = softmax(MLP(카테고리 concat)) | `self.cate_net(concat_categories)` | ✅ |
| Σᵢ (x ⊙ mᵢ) · wᵢ | `torch.sum(weighted_feature, dim=1)` | ✅ |

세부도 다 맞습니다.

- **softmax `dim=1`** — (B, 5)에서 subspace 축으로 정규화. 배치 축으로 잘못 걸면 망하는데 제대로 됐습니다
- **마스크 차원 = 임베딩 차원(64)** — `backbone.fc` 를 `Linear(512, 64)` 로 갈아끼워 x 를 64차원으로 만든 뒤 마스크를 곱합니다. 논문의 "마스크는 이미지 특징 벡터와 같은 차원"과 임베딩 64를 동시에 만족시키는 유일한 배치입니다. 512차원 풀링 출력에 마스크를 걸었다면 최종 임베딩이 512가 되어 논문과 어긋났을 텐데, 정확히 피했습니다
- **카테고리를 concat 해서 넣는 것** — `Linear(20 → 5)` 에 concat 을 넣으면 결과가 `W_소스·e_소스 + W_타깃·e_타깃` 가 되어 **슬롯마다 다른 가중치 행렬**을 씁니다. 즉 (A,B)와 (B,A)를 구분할 수 있습니다. 논문이 4.4절에서 순서 뒤집기 실험을 할 수 있었던 것도 구조가 비대칭이기 때문입니다. 반면 bigohofone 은 카테고리 임베딩을 **더해서** 넣으므로 순서 구분이 원천 불가 — 이 지점만큼은 Jungjaewon 쪽이 논문에 더 충실합니다

#### ⚠️ 문제 셋

**1. `.cuda()` 가 GPU 0 으로 고정 — 이 저장소 설정에서 실제로 터짐**

```python
if image.is_cuda:
    index = index.cuda()      # ← 디바이스 인자 없음 = cuda:0
```

그런데 config.yml 은 `GPU: 3` 이고 solver 는 `self.CSA.to(3)` 합니다.

| 텐서 | 위치 |
|---|---|
| `self.masks.weight` | cuda:**3** |
| `index` | cuda:**0** |

`self.masks(index)` 에서 device mismatch 로 RuntimeError. GPU 가 1장인 환경에서만 우연히 통과합니다.
**고치려면**: `index = index.to(image.device)` (`is_cuda` 분기 자체가 불필요)

**2. 매 forward 마다 index 텐서를 새로 만들어 GPU 로 복사**

`nn.Embedding` 으로 마스크를 꺼내는 건 CSN 원본에서 온 관행인데, 여기서는 **항상 0~4 전부**를 꺼냅니다. 즉 조회할 게 없습니다. `embed = self.masks.weight.unsqueeze(0)` 한 줄과 같습니다. `Variable` 도 PyTorch 0.4 이후 폐기된 API 입니다.

**3. `expand` 3개가 전부 불필요**

`(B,1,64) * (1,5,64)` 는 브로드캐스팅으로 알아서 `(B,5,64)` 가 됩니다. 정리하면 forward 전체가 두 줄입니다.

```python
w = self.cate_net(concat_categories)          # (B, K)
return feature_x * (w @ self.masks.weight)    # (B, D)
```

#### 논문 미명시 사항

| 항목 | 코드 값 | 논문 |
|---|---|---|
| cate_net 은닉 폭 | **5** (= 출력과 동일) | 명시 없음 ("two fully connected layers") |
| 마스크 초기화 | 블록 대각 0.1/1.0 또는 `normal_(0.9, 0.7)` | 명시 없음 |
| 최종 임베딩 L2 정규화 | 없음 | 명시 없음 |

은닉 폭 5는 위반은 아니지만 상당히 빡빡합니다. `Linear(20→5) → ReLU → Linear(5→5)` 구조라 어텐션 테이블이 5차원 병목을 통과해야 합니다.

#### 결론

**`forward` 자체는 논문 구현으로 인정할 수 있습니다.** 수식이 정확하고, 마스크를 512차원이 아닌 64차원 임베딩에 거는 판단도 맞습니다.

문제는 **이 함수가 호출되기 전에 죽는다**는 것입니다. `concat_categories` 가 data_loader 에서 10차원으로 만들어져 들어오는데 `cate_net` 첫 층은 20차원을 기대합니다 (→ Q3 의 A-1 ②). 즉 이 잘 짜인 forward 는 **한 번도 실행된 적이 없습니다.** README 의 "the code is not tested yet" 이 그대로 드러나는 대목입니다.

---

### Q5. 인덱스 저장 비용이 정확히 몇 배로 늘어나나?

*카테고리 쌍마다 유효 마스크가 다르므로, 아이템 하나가 임베딩 하나로 끝나지 않습니다.*

Polyvore Outfits 의 상위 카테고리(semantic category)는 **11개**입니다. DB 아이템 하나를 색인할 때, 그 아이템이 나중에 어떤 카테고리의 질의로부터 검색될지 모르므로 **가능한 소스 카테고리를 전부 열거해** 임베딩을 만들어 둬야 합니다.

| 항목 | 수 |
|---|---|
| 아이템 하나당 임베딩 개수 | 최대 11개 (자기 카테고리 제외하면 10개) |
| 임베딩 차원 | 64 |
| 아이템당 저장량 | 64 × 11 × 4바이트 ≈ **2.8KB** (float32 기준) |

논문 표현 그대로 "embedding size is linear in the number of high-level categories"입니다. 카테고리가 세분화될수록 이 비용이 커지므로, **fine-grained 153개** 수준으로 조건을 세밀하게 바꾸려 하면 곧바로 벽에 부딪힙니다. 이 논문이 굳이 상위 11개 카테고리만 조건으로 쓴 이유가 여기 있다고 보는 게 자연스럽습니다.

### Q6. recall@10 이 8.27% 면 좋은 건가 나쁜 건가?

*절대 수치가 낮아 보여 판단이 어려우므로 기준선과 비교합니다.*

| 기준 | 계산 | 값 |
|---|---|---|
| 무작위 추측 | 10 / 3000 | **0.33%** |
| CSA-Net (non-D top10) | — | **8.27%** |
| 무작위 대비 | 8.27 / 0.33 | **약 25배** |
| SCE-Net 대비 상대 개선 | (8.27 − 5.10) / 5.10 | **+62%** |

즉 절대 수치는 낮지만 무의미한 수준이 아닙니다. 다만 이 지표가 낮게 나오는 큰 이유는 **정답이 1장뿐**이기 때문이며(7장 (4) 참조), 실제 사용자 만족도와는 다른 이야기입니다. 논문 스스로 "카탈로그에 어울리는 아이템이 여럿 있을 수 있으므로 이 지표가 시스템의 실용 성능을 반영하지 못할 수 있다"고 명시했습니다.

### Q7. 왜 subspace 를 하필 5개만 쓰나?

*표현력을 늘리려면 많을수록 좋을 것 같은데 5개인 이유가 궁금해집니다.*

논문은 **자체 ablation 을 하지 않았고**, SCE-Net [15]이 같은 Polyvore Outfits 에서 수행한 subspace 개수 실험을 인용하며 그대로 5를 채택했습니다.

다만 2.2절의 관점에서 보면 5라는 숫자의 의미가 분명해집니다. **66개 유효 마스크를 5차원 기저로 표현**한다는 뜻이므로, 5는 "표현력 예산"이 아니라 **압축률 파라미터**입니다. 5를 66까지 키우면 Type-aware [16]와 동일해지고 정규화 효과가 사라집니다. 즉 **작게 유지하는 것이 이 방법의 핵심**입니다. 논문이 이 관점을 명시하지 않은 것은 아쉬운 부분입니다.

### Q8. 실서비스에 그대로 쓸 수 있나?

*이 논문의 목적이 실서비스이므로 직접 답해 둘 가치가 있습니다.*

| 조건 | 판정 |
|---|---|
| 인덱싱 가능성 | ✅ 설계 목적 그대로 달성 |
| 신상품(cold-start) 대응 | ✅ 이미지 + 카테고리만 있으면 임베딩이 나옴 (GCN 대비 결정적 우위) |
| 카탈로그 규모 | ⚠️ 미검증 — 3000장 실험이 전부, 지연시간 측정 0건 |
| 저장 비용 | ⚠️ 카테고리 수 배수 (Q5) |
| 재현 | ❌ 코드·데이터 분할 미공개. 비공식 구현 2종도 실행 불가 → **직접 구현 필요** (→ Q3 의 D) |
| 카테고리 체계 의존 | ⚠️ 조건이 **카테고리 one-hot** 이므로, 카테고리 체계를 바꾸면 attention 테이블을 재학습해야 함 |

구조 자체는 단순해서(ResNet18 + 마스크 5개 + 2층 MLP) 재구현 난이도는 낮습니다. 다만 논문의 검색 수치를 그대로 재현하려는 목표는 포기하는 편이 낫습니다.

---

## 🎯 한 줄 요약 (전체)

**설계 판단은 옳고 실험 서사는 과장되어 있습니다.**

| 항목 | 평가 |
|---|---|
| 기여 #1 (CSA-Net) | ✅ 실질적. 다만 "attention"이 아니라 **마스크 테이블의 low-rank 압축**이라고 부르는 게 정확 |
| 기여 #2 (outfit ranking loss) | ⚠️ Table 2·3 을 겹쳐 보면 **hard negative mining 으로 환원**됨. average 집계일 때 triplet 보다 낮다는 사실이 결정적 |
| "대규모 검색" 주장 | ❌ 효율 측정치가 하나도 없어 **미검증** |
| 읽을 가치 | ✅ 충분. **"모델이 무엇에 조건부여되는가"가 곧 "인덱싱이 가능한가"를 결정한다**는 교훈은 추천·검색 시스템 설계 전반에 그대로 적용됨 |

---

## 🔗 관련 메모리

- [[paper-colpali]] — 마찬가지로 late interaction / 인덱싱 비용 트레이드오프를 다룬 검색 논문
- [[paper-lpips]] — 딥 feature 거리를 지각 지표로 쓰는 계보 (거리 학습 관점)
