# Safe RAG 구현

금융 문서 RAG 시스템에서 **임베딩 벡터 역전공격(Vec2Text Attack)** 에 대한 방어 기법을 연구하고 측정하는 프로젝트입니다.

---

## 프로젝트 개요

RAG(Retrieval-Augmented Generation) 시스템은 문서를 임베딩 벡터로 변환하여 벡터 DB에 저장합니다.
그러나 이 벡터가 탈취될 경우, **Vec2Text 공격**을 통해 원본 텍스트를 역복원하는 것이 가능합니다.
본 프로젝트는 PII(개인식별정보)가 포함된 금융 문서를 대상으로 공격 성능을 측정하고, 방어 기법의 효과를 정량적으로 분석합니다.

---

## 실험 흐름

### Step 1. 금융 문서 생성 및 임베딩 벡터 저장

- GPT-4o-mini로 10가지 금융 문서 유형 × 10개 변형 = **100개 영어 금융 문서** 생성
- 포함된 PII: 이름, SSN, 계좌번호, 금액, 날짜 등
- 문서 유형: 대출 심사, 사기 탐지, PB 자산관리, 의심 계좌(머니뮬), 해외 송금, 급여 이체, 담보 대출, 내부 감사, 보험 청구, 가계 부채
- `text-embedding-ada-002`로 임베딩 생성 → **343개 청크** (chunk_size=512, overlap=64)
- FAISS 인덱스 및 원본 벡터(`.npy`) 저장

| 파일 | 설명 |
|------|------|
| `embeddings_unsafe_raw.npy` | 원본 임베딩 벡터 (343, 1536) |
| `faiss_unsafe.index` | L2 정규화된 FAISS 인덱스 |

---

### Step 2. 원본 벡터 Vec2Text 공격 (베이스라인 측정)

- `vec2text.load_corrector("text-embedding-ada-002")` 사용
- 10개 샘플에 대해 역전공격 수행 (num_steps=10, beam_width=4)
- 복원 성능 측정 지표: **ROUGE-1 F1**

| 구분 | 평균 ROUGE-1 |
|------|------------|
| 원본 벡터 (베이스라인) | **0.513** |

---

### Step 3. L2 정규화 벡터 공격

- FAISS 저장 시 L2 정규화된 벡터로 공격
- 결과: 평균 ROUGE-1 **0.514** (정규화 자체는 방어 효과 없음)

---

### Step 4-A. PCA 방어

- 1536d → 128d로 압축 후 다시 1536d로 재구성 (정보 손실 이용)
- 분산 보존율: **93.1%**

| 지표 | 결과 |
|------|------|
| 평균 ROUGE-1 | 0.441 |
| 공격 감소율 | **14.1%** |
| Recall@5 (RAG 정확성) | **100%** |

---

### Step 4-B. PCA+PII 복합 방어

1. **PII Guard**: PII 포함 청크 200개 식별 → PII 집중 차원 200개를 0으로 억제
2. **PCA 재구성**: 억제된 벡터에 추가로 128 차원 PCA 적용 (분산 보존 94.4%)

| 지표 | 결과 |
|------|------|
| 평균 ROUGE-1 | 0.246 |
| 공격 감소율 | **52.1%** |
| Recall@5 (RAG 정확성) | **100%** |

---

### Step 5. PCA n_components 하이퍼파라미터 탐색

- 후보: `[512, 384, 256, 192, 128, 96, 64, 32]`
- 각 n에 대해 Vec2Text 공격 성능 + Recall@5 동시 측정
- 결과 저장: `results.csv`, 시각화: `pca_tradeoff.png`

---

## 최종 결과 요약

| 방어 방식 | 평균 ROUGE-1 | 공격 감소율 | Recall@5 |
|-----------|-------------|------------|---------|
| 원본 (무방어) | 0.513 | - | 100% |
| L2 정규화 | 0.514 | 0% | 100% |
| PCA (128d) | 0.441 | 14.1% | 100% |
| **PCA + PII Guard** | **0.246** | **52.1%** | **100%** |

> RAG 검색 정확성(Recall@5) 100% 유지하면서 공격 성공률을 52.1% 감소시키는 데 성공

---

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` 파일 생성:
```
OPENAI_API_KEY=sk-...
```

---

## 주요 파일

```
Safe RAG 구현/
├── _m4_Safe_RAG_Generate_and_judgement.ipynb  # 메인 실험 노트북
├── docs_english/                               # 생성된 영어 금융 문서 100개
├── embeddings_unsafe_raw.npy                  # 원본 임베딩 벡터
├── faiss_unsafe.index                         # FAISS 인덱스
├── requirements.txt                           # 의존 패키지
└── .env                                       # API 키 (git 제외)
```

---

## 사용 라이브러리

- `openai` — 임베딩 생성 및 문서 생성 (GPT-4o-mini)
- `vec2text` — 임베딩 역전공격
- `faiss` — 벡터 검색 인덱스
- `langchain` — 문서 로딩 및 청킹
- `scikit-learn` — PCA 방어
- `rouge-score` — 복원 성능 측정
