# Safe RAG — 한국어 금융 임베딩 역전공격·방어 연구

한국어 금융 RAG 시스템에서 **임베딩 벡터 역전공격(Vec2Text)** 의 위협을 재현하고, 다양한 방어 기법의 **프라이버시–유용성 트레이드오프를 다각도로 측정**하는 프로젝트입니다.

> RAG 벡터 DB에 저장된 임베딩이 탈취되면 Vec2Text로 원문(특히 PII)이 복원될 수 있습니다. 본 프로젝트는 **한국어·금융·규제 도메인**에서 이 위협과 방어를 정량 분석합니다.

---

## 사용 모델 · 라이선스

| 아티팩트 | 역할 | 차원/백본 | 라이선스 |
|---|---|---|---|
| [`dragonkue/BGE-m3-ko`](https://huggingface.co/dragonkue/BGE-m3-ko) | 문장 임베딩 | 1024d, CLS+정규화 | Apache-2.0 |
| [`terriapurplewave/bge-m3-ko-inversion`](https://huggingface.co/terriapurplewave/bge-m3-ko-inversion) | 임베딩→1차 복원 | mt5-base | Apache-2.0 |
| [`terriapurplewave/bge-m3-ko-corrector`](https://huggingface.co/terriapurplewave/bge-m3-ko-corrector) | 반복 교정(20-step) | mt5-base | Apache-2.0 |

- **인용**: Morris et al., *Text Embeddings Reveal (Almost) As Much As Text* (EMNLP 2023, vec2text) / Chen et al., *BGE M3-Embedding* (2024) / Xue et al., *mT5* (NAACL 2021).
- ⚠️ BGE-m3-ko vec2text는 **최대 64토큰**까지만 임베딩·복원합니다. 예시 문서는 64토큰 이내로 생성합니다.

---

## 메인 노트북: `_colab_Safe_RAG_Generate_and_judgement.ipynb`

Colab/로컬(CUDA) 공용. **위→아래 순서 실행**. 구성:

| 단계 | 내용 |
|---|---|
| **SETUP A/B/C** | 설치 · 임포트/경로 · 모델 로드(임베더 `bge_ko` 등록, corrector `embedder_dim` 1024 보정) + 헬퍼(`bge_embed`, `attack_invert`, `run_attack`) |
| **측정 도구모음** | 한국어 ROUGE·PII 복원율·의미누출·랭킹·이웃보존·부트스트랩 CI + 중앙 `evaluate_defense()` |
| **Step 1** | 한국어 금융 PII 데이터 생성(≤64토큰) → BGE-m3-ko 임베딩 → FAISS |
| **Step 2–3** | 원본·정규화 벡터 역전공격 (기준선) |
| **Step 4** | PCA(4-A/B) · **가우시안 노이즈(4-C)** · **비밀 직교회전+양자화(4-D)** · **INLP/LEACE 소거 W(4-E)** |
| **Step 5** | PII-aware 억제 + PCA 복합, **12개 방어 다각도 종합비교(5-4)** |
| **Step 6** | 최적 PCA n_components 탐색 |
| **Step 7** | Defense-in-Depth 재포지셔닝(L1 마스킹·L2 DP·L3~L5 계층 서술) |
| **Step 8** | 온라인 역전공격 탐지·차단 (Stateful Detection) |

---

## 데이터셋 (합성)

- 생성기: `generate_korean_pii_dataset.py` — **gpt-4o-mini**(자연스러운 짧은 문장) + **Faker `ko-KR`**(한국어 이름), 실패 시 템플릿 폴백.
- 규격: 문서당 **PII 4개**(이름·계좌번호·전화 + 이메일/IBAN/SSN 중 1개), 평균 **~43토큰(≤64)**.
- 각 문서에 `pii`(타입→값), `pii_spans`, `redacted_text` 기록 → PII 지표가 정확히 참조.
- 출력: `output/kor_financial_pii_dataset_64tok.jsonl`

---

## 방어 기법

| 방어 | 원리 | 위협모델 유효범위 |
|---|---|---|
| L2 정규화 | 크기 제거 | (효과 거의 없음 — BGE는 이미 단위벡터) |
| PCA 재구성 | 저분산 주성분 제거 | 비적응형·약함 |
| PII-aware 소프트 억제 | PII 신호 차원 감쇠 | 비적응형 |
| **가우시안 노이즈(λ)** | 임베딩에 노이즈(질의·인덱스 독립) | 비적응형, DP 보장 가능 |
| **비밀 직교회전** | 비밀 Q로 회전(코사인 보존) | 비적응형(검색 무손실) |
| **양자화(8-bit / PQ)** | 정밀도 축소 | 비적응형, 인덱스 축소 |
| **INLP / LEACE 소거 W** | redacted 대비로 PII 부분공간 선형 제거 | 선형 누출 제거(비선형 잔여는 실측) |
| 온라인 탐지(Step 8) | 근접·수렴 질의열 탐지→잠금 | **온라인** 공격 한정 |

> 이들은 **암호화·접근통제(L4)** 위에 얹는 **완화 계층(L3~L5)** 이며, 적응형 공격엔 보장이 없습니다.

---

## 측정 지표 (다각도 + 부트스트랩 95% CI)

- **프라이버시(공격)**: 한국어 ROUGE-1(어절/음절) · **PII exact-match/퍼지 복원율**(이름 편집거리·계좌 부분숫자) · **worst-case(p95/max)** · **의미 누출**(복원문 재임베딩 코사인)
- **유용성(검색)**: Recall@1/5/10 · MRR · nDCG@10(질의=redacted→원문) · **이웃 보존도**(Jaccard@k·Spearman)
- ⚠️ 기본 `rouge_score`는 비-ASCII(한국어)를 삭제하므로 **어절/음절 토크나이저로 교체** 필수.

---

## 주요 결과 (요약)

**✅ 확실(실측)**
- 파이프라인 정합성: 위키형 예문 **exact match 5/5 (100%)**.
- 한국어 금융 PII 복원: beam=1(n=300) 음절 ROUGE **0.342**, **beam=4(n=25) 0.405**. 어절 ROUGE≈0 → **구조·의미는 부분 복원, 정확 단어/PII는 못 맞춤**.
- PII 단위 누출: **한글 이름은 잘 복원, 숫자형(계좌·주민번호)은 거의 안 새는** 편중.

**⚠️ 예비(소표본 방향성)**
- **비밀 직교회전**: 검색(Recall·nDCG·이웃)을 **정확히 보존**하며 PII·의미 누출을 0 수준으로 붕괴(비적응형 기준).
- **INLP/LEACE 소거 W**: PII 제거 + 검색을 **오히려 개선**(redacted 질의와 정렬).
- **PCA**: 방어율이 약함.

> 정식 수치는 300문서·20-step·적응형 공격 평가로 확정 예정. 자세한 내용은 `실험_임시레포트.md` 참고.

---

## 환경 설정

```bash
python -m venv safeRag
safeRag\Scripts\activate            # (Windows)
pip install -r colab_requirements.txt
```

주요 의존: `vec2text==0.0.13`, `transformers==4.44.2`, `sentence-transformers`, `sentencepiece`, `protobuf`, `faiss-cpu`, `rouge-score`, `faker`, `scikit-learn`, `matplotlib`.

`.env` (데이터 **생성**에만 필요, 임베딩·공격은 로컬 GPU로 동작):
```
OPENAI_API_KEY=sk-...
```

**GPU**: 로컬 8GB(RTX 4060) → `ATTACK_BEAM=1` / Colab T4(16GB)↑ → `ATTACK_BEAM=4`. `ATTACK_N_SAMPLES`로 공격 표본 수 조절(작을수록 빠름).

---

## 주요 파일

```
Safe_RAG/
├── _colab_Safe_RAG_Generate_and_judgement.ipynb  # 메인 실험 노트북 (Step 1~8)
├── generate_korean_pii_dataset.py                # 한국어 PII 데이터 생성기 (LLM+Faker)
├── output/kor_financial_pii_dataset_64tok.jsonl  # 합성 데이터셋 (≤64토큰, PII 4개/문서)
├── embeddings_unsafe_raw.npy                      # 원본 임베딩 (N, 1024)
├── faiss_unsafe.index                            # FAISS 인덱스
├── W_pii_erase_{subspace,leace}.npy              # PII 소거 W (연구 산출물)
├── 실험_임시레포트.md                             # 실험 결과 임시 레포트
├── colab_requirements.txt                        # 의존 패키지
└── .env                                          # OpenAI 키 (git 제외, 생성 전용)
```

---

## 참고
- 상세 결과·한계·향후 계획: **`실험_임시레포트.md`**
- 방어 대안 조사: 임베딩 변환/양자화(SIGIR-AP 2024), 선형 개념 소거 INLP(ACL 2020)·LEACE(NeurIPS 2023), 온라인 탐지 Blacklight(USENIX 2022)·PRADA(EuroS&P 2019).
