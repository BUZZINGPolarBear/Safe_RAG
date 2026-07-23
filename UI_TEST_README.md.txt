🛡️ 금융 RAG 시스템 이중 보안 가드레일 (Security Guardrails for Financial RAG)
본 리포지토리는 금융보안원 '망분리 규제 완화' 환경에 맞춰, 사내 생성형 AI(RAG) 도입 시 발생할 수 있는 보안 위협과 컴플라이언스 리스크를 원천 차단하기 위한 다중 계층 방어(Defense in Depth) 샌드박스 시스템입니다.
🌟 프로젝트 개요 (Overview)
생성형 AI 시스템으로 유입되는 악의적인 프롬프트를 차단하는 입구(Inbound) 방어와, AI가 생성한 환각 및 기밀 데이터를 내보내기 전 통제하는 출구(Outbound) 방어를 통합 구현했습니다.
🔐 핵심 보안 아키텍처 (Core Architecture)
* [Step 1] Inbound 가드레일 (inbound.py)
   * 신용정보법 준수: 주민등록번호, 계좌번호 등 PII(개인식별정보) 마스킹 및 유출 차단
   * 우회 공격 무력화: 제로폭 문자(Zero-width), 특수문자 등 텍스트 정규화
   * 자원 보호: Rate Limit을 통한 도배 공격 및 시스템 과부하 방지 (Fail-Closed)
* [Step 2] Vector DB 보안 확인 (시뮬레이션)
   * 역전 공격(Vec2Text) 방어를 위한 임베딩 복합 방어율(PCA 축소 + L2 정규화) 검증
* [Step 3] Outbound 가드레일 (outbound.py)
   * 자본시장법 준수: M&A 등 미공개 중요 정보(MNPI) 카나리아 토큰(Canary Token) 유출 감지
   * 환각(Hallucination) 제어: 결정론적 숫자 그라운딩을 통한 가짜 금리/수익률 차단
* [Audit] 무결성 감사 로그 시스템
   * 해시체인(Hash Chain) 기반으로 위변조가 불가능한 판정 이력 실시간 저장
🚀 빠른 실행 가이드 (Quick Start)
로컬 환경에서 Streamlit 기반의 통합 시연 UI(app.py)를 실행하여 방어 시나리오를 직접 테스트할 수 있습니다.
1. 환경 준비 (Prerequisites)
Python 3.9 이상의 환경이 필요합니다. 리포지토리를 클론한 후 필수 패키지를 설치합니다.
git clone [Repository URL]
cd [Repository Name]

# 가상환경 생성 및 활성화 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install streamlit openai python-dotenv

2. API 키 설정 (Environment Variables)
프로젝트 루트 디렉토리에 .env 파일을 생성하고 실제 OpenAI API 키를 입력합니다.
(API 키가 없어도 일부 시나리오는 Fallback Mock 모드로 동작합니다.)
OPENAI_API_KEY=sk-your-openai-api-key-here

3. 애플리케이션 실행 (Run)
아래 명령어를 통해 웹 UI를 실행합니다.
streamlit run app.py

🎯 데모 시나리오 테스트 방법 (How to Test)
앱이 실행되면 좌측 'Guardrail Control Center (사이드바)'에서 원클릭 시나리오 버튼을 눌러 시스템의 반응을 테스트하세요.
1. 🚨 시나리오 1 (PII/도배 공격)
   * 행동: 사용자가 주민등록번호가 포함된 질의를 전송합니다.
   * 결과: [Inbound 차단] LLM에 도달하기 전 선제적으로 즉시 차단되며 파이프라인이 조기 종료됩니다.
2. ✅ 시나리오 2 (정상 질의)
   * 행동: 일반적인 대출 금리 가이드라인을 질문합니다.
   * 결과: [정상 통과] Inbound와 Outbound를 모두 통과하여 안전한 답변이 스트리밍(Streaming)으로 출력됩니다.
3. 🔓 시나리오 3 (대외비/환각 유도)
   * 행동: 교묘하게 정상적인 질문으로 포장하여 AI의 대외비 발설을 유도합니다.
   * 결과: [Outbound 차단] Inbound는 통과했으나, LLM이 생성한 답변에 포함된 카나리아 토큰(기밀)을 Outbound 엔진이 낚아채어 최종 차단합니다. 가드레일이 없었다면 노출됐을 원본 응답도 함께 비교하여 확인할 수 있습니다.
🔍 추가 테스트 팁
* 자유 질의 입력: 정해진 버튼 외에도 중앙의 텍스트 박스에 직접 악의적인 질문을 타이핑하여 가드레일이 어떻게 방어하는지 테스트해 보세요.
* 감사 로그 무결성 검사: 사이드바 하단의 [🔐 감사로그 무결성 검증 실행] 버튼을 눌러, 생성된 demo_inbound_logs.json 파일의 해시체인이 훼손되지 않았는지 실시간으로 검증할 수 있습니다.
📂 파일 구조 (File Structure)
├── app.py                     # Streamlit 기반 통합 데모 UI (Executive Demo)
├── inbound.py                 # 1차 방어선: 입력 프롬프트 검열 로직
├── outbound.py                # 2차 방어선: 출력 답변 검열 로직 (카나리아 토큰 등)
├── demo_inbound_logs.json     # (자동 생성) Inbound 판정 감사 로그 
├── demo_outbound_logs.json    # (자동 생성) Outbound 판정 감사 로그
├── .env                       # 환경변수 파일 (API Key 등 세팅 - Git 제외)
└── README.md                  # 본 프로젝트 설명서

Maintainers & Authors: AI Security Architecture Team
Last Updated: 2026.07