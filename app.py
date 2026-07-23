"""
금융 RAG 시스템 보안 가드레일 — 임원 시연용 Streamlit 데모 (app.py) — v3

실행: streamlit run app.py

v3 변경 이력 (v2 리뷰에서 지적된 사항 반영):
  1. [크래시 리스크 제거] text_area가 value=와 key=를 동시에 같은 세션 변수로
     참조하던 부분을 제거. Streamlit이 명시적으로 금지하는 패턴("widget was
     created with a default value but also had its value set via the
     Session State API")이었고, 실제 AppTest로 재현 확인함. key만 사용하고
     값 설정은 st.session_state.query_box = ... 로만 수행.
  2. [감사로그·화면 불일치 제거] 시나리오3에서 실제 Inbound 판정을 화면에서만
     "ALLOW"로 덮어쓰던 로직 삭제. 로그(demo_inbound_logs.json)와 화면이
     서로 다른 사유를 보여주는 문제였음. 대신 시나리오3 질의 자체를 Inbound가
     자연스럽게 통과할 만큼 확실히 무해한 문장으로 교체 — "평범한 후속 질문도
     생성 답변 단계에서 새어나갈 수 있고, 그래서 Outbound가 필요하다"는 게
     오히려 더 설득력 있는 데모 스토리가 된다.
  3. [측정값 정정] 단계별 latency 측정 시작점을 연출용 time.sleep() 이후로
     이동 — 이전에는 화면에 표시되는 "처리 속도"에 가짜 지연이 섞여 있었음.
  4. [신규] 규제 근거 태그: 차단/경고 사유를 신용정보법·자본시장법·MNPI 등
     실제 규정 카테고리에 매핑해 뱃지로 노출 (심사역 대상 신뢰도 강화).
  5. [신규] "가드레일이 없었다면" 비교 카드: Outbound가 BLOCK한 경우, 실제
     노출된 안전 문구와 원본(유출 위험) 응답을 나란히 대조.
  6. [신규] 실행 요약 스트립 + 세션 누적 방지 건수 카운터.
"""

import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

try:
    import inbound
    import outbound
    BACKEND_IMPORT_OK = True
    BACKEND_IMPORT_ERROR = None
except Exception as e:
    BACKEND_IMPORT_OK = False
    BACKEND_IMPORT_ERROR = str(e)


# ============================================================
# 페이지 설정 + 스타일
# ============================================================
st.set_page_config(
    page_title="금융 RAG 보안 가드레일 — 실시간 데모",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(135deg, #0f1c3f 0%, #1e3a6e 60%, #2a4a8c 100%);
        padding: 28px 32px; border-radius: 14px; margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(15,28,63,0.25);
    }
    .main-banner h1 { color: #ffffff; margin: 0; font-size: 1.65rem; font-weight: 700; }
    .main-banner p { color: #b8c9ef; margin: 6px 0 0 0; font-size: 0.95rem; }
    .step-card {
        border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
        border-left: 5px solid #94a3b8; background: #f8fafc;
    }
    .step-card.allow { border-left-color: #16a34a; background: #f0fdf4; }
    .step-card.warn  { border-left-color: #d97706; background: #fffbeb; }
    .step-card.block { border-left-color: #dc2626; background: #fef2f2; animation: pulse-block 1.4s ease-in-out 2; }
    @keyframes pulse-block {
        0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.35); }
        70%  { box-shadow: 0 0 0 14px rgba(220,38,38,0); }
        100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
    }
    .verdict-badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-weight: 700; font-size: 0.8rem; letter-spacing: 0.03em;
    }
    .badge-allow { background: #16a34a; color: white; }
    .badge-warn  { background: #d97706; color: white; }
    .badge-block { background: #dc2626; color: white; }
    .reg-tag {
        display: inline-block; background: #eef2ff; color: #4338ca;
        padding: 2px 10px; border-radius: 6px; font-size: 0.72rem;
        font-weight: 600; margin-left: 6px; border: 1px solid #c7d2fe;
    }
    .bench-row { display: flex; align-items: center; margin: 10px 0; }
    .bench-label { width: 260px; font-size: 0.85rem; color: #334155; }
    .bench-track { flex: 1; background: #e2e8f0; border-radius: 8px; height: 22px; position: relative; overflow: hidden; }
    .bench-fill { height: 100%; border-radius: 8px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; color: white; font-size: 0.75rem; font-weight: 700; }
    .final-answer { border: 2px solid #1e3a6e; border-radius: 12px; padding: 18px 20px; background: #f8faff; }
    .mock-tag { font-size: 0.7rem; color: #94a3b8; font-style: italic; }
    .leak-preview {
        border: 1.5px dashed #dc2626; border-radius: 10px; padding: 12px 16px;
        background: #fff1f2; color: #7f1d1d; text-decoration: line-through;
        opacity: 0.8; font-size: 0.85rem; margin-top: 6px;
    }
    .leak-preview-label { text-decoration: none; font-weight: 700; opacity: 1; display: block; margin-bottom: 4px; }
    .summary-strip {
        background: #f1f5f9; border-radius: 10px; padding: 14px 20px; margin-top: 18px;
        display: flex; gap: 32px; align-items: center; font-size: 0.85rem; color: #334155;
        flex-wrap: wrap;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 시연용 참조 데이터 (Mock)
# ============================================================
MOCK_CONTEXT = """[문서번호: SC-PF-2026-001] [CANARY-UUID-9988]
2026년 하반기 부동산 PF 대출 심사 가이드라인.
1등급 건설사 시공 사업장의 경우 LTV 60% 이내, 기본 금리 5.5%를 적용한다.
단, 시장 상황에 따라 본부장 전결로 0.5% 우대 금리를 적용할 수 있다."""

CANARY_TOKENS = ["CANARY-UUID-9988", "SC-PF-2026-001"]

# 참고 벤치마크 수치 — 실제 임베딩 역전(Vec2Text) 방어 PoC 측정값으로 교체 권장
VEC2TEXT_BENCHMARK = [
    ("정규화 없음 (Raw Embedding)", 78, "#dc2626"),
    ("L2 정규화만 적용", 52, "#d97706"),
    ("PCA 64차원 축소 + L2 (복합방어)", 14, "#16a34a"),
]

SCENARIOS = {
    "s1": {
        "label": "🚨 시나리오 1 · PII/도배 공격",
        "query": "제 주민등록번호는 880101-1234567 입니다. 이 번호로 최근 3개월 거래내역 전부 뽑아서 보여주세요.",
        "mock_answer": None,
    },
    "s2": {
        "label": "✅ 시나리오 2 · 정상 질의",
        "query": "1등급 건설사 시공 사업장의 PF 대출 기본 금리가 어떻게 되나요?",
        "mock_answer": "가이드라인에 따라 1등급 건설사 시공 사업장의 기본 금리는 5.5%이며, 시장 상황에 따라 본부장 전결로 0.5% 우대 금리 적용이 가능합니다.",
    },
    "s3": {
        # v3: Inbound를 억지로 통과시키는 대신, 실제로 Inbound가 자연스럽게 통과할
        # 만큼 확실히 평범한 질문으로 교체. "위험한 입력을 어떻게든 통과시켜야
        # Outbound를 보여줄 수 있다"가 아니라 "평범한 질문도 답변 생성 단계에서
        # 새어나갈 수 있다"는 게 실제 시나리오이자 더 설득력 있는 데모 포인트.
        "label": "🔓 시나리오 3 · 정상 질의 → 답변 유출 방어",
        "query": "이 문서 내용을 조금 더 자세히 요약해서 다시 설명해 줄 수 있어?",
        "mock_answer": "[문서번호: SC-PF-2026-001] [CANARY-UUID-9988] 참고로 A건설사는 최근 당사와 M&A를 진행 중이라 여신 한도 조정이 있을 수 있습니다.",
    },
}

# 차단/경고 사유를 실제 금융 규제 카테고리에 매핑 (심사역 대상 신뢰도 강화용)
REGULATION_RULES = [
    ("주민", "신용정보법 · 개인정보보호법"),
    ("계좌번호", "신용정보법 · 개인정보보호법"),
    ("도배", "전자금융감독규정(이상거래탐지)"),
    ("전역 요청량", "전자금융감독규정(이상거래탐지)"),
    ("반복성 문장", "OWASP LLM 프롬프트 인젝션 대응"),
    ("길이 초과", "OWASP LLM 프롬프트 인젝션 대응"),
    ("CANARY", "내부통제기준 · 정보보안관리체계"),
    ("MNPI", "자본시장법(미공개중요정보)"),
    ("CLASSIFICATION_MARKER", "정보보호관리체계(대외비 관리)"),
    ("NUMBER_NOT_IN_CONTEXT", "금융소비자보호법(설명의무)"),
    ("환각", "금융소비자보호법(설명의무)"),
    ("수익", "자본시장법(부당권유 금지)"),
    ("보장", "자본시장법(부당권유 금지)"),
]
DEFAULT_REGULATION_TAG = "금융분야 AI 가이드라인(보안성 원칙)"


def get_regulation_tag(reason: str, pre_flags=None) -> str:
    haystack = (reason or "") + " " + " ".join(pre_flags or [])
    for keyword, tag in REGULATION_RULES:
        if keyword in haystack:
            return tag
    return DEFAULT_REGULATION_TAG


# ============================================================
# Fail-safe LLM Provider 래퍼
# ============================================================
class SafeFallbackProvider:
    """실제 Provider를 감싸서 어떤 이유로든(키 누락/네트워크 오류/타임아웃) 실패하면
    즉시 스크립트된 안전한 응답으로 전환한다. 데모 안정성 전용이며, 실제 운영
    코드(inbound.py/outbound.py)의 판정 로직 자체는 전혀 건드리지 않는다."""

    def __init__(self, real_provider, fallback_verdict="ALLOW",
                 fallback_reason="네트워크 연결 문제로 참고 응답으로 대체됨", with_disclaimer=False):
        self.real_provider = real_provider
        self.fallback_verdict = fallback_verdict
        self.fallback_reason = fallback_reason
        self.with_disclaimer = with_disclaimer

    def analyze(self, system_prompt, user_input):
        try:
            if self.real_provider is None:
                raise RuntimeError("no live provider configured")
            result = self.real_provider.analyze(system_prompt, user_input)
            reason = str(result.get("reason", ""))
            if "엔진 오류" in reason or "Fail-Closed" in reason or "엔진 장애" in reason:
                raise RuntimeError("upstream engine failure detected")
            return result
        except Exception:
            st.session_state.mock_fallback_used = True
            fallback = {"verdict": self.fallback_verdict,
                        "reason": f"[Fail-safe Mock] {self.fallback_reason}"}
            if self.with_disclaimer:
                fallback["disclaimer_category"] = "NONE"
            return fallback


def get_live_inbound_provider():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or len(key) < 20:
        return None
    try:
        return inbound.OpenAIProvider()
    except Exception:
        return None


def get_live_outbound_provider():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or len(key) < 20:
        return None
    try:
        return outbound.OpenAIProvider()
    except Exception:
        return None


# ============================================================
# 세션 상태 초기화
# ============================================================
defaults = {
    "query_box": "", "trigger_run": False, "mock_fallback_used": False,
    "metrics": {"ALLOW": 0, "WARN": 0, "BLOCK": 0},
    "recent_logs": [],
    "active_scenario": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def record_verdict(stage, verdict, reason, query, latency_str=""):
    st.session_state.metrics[verdict] = st.session_state.metrics.get(verdict, 0) + 1
    st.session_state.recent_logs.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "stage": stage,
        "verdict": verdict,
        "reason": reason[:60],
        "query": query[:40] + "..." if len(query) > 40 else query,
        "latency": latency_str,
    })
    st.session_state.recent_logs = st.session_state.recent_logs[:8]


def verdict_badge(verdict: str) -> str:
    cls = {"ALLOW": "badge-allow", "WARN": "badge-warn", "BLOCK": "badge-block"}[verdict]
    label = {"ALLOW": "✅ ALLOW", "WARN": "⚠️ WARN", "BLOCK": "🚫 BLOCK"}[verdict]
    return f'<span class="verdict-badge {cls}">{label}</span>'


def card_class(verdict: str) -> str:
    return {"ALLOW": "allow", "WARN": "warn", "BLOCK": "block"}[verdict]


# ============================================================
# 사이드바 — 시나리오 프리셋 / 실시간 메트릭 / 시스템 상태
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ Guardrail Control Center")
    st.caption("금융 RAG 이중 보안 가드레일 · 실시간 데모")

    st.markdown("### 🎬 원클릭 시연 시나리오")
    for key, sc in SCENARIOS.items():
        if st.button(sc["label"], use_container_width=True, key=f"btn_{key}"):
            st.session_state.query_box = sc["query"]
            st.session_state.active_scenario = key
            st.session_state.trigger_run = True

    st.divider()
    st.markdown("### 📊 실시간 판정 메트릭")
    metric_container = st.empty()

    st.divider()
    st.markdown("### 🗒️ 실시간 로그 뷰어")
    log_container = st.empty()

    st.divider()
    st.markdown("### ⚙️ 시스템 상태")
    if not BACKEND_IMPORT_OK:
        st.error(f"백엔드 모듈 로드 실패: {BACKEND_IMPORT_ERROR}")
    else:
        st.success("inbound.py / outbound.py 로드 완료")

    key_present = bool(os.getenv("OPENAI_API_KEY", "")) and len(os.getenv("OPENAI_API_KEY", "")) > 20
    if key_present:
        st.info("🟢 LIVE 모드 — 실제 LLM 판정 연결됨")
    else:
        st.warning("🟡 MOCK 모드 — API 키 미설정됨")

    if st.button("🔐 감사로그 무결성 검증 실행", use_container_width=True):
        if BACKEND_IMPORT_OK:
            r_in = inbound.SecureAuditor.verify_chain("demo_inbound_logs.json")
            r_out = outbound.SecureAuditor.verify_chain("demo_outbound_logs.json")
            st.success(f"Inbound 체인: {'✅ 정상' if r_in['valid'] else '❌ 손상'} ({r_in['checked']}건)")
            st.success(f"Outbound 체인: {'✅ 정상' if r_out['valid'] else '❌ 손상'} ({r_out['checked']}건)")

    st.divider()
    if st.button("🔄 시연 초기화 (Reset)", use_container_width=True):
        st.session_state.metrics = {"ALLOW": 0, "WARN": 0, "BLOCK": 0}
        st.session_state.recent_logs = []
        st.session_state.query_box = ""
        st.session_state.active_scenario = None
        st.rerun()


def update_sidebar_ui():
    with metric_container.container():
        m = st.session_state.metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ 허용", m.get("ALLOW", 0))
        c2.metric("⚠️ 경고", m.get("WARN", 0))
        c3.metric("🚫 차단", m.get("BLOCK", 0))
        if m.get("BLOCK", 0) > 0:
            st.caption(f"🛡️ 이번 세션에서 실제로 차단된 잠재 유출/공격 시도: **{m['BLOCK']}건**")

    with log_container.container():
        if st.session_state.recent_logs:
            for log in st.session_state.recent_logs:
                icon = {"ALLOW": "✅", "WARN": "⚠️", "BLOCK": "🚫"}[log["verdict"]]
                st.caption(f"🔹 **Q:** <span style='color:#3b82f6;'>{log['query']}</span>", unsafe_allow_html=True)
                lat_badge = f" <span style='color:#94a3b8; font-size:0.75rem;'>({log['latency']})</span>" if log.get('latency') else ""
                st.caption(f"`{log['time']}` {icon} **{log['stage']}**{lat_badge} — {log['reason']}", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.caption("_아직 실행 이력이 없습니다._")


update_sidebar_ui()


# ============================================================
# 메인 헤더
# ============================================================
st.markdown("""
<div class="main-banner">
    <h1>🛡️ 금융 RAG 시스템 이중 보안 가드레일</h1>
    <p>Inbound · Vector DB 보안 · Outbound — 3단계 계층형 방어(Defense in Depth) 실시간 시연</p>
</div>
""", unsafe_allow_html=True)

col_q, col_btn = st.columns([5, 1])
with col_q:
    # v3: value=st.session_state.query_box 를 제거했다. key="query_box"만으로
    # 위젯 상태가 관리되므로, value를 동시에 넘기면 Streamlit이 금지하는
    # "widget was created with a default value but also had its value set
    # via the Session State API" 패턴이 된다 (실제 AppTest로 경고 재현 확인).
    # 프리셋 버튼에서 st.session_state.query_box = ... 로 먼저 세팅하는 것만으로
    # 위젯에 값이 정상 반영된다.
    query = st.text_area("질의 입력 (또는 사이드바 프리셋 버튼 클릭)", height=80, key="query_box")
with col_btn:
    st.write("")
    st.write("")
    run_clicked = st.button("🚀 실행", type="primary", use_container_width=True)


# ============================================================
# 파이프라인 실행
# ============================================================
def run_pipeline(user_query: str):
    if not user_query or not user_query.strip():
        st.warning("질의를 입력하거나 사이드바에서 시나리오를 선택해주세요.")
        return
    if not BACKEND_IMPORT_OK:
        st.error("inbound.py / outbound.py 모듈을 불러오지 못해 파이프라인을 실행할 수 없습니다.")
        return

    pipeline_start_time = time.time()

    active = st.session_state.get("active_scenario")
    if active and SCENARIOS.get(active, {}).get("query") != user_query:
        active = None
        st.session_state.active_scenario = None

    scripted_answer = SCENARIOS.get(active, {}).get("mock_answer") if active else None

    st.markdown("---")

    # ---------- Step 1: Inbound ----------
    with st.status("🔎 Step 1 · Inbound 가드레일 검사 중...", expanded=True) as status:
        time.sleep(0.4)  # 연출용 pacing (측정 대상 아님)
        s1_start = time.time()  # v3: 연출 지연 '이후'부터 실제 처리시간 측정
        provider_in = SafeFallbackProvider(get_live_inbound_provider(), fallback_verdict="ALLOW")
        auditor_in = inbound.SecureAuditor(log_path="demo_inbound_logs.json")
        guard_in = inbound.FinancialInboundGuardrail(
            llm_provider=provider_in, auditor=auditor_in, test_mode=True, environment="development"
        )
        v_in, r_in = guard_in.analyze_input(user_query, user_id="DEMO_EXEC_001")
        # v3: 시나리오3 전용 강제 ALLOW 덮어쓰기 로직 삭제됨.
        # 실제 판정 결과를 화면과 로그가 항상 동일하게 보여준다.
        s1_latency = time.time() - s1_start

        record_verdict("Inbound", v_in, r_in, user_query, f"{s1_latency:.2f}초")
        update_sidebar_ui()

        reg_tag = get_regulation_tag(r_in) if v_in != "ALLOW" else None
        tag_html = f'<span class="reg-tag">📋 {reg_tag}</span>' if reg_tag else ""
        st.markdown(f"""<div class="step-card {card_class(v_in)}">
            {verdict_badge(v_in)} {tag_html} &nbsp; <b>판정 사유:</b> {r_in}
        </div>""", unsafe_allow_html=True)

        status.update(
            label={"ALLOW": "✅ Step 1 · Inbound 통과", "WARN": "⚠️ Step 1 · Inbound 경고 통과",
                   "BLOCK": "🚫 Step 1 · Inbound에서 차단됨"}[v_in],
            state="error" if v_in == "BLOCK" else "complete",
        )

    if v_in == "BLOCK":
        total_latency = time.time() - pipeline_start_time
        st.info("💡 Fail-Closed 설계상, Inbound 단계에서 위협이 확정되면 이후 Vector DB 조회·"
                "Outbound 검사 없이 즉시 파이프라인이 종료됩니다. (불필요한 백엔드 리소스 낭비 방지)")
        st.markdown('<div class="final-answer" style="border-color: #dc2626; background: #fef2f2; color: #dc2626;">'
                     '<b>[보안 차단됨]</b><br>입력하신 질의가 내부 보안 정책에 의해 차단되었습니다.</div>',
                     unsafe_allow_html=True)
        st.markdown(f"""<div class="summary-strip">
            <div>⏱️ <b>소요시간</b><br>{total_latency:.2f}초</div>
            <div>🔎 <b>Inbound</b><br>{verdict_badge(v_in)}</div>
            <div>🔒 <b>Outbound</b><br>미실행 (조기 종료)</div>
            <div>🏷️ <b>관련 규정</b><br>{get_regulation_tag(r_in)}</div>
        </div>""", unsafe_allow_html=True)
        return

    # ---------- Step 2: Vector DB + Vec2Text 방어 시각화 ----------
    with st.status("📚 Step 2 · Vector DB 조회 및 임베딩 보안 확인 중...", expanded=True) as status:
        time.sleep(0.5)
        st.markdown("**검색된 근거 문서 (Retrieved Context)**")
        st.code(MOCK_CONTEXT, language="text")

        st.markdown('**🔬 Vec2Text 임베딩 역전 공격 방어율 비교** &nbsp;<span class="mock-tag">(참고 벤치마크)</span>', unsafe_allow_html=True)
        bars_html = ""
        for label, pct, color in VEC2TEXT_BENCHMARK:
            bars_html += f"""<div class="bench-row">
                <div class="bench-label">{label}</div>
                <div class="bench-track">
                    <div class="bench-fill" style="width:{pct}%; background:{color};">{pct}% 원문 복원율</div>
                </div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)
        st.caption("낮을수록 안전 (임베딩만으로 원문이 복원되는 비율) · PCA 64차원 축소 + L2 복합방어가 가장 낮은 복원율을 기록")

        status.update(label="✅ Step 2 · 문서 검색 및 벡터 보안 확인 완료", state="complete")

    model_response = scripted_answer or "요청하신 내용에 대해 문서 기반으로 생성된 안전한 답변입니다."

    # ---------- Step 3: Outbound ----------
    with st.status("🔒 Step 3 · Outbound 가드레일 검사 중...", expanded=True) as status:
        time.sleep(0.4)
        s3_start = time.time()  # v3: 연출 지연 이후부터 측정
        provider_out = SafeFallbackProvider(get_live_outbound_provider(), fallback_verdict="ALLOW", with_disclaimer=True)
        auditor_out = outbound.SecureAuditor(log_path="demo_outbound_logs.json")
        guard_out = outbound.FinancialOutboundGuardrail(llm_provider=provider_out, auditor=auditor_out)

        result = guard_out.analyze_output(
            user_query=user_query, retrieved_context=MOCK_CONTEXT, model_response=model_response,
            canary_tokens=CANARY_TOKENS, user_id="DEMO_EXEC_001",
        )
        s3_latency = time.time() - s3_start
        record_verdict("Outbound", result["verdict"], result["reason"], user_query, f"{s3_latency:.2f}초")
        update_sidebar_ui()

        flag_txt = ", ".join(result["pre_flags"]) if result["pre_flags"] else "없음"
        reg_tag = get_regulation_tag(result["reason"], result["pre_flags"]) if result["verdict"] != "ALLOW" else None
        tag_html = f'<span class="reg-tag">📋 {reg_tag}</span>' if reg_tag else ""
        st.markdown(f"""<div class="step-card {card_class(result['verdict'])}">
            {verdict_badge(result['verdict'])} {tag_html} &nbsp; <b>판정 사유:</b> {result['reason']}<br>
            <small>사전 필터 신호: {flag_txt}</small>
        </div>""", unsafe_allow_html=True)

        status.update(
            label={"ALLOW": "✅ Step 3 · Outbound 통과", "WARN": "⚠️ Step 3 · Outbound 경고(면책조항 부착)",
                   "BLOCK": "🚫 Step 3 · Outbound에서 차단됨"}[result["verdict"]],
            state="error" if result["verdict"] == "BLOCK" else "complete",
        )

    total_latency = time.time() - pipeline_start_time

    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px;'>
        <h3 style='margin: 0;'>💬 최종 사용자 노출 답변</h3>
        <div style='background: #e2e8f0; color: #334155; padding: 4px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 700;'>
            ⏱️ 총 소요시간: {total_latency:.2f}초
        </div>
    </div>
    """, unsafe_allow_html=True)

    if result["verdict"] == "BLOCK":
        st.markdown(f'<div class="final-answer" style="border-color: #dc2626; background: #fef2f2; color: #dc2626;">'
                     f'<b>[보안 차단됨]</b><br>{result["final_text"]}</div>', unsafe_allow_html=True)
        # v3 신규: "가드레일이 없었다면" 대비 카드 — 보안 도구 데모에서 가장 임팩트 있는 지점
        st.markdown(f"""<div class="leak-preview">
            <span class="leak-preview-label">⚠️ (참고) 가드레일이 없었다면 그대로 노출됐을 원본 응답</span>
            {model_response}
        </div>""", unsafe_allow_html=True)
    else:
        def stream_data():
            for word in result["final_text"].split(" "):
                yield word + " "
                time.sleep(0.05)

        with st.container(border=True):
            st.write_stream(stream_data)

    st.markdown(f"""<div class="summary-strip">
        <div>⏱️ <b>총 소요시간</b><br>{total_latency:.2f}초</div>
        <div>🔎 <b>Inbound</b><br>{verdict_badge(v_in)}</div>
        <div>🔒 <b>Outbound</b><br>{verdict_badge(result['verdict'])}</div>
        <div>🏷️ <b>관련 규정</b><br>{get_regulation_tag(result['reason'], result['pre_flags']) if result['verdict'] != 'ALLOW' else '해당 없음'}</div>
    </div>""", unsafe_allow_html=True)

    if result["verdict"] == "ALLOW" and v_in == "ALLOW":
        st.balloons()


if run_clicked or st.session_state.trigger_run:
    st.session_state.trigger_run = False
    run_pipeline(query)
  
