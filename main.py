import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="스포츠 베팅 이탈 예측",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS (라이트 모드 · 파스텔) ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Noto Sans KR', 'Inter', sans-serif !important;
    background-color: #F5F7FA !important;
    color: #1E293B !important;
}
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }
.stApp, .stApp p, .stApp div, .stApp span, .stApp label { color: #1E293B !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #475569 !important; }

/* ── KPI Card ── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-bottom: 0.55rem;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.73rem;
    color: #94A3B8;
    margin-top: 0.35rem;
}

/* ── Section header ── */
.sec-hdr {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #94A3B8;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 0.4rem;
}

/* ── Page title ── */
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1E293B;
    letter-spacing: -0.02em;
}
.page-desc {
    font-size: 0.85rem;
    color: #94A3B8;
    margin-top: 0.2rem;
}

/* ── Outcome badge ── */
.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

/* ── Strategy card ── */
.strat {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.strat-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 0.3rem;
}
.strat-body {
    font-size: 0.81rem;
    color: #64748B;
    line-height: 1.6;
}

/* ── Sidebar label ── */
.sb-label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    color: #7C9BF0 !important;
}

/* ── Button ── */
div.stButton > button {
    background: linear-gradient(135deg, #A78BFA, #818CF8);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-size: 0.88rem;
    font-weight: 600;
    width: 100%;
    font-family: 'Noto Sans KR', sans-serif;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }

/* ── Live dot ── */
.ldot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #86EFAC;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 5px #86EFAC;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Divider ── */
.hdiv { border: none; border-top: 1px solid #E2E8F0; margin: 1.4rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Palette (파스텔) ─────────────────────────────────────────
C_CHURN = "#F87171"   # 파스텔 레드
C_STAY  = "#34D399"   # 파스텔 그린
C_MID   = "#FBBF24"   # 파스텔 앰버
C_BLUE  = "#60A5FA"   # 파스텔 블루
C_PURP  = "#A78BFA"   # 파스텔 퍼플
C_ORG   = "#FDBA74"   # 파스텔 오렌지
CARD_BG = "#FFFFFF"
BORDER  = "#E2E8F0"

# ─── Age group mapping ───────────────────────────────────────
AGE_LABELS = {
    0: "10대 이하", 1: "20대", 2: "30대", 3: "40대", 4: "50대",
    5: "60대",      6: "70대", 7: "80대", 8: "90대 이상",
}
AGE_REV = {v: k for k, v in AGE_LABELS.items()}

# ─── Feature metadata (XGBoost 20 features) ──────────────────
FEAT_NAMES = [
    "gender", "age_group",
    "fixed_bet_amount", "live_bet_amount",
    "fixed_bet_cnt", "live_bet_cnt", "total_bet_cnt",
    "fixed_active_days", "live_active_days",
    "fixed_hit_days", "total_hit_days",
    "fixed_win_rate", "live_win_rate", "total_win_rate",
    "fixed_avg_roi", "live_avg_roi", "total_avg_roi",
    "days_since_reg", "days_to_first_deposit", "days_to_first_bet",
]

FEAT_KR = {
    "gender": "성별",
    "age_group": "연령대",
    "fixed_bet_amount": "일반 베팅 금액",
    "live_bet_amount": "라이브 베팅 금액",
    "fixed_bet_cnt": "일반 베팅 횟수",
    "live_bet_cnt": "라이브 베팅 횟수",
    "total_bet_cnt": "총 베팅 횟수",
    "fixed_active_days": "일반 활동 일수",
    "live_active_days": "라이브 활동 일수",
    "fixed_hit_days": "일반 적중 일수",
    "total_hit_days": "총 적중 일수",
    "fixed_win_rate": "일반 승률",
    "live_win_rate": "라이브 승률",
    "total_win_rate": "총 승률",
    "fixed_avg_roi": "일반 평균 ROI",
    "live_avg_roi": "라이브 평균 ROI",
    "total_avg_roi": "평균 ROI",
    "days_since_reg": "가입 경과일",
    "days_to_first_deposit": "첫 입금까지 일수",
    "days_to_first_bet": "첫 베팅까지 일수",
}

# ─── XGBoost 최종 모델 로드 (cached) ─────────────────────────
THRESHOLD = 0.5724  # 노트북 F1 최적 임계값

@st.cache_resource
def _load_model():
    import pandas as pd
    m = joblib.load("models/xgb_churn_v1/xgb_churn_v1.joblib")
    feat_imp = (
        pd.Series(m.feature_importances_, index=FEAT_NAMES)
        .sort_values(ascending=True)
        .tail(15)
    )
    return m, feat_imp

model, FEAT_IMP = _load_model()


def _build_x(gender, age_group, total_bet, live_ratio_pct,
              bet_cnt, active_days, win_rate, avg_roi,
              days_since_reg, days_to_first_deposit, days_to_first_bet) -> np.ndarray:
    """입력값으로부터 XGBoost 20개 피처 벡터를 구성합니다."""
    lr = live_ratio_pct / 100.0
    fr = 1.0 - lr

    fixed_bet  = total_bet * fr
    live_bet   = total_bet * lr
    fixed_cnt  = round(bet_cnt * fr)
    live_cnt   = round(bet_cnt * lr)
    fixed_days = round(active_days * max(0.05, fr))
    live_days  = round(active_days * lr)
    fixed_hit  = round(fixed_days * win_rate)
    total_hit  = round(active_days * win_rate)

    return np.array([
        gender, age_group,
        fixed_bet, live_bet,
        fixed_cnt, live_cnt, bet_cnt,
        fixed_days, live_days,
        fixed_hit, total_hit,
        win_rate, win_rate, win_rate,
        avg_roi, avg_roi, avg_roi,
        days_since_reg, days_to_first_deposit, days_to_first_bet,
    ], dtype=float)


# ─── Session state defaults ───────────────────────────────────
_defaults = {
    "w_gender": "남성", "w_age": "30대",
    "w_total_bet": 5000, "w_live_ratio": 40,
    "w_bet_cnt": 76,    "w_active_days": 23,
    "w_win_rate": 0.30, "w_avg_roi": -0.10,
    "w_days_reg": 16,   "w_days_dep": 2, "w_days_bet": 3,
}
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

# ─── Handle random sample ─────────────────────────────────────
if st.session_state.pop("do_random", False):
    rng = np.random.default_rng()
    st.session_state["w_gender"]     = str(rng.choice(["남성", "여성"]))
    st.session_state["w_age"]        = str(rng.choice(list(AGE_LABELS.values())))
    st.session_state["w_total_bet"]  = int(rng.integers(1, 41) * 500)
    st.session_state["w_live_ratio"] = int(rng.integers(0, 21) * 5)
    st.session_state["w_bet_cnt"]    = int(rng.integers(1, 51) * 10)
    st.session_state["w_active_days"]= int(rng.integers(1, 101))
    st.session_state["w_win_rate"]   = round(float(rng.uniform(0.0, 0.7)), 2)
    st.session_state["w_avg_roi"]    = round(float(rng.uniform(-1.0, 1.0)), 2)
    st.session_state["w_days_reg"]   = int(rng.integers(0, 28))
    st.session_state["w_days_dep"]   = int(rng.integers(0, 31))
    st.session_state["w_days_bet"]   = int(rng.integers(0, 31))

# ─── Sidebar ─────────────────────────────────────────────────
DIV = '<hr style="border:none;border-top:1px solid #E2E8F0;margin:0.8rem 0">'

with st.sidebar:
    st.markdown('<div class="sb-label">⚙ 고객 특성 입력</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)

    gender_lbl = st.radio("👤 성별", ["남성", "여성"], horizontal=True, key="w_gender")
    age_lbl    = st.selectbox("🎂 연령대", list(AGE_LABELS.values()), key="w_age")

    st.markdown(DIV, unsafe_allow_html=True)
    st.markdown('<div class="sb-label">베팅 행동</div>', unsafe_allow_html=True)

    total_bet   = st.slider("💰 총 베팅 금액 (₩)", 0, 20_000, step=500,  key="w_total_bet")
    live_ratio  = st.slider("📡 라이브 베팅 비율 (%)", 0, 100, step=5,    key="w_live_ratio")
    bet_cnt     = st.slider("🎲 총 베팅 횟수", 0, 500, step=5,             key="w_bet_cnt")
    active_days = st.slider("📅 총 활동 일수", 0, 100, step=1,             key="w_active_days")
    win_rate    = st.slider("🏆 총 승률", 0.0, 1.0, step=0.01, format="%.2f", key="w_win_rate")
    avg_roi     = st.slider("📈 평균 ROI", -1.0, 2.0, step=0.05, format="%.2f", key="w_avg_roi")

    st.markdown(DIV, unsafe_allow_html=True)
    st.markdown('<div class="sb-label">가입 이력</div>', unsafe_allow_html=True)

    days_since_reg        = st.slider("🗓 가입 경과일", 0, 27,  step=1, key="w_days_reg")
    days_to_first_deposit = st.slider("💳 첫 입금까지 일수", 0, 50, step=1, key="w_days_dep")
    days_to_first_bet     = st.slider("🎯 첫 베팅까지 일수", 0, 50, step=1, key="w_days_bet")

    st.markdown(DIV, unsafe_allow_html=True)

    if st.button("✨ 샘플 데이터 생성"):
        st.session_state["do_random"] = True
        st.rerun()

    st.markdown(
        '<div style="margin-top:1rem;font-size:0.73rem;color:#94A3B8;">'
        '<span class="ldot"></span>XGBoost · AUC 0.918 · threshold 0.572</div>',
        unsafe_allow_html=True,
    )

# ─── Encode & predict ────────────────────────────────────────
gender_val = 1 if gender_lbl == "남성" else 0
age_val    = AGE_REV[age_lbl]

x_vec      = _build_x(gender_val, age_val, total_bet, live_ratio,
                       bet_cnt, active_days, win_rate, avg_roi,
                       days_since_reg, days_to_first_deposit, days_to_first_bet)
churn_prob = float(model.predict_proba(x_vec.reshape(1, -1))[0, 1])

risk    = "높음" if churn_prob >= 0.66 else ("보통" if churn_prob >= 0.33 else "낮음")
r_color = C_CHURN if churn_prob >= 0.66 else (C_MID if churn_prob >= 0.33 else C_STAY)
outcome = "이탈 위험" if churn_prob >= THRESHOLD else "잔존 유지"
o_color = C_CHURN if churn_prob >= THRESHOLD else C_STAY

# ─── Header ──────────────────────────────────────────────────
hc1, hc2 = st.columns([7, 1])
with hc1:
    st.markdown('<div class="page-title">🎯 스포츠 베팅 고객 이탈 예측 분석</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">'
        '머신러닝 기반 실시간 이탈 확률 예측 · XGBoost (AUC 0.918) · '
        '좌측 사이드바에서 고객 특성을 조정하여 시뮬레이션하세요</div>',
        unsafe_allow_html=True,
    )
with hc2:
    st.markdown(
        f'<div style="text-align:right;padding-top:0.7rem;">'
        f'<span class="badge" style="background:{o_color}22;'
        f'color:{o_color};border:1px solid {o_color}55;">'
        f'● {outcome}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

# ─── KPI Row ─────────────────────────────────────────────────
k1, k2 = st.columns(2)
kpis = [
    (k1, "이탈 확률", f"{churn_prob*100:.1f}%", r_color, f"임계값 · {THRESHOLD}"),
    (k2, "위험 등급", risk,                      r_color, "낮음 / 보통 / 높음"),
]
for col, lbl, val, clr, sub in kpis:
    with col:
        st.markdown(
            f'<div class="kpi-card" style="border-top:3px solid {clr};">'
            f'<div class="kpi-label">{lbl}</div>'
            f'<div class="kpi-value" style="color:{clr};">{val}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)

# ─── Chart Row ───────────────────────────────────────────────
cl, cr = st.columns(2)

# ── 이탈 확률 게이지 ──────────────────────────────────────────
with cl:
    st.markdown(
        '<div class="sec-hdr">📊 예측 확률 게이지 · 실시간 이탈 위험도</div>',
        unsafe_allow_html=True,
    )
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(churn_prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": r_color, "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#CBD5E1",
                "tickfont": {"color": "#94A3B8", "size": 10},
                "dtick": 25,
            },
            "bar": {"color": r_color, "thickness": 0.22},
            "bgcolor": CARD_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  33], "color": "#F0FDF4"},
                {"range": [33, 66], "color": "#FFFBEB"},
                {"range": [66, 100], "color": "#FFF1F2"},
            ],
            "threshold": {
                "line": {"color": "rgba(100,100,100,0.25)", "width": 2},
                "thickness": 0.8,
                "value": THRESHOLD * 100,
            },
        },
        domain={"x": [0.05, 0.95], "y": [0.05, 1]},
    ))
    gauge.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=20, r=20, t=30, b=10),
        height=310,
        font={"family": "Noto Sans KR, Inter", "color": "#1E293B"},
    )
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

# ── 피처 중요도 (feature_importances_) ───────────────────────
with cr:
    st.markdown(
        '<div class="sec-hdr">🔍 피처 중요도 (Feature Importance) · 이탈 예측에 기여하는 상위 피처</div>',
        unsafe_allow_html=True,
    )
    names = [FEAT_KR.get(n, n) for n in FEAT_IMP.index]
    vals  = FEAT_IMP.values.tolist()

    bar = go.Figure(go.Bar(
        x=vals, y=names,
        orientation="h",
        marker=dict(color="#93C5FD", opacity=0.9, line=dict(width=0)),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#94A3B8", size=10, family="Noto Sans KR, Inter"),
    ))
    bar.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=70, t=10, b=20),
        height=310,
        font={"family": "Noto Sans KR, Inter", "color": "#64748B", "size": 11},
        xaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                   tickfont=dict(color="#94A3B8"), title="Importance"),
        yaxis=dict(showgrid=False, tickfont=dict(color="#475569", size=11)),
        bargap=0.3,
        showlegend=False,
    )
    st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

# ─── Retention Strategies ────────────────────────────────────
st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-hdr">💡 이탈 방지 전략 추천 · 예측 분석 기반 맞춤형 액션 플랜</div>',
    unsafe_allow_html=True,
)
st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

strats = []

if churn_prob >= THRESHOLD:
    if days_to_first_bet > 7:
        strats.append((C_BLUE, "초기 온보딩 강화 프로그램",
            f"가입 후 첫 베팅까지 {days_to_first_bet}일이 걸렸습니다. "
            "신규 가입자 대상 베팅 튜토리얼, 소액 무료 체험 쿠폰을 "
            "즉시 발송하여 플랫폼 적응을 도와주세요."))
    if active_days < 10:
        strats.append((C_BLUE, "재참여 이벤트 즉시 발송",
            f"총 활동 일수가 {active_days}일로 매우 낮습니다. "
            "무료 베팅 쿠폰, 첫 충전 보너스 등 맞춤형 재참여 이벤트를 "
            "48시간 이내 푸시 알림으로 발송하여 재방문을 유도하세요."))
    if win_rate < 0.15:
        strats.append((C_ORG, "베팅 가이드 & 전문가 분석 서비스 제공",
            f"총 승률이 {win_rate*100:.1f}%로 평균(30%) 대비 낮습니다. "
            "전문가 경기 분석 리포트, AI 베팅 추천 서비스를 무료로 제공하여 "
            "성공 경험을 높이고 이탈 위험을 줄이세요."))
    if avg_roi < -0.5:
        strats.append((C_CHURN, "손실 보전 캐시백 프로그램 제공",
            f"평균 ROI가 {avg_roi:.2f}로 손실이 큽니다. "
            "일정 기간 손실액의 10~20%를 보너스 크레딧으로 환급하는 "
            "캐시백 프로그램을 제안하여 심리적 이탈을 방지하세요."))
    if total_bet >= 10_000:
        strats.append((C_PURP, "VIP 전용 혜택 패키지 즉시 제공",
            f"총 베팅 금액이 ₩{total_bet:,}인 고가치 고객입니다. "
            "전담 VIP 매니저 배정, 전용 이벤트 초청, "
            "강화된 보너스 배율 등 프리미엄 혜택을 즉시 제공하세요."))
    if live_ratio > 60:
        strats.append((C_MID, "라이브 베팅 특별 프로모션 진행",
            f"라이브 베팅 비율이 {live_ratio}%로 높습니다. "
            "라이브 전용 배당 부스트, 실시간 경기 알림 서비스를 통해 "
            "라이브 베팅 경험을 강화하고 재방문율을 높이세요."))
    if not strats:
        strats.append((C_MID, "종합 리텐션 패키지 제공",
            f"이탈 확률이 {churn_prob*100:.1f}%입니다. "
            "충전 보너스 + AI 맞춤 경기 추천 + 1:1 고객 상담 연결을 "
            "패키지로 제공하여 이탈을 방지하세요."))

else:
    strats.append((C_STAY, "고객 건강 상태: 양호 — 업셀 전략 실행",
        f"이탈 확률이 낮습니다 ({churn_prob*100:.1f}%). "
        "현재 충성 고객으로, 프리미엄 서비스 업셀 및 크로스셀 전략을 "
        "검토하세요. 이 세그먼트는 프리미엄 상품 수용도가 2배 이상 높습니다."))
    if active_days > 50:
        strats.append((C_BLUE, "충성 고객 VIP 리워드 프로그램 안내",
            f"활동 일수 {active_days}일의 장기 활성 고객입니다. "
            "플래티넘 멤버십 등급 부여, 우선 출금 처리, 전용 CS 라인 등 "
            "VIP 전용 혜택을 제공하여 NPS 점수를 높이세요."))
    if total_bet >= 8_000 and churn_prob < 0.3:
        strats.append((C_PURP, "고가치 잔존 고객 심층 관리",
            f"₩{total_bet:,} 베팅의 고가치 잔존 고객입니다. "
            "분기별 개인 맞춤 혜택 제안과 전담 담당자 연결로 "
            "장기 충성도를 극대화하고 LTV를 높이세요."))

ncols = min(len(strats), 2)
sc = st.columns(ncols if ncols > 0 else 1)
for i, (clr, title, body) in enumerate(strats):
    with sc[i % len(sc)]:
        st.markdown(
            f'<div class="strat" style="border-left-color:{clr};">'
            f'<div class="strat-title">{title}</div>'
            f'<div class="strat-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
