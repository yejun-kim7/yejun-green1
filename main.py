# main.py
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── 한글 폰트 (Streamlit CSS) ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
#  데이터 로딩 (인코딩 자동 시도 + @st.cache_data)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_csv(file):
    """utf-8-sig → cp949 순서로 시도하고, 실패 시 빈 DataFrame 반환."""
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()


def read_uploaded(file):
    """업로드된 파일을 읽고 std_cols까지 적용한 DataFrame 반환."""
    df = load_csv(file)
    if df.empty:
        return pd.DataFrame()
    return std_cols(df)


# ════════════════════════════════════════════════════════════════════════════
#  열 이름 표준화 (KeyError 방지)
# ════════════════════════════════════════════════════════════════════════════
def std_cols(df):
    """CSV 열 이름을 표준 이름으로 통일하고 숫자 열은 숫자형으로 변환."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"

    df = df.rename(columns=rename)

    # 숫자 열은 숫자형으로 강제 변환 (글자·단위 혼재 대응)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    # 생중량 열이 없으면 지상부 질량 + 지하부 질량으로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # 시간 열 파싱 (날짜_시 형식도 처리)
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed

    return df


# ════════════════════════════════════════════════════════════════════════════
#  학교명 확인 (파일명 앞부분에서 추출)
# ════════════════════════════════════════════════════════════════════════════
def get_school_name(filename):
    """파일명에서 '_생육' 또는 '_환경' 앞부분을 학교명으로 추출."""
    if not filename:
        return None
    name = filename.replace(".csv", "")
    for sep in ("_생육", "_환경", "_생육결과", "_환경데이터"):
        idx = name.find(sep)
        if idx != -1:
            return name[:idx]
    return name[:8] if len(name) >= 8 else name


# ════════════════════════════════════════════════════════════════════════════
#  Plotly 공통 스타일
# ════════════════════════════════════════════════════════════════════════════
PLOT_FONT = dict(family="Malgun Gothic, 'Apple SD Gothic Neo', sans-serif")


def style_fig(fig):
    """그래프에 한글 폰트 적용."""
    fig.update_layout(font=PLOT_FONT)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  재사용 가능한 막대 그래프 데이터 준비 (서브플롯용 트레이스 반환)
# ════════════════════════════════════════════════════════════════════════════
def make_bar_trace(df, col, color="#607D8B"):
    """개체별 막대 그래프 트레이스 하나 생성 (subplots에 추가용)."""
    d = df[["개체번호", col]].dropna()
    if len(d) < 2:
        return None
    # 생중량이면 상위 25% / 하위 25% 색으로 구분
    if col == "생중량":
        q75 = d[col].quantile(0.75)
        q25 = d[col].quantile(0.25)
        colors = [
            "#4CAF50" if v >= q75 else
            "#F44336" if v <= q25 else
            "#9E9E9E"
            for v in d[col]
        ]
    else:
        colors = [color] * len(d)

    return go.Bar(
        x=d["개체번호"].astype(str),
        y=d[col],
        marker_color=colors,
        hovertemplate=f"{col}<br>개체: %{{x}}<br>값: %{{y:.2f}}<extra></extra>",
    )


# ════════════════════════════════════════════════════════════════════════════
#  일별 평균 꺾은선 그래프 (재사용 함수)
# ════════════════════════════════════════════════════════════════════════════
def daily_line(df, col, label, color, unit=""):
    """일별 평균 꺾은선 그래프 하나."""
    if col not in df.columns:
        return None
    daily = df.groupby("date")[col].mean().reset_index().dropna()
    if len(daily) < 2:
        return None
    fig = go.Figure(
        data=[
            go.Scatter(
                x=daily["date"],
                y=daily[col],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"{label}<br>날짜: %{{x}}<br>값: %{{y:.2f}}{unit}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"{label} 일별 평균",
        xaxis_title="날짜",
        yaxis_title=f"{label} {unit}",
        margin=dict(t=40, b=40, l=40, r=20),
        height=280,
    )
    style_fig(fig)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  사이드바 구성
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

st.sidebar.header("📁 CSV 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type=["csv"])
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("📏 생육 측정 항목")

growth_cols_options = [
    "생중량",
    "잎 수",
    "지상부 길이",
    "지하부 길이",
    "지상부 질량",
    "지하부 질량",
]
selected_measures = st.sidebar.multiselect(
    "비교할 항목",
    growth_cols_options,
    default=["생중량", "잎 수", "지상부 길이", "지하부 길이"],
)

# ── 업로드된 CSV 읽기 (한 번만) ──────────────────────────────────────────
with st.spinner("CSV를 읽어오는 중..."):
    env_df = read_uploaded(env_file)
    growth_df = read_uploaded(growth_file)

school_name_env = get_school_name(env_file.name) if env_file else None
school_name_g = get_school_name(growth_file.name) if growth_file else None
school_name = school_name_g or school_name_env or "우리 학교"


# ════════════════════════════════════════════════════════════════════════════
#  탭 1 — 📖 실험 개요
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header(f"🔬 {school_name} 극지식물 생육 실험 개요")

        col_bg, col_obj = st.columns(2)
        with col_bg:
            st.markdown("""
            **연구 배경 및 목적**

            극지식물은 강한 바람, 낮은 온도, 적은 수분 등 혹독한 환경에서도 살아남는 식물입니다.
            이 대시보드는 우리 학교 극지식물 개체별로 생육 상태에 어떤 차이가 있는지 살펴보고,
            잎 수·지상부 길이와 생중량 사이에 어떤 관계가 있는지 분석합니다.

            - **평균**: 모든 개체 값을 더해 개체 수로 나눈 값 (대표값)
            - **표준편차**: 개체들이 평균에서 얼마나 퍼져 있는지 나타내는 값
            - **상관계수**: 두 변수가 함께 증가하거나 감소하는지 나타내는 값 (−1 ~ 1)
            - **CV(변동계수)**: 표준편차를 평균으로 나눈 값 — 단위가 다른 집단 간 비교에 유용
            """)

        with col_obj:
            if not growth_df.empty:
                st.metric("총 개체 수", f"{len(growth_df)}개체")
            else:
                st.info("📌 생육 결과 CSV를 업로드하면 총 개체 수가 표시됩니다.")

            if not env_df.empty and "time" in env_df.columns:
                t = env_df["time"].dropna()
                if len(t) > 0:
                    st.metric(
                        "측정 기간",
                        f"{t.min().strftime('%Y-%m-%d')} ~ {t.max().strftime('%Y-%m-%d')}",
                    )
                else:
                    st.info("📌 측정 기간을 확인할 수 없습니다.")
            else:
                st.info("📌 환경 CSV를 업로드하면 측정 기간이 표시됩니다.")

            avail_g = [c for c in growth_cols_options if c in growth_df.columns]
            if avail_g:
                st.metric("측정 항목 (생육)", ", ".join(avail_g))
            else:
                st.info("📌 생육 CSV를 업로드하면 측정 항목이 표시됩니다.")

            env_avail = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_avail:
                st.metric("환경 측정 항목", ", ".join(env_avail))
            else:
                st.info("📌 환경 CSV 업로드 시 환경 측정 항목이 표시됩니다.")

        st.markdown("---")
        st.subheader("📋 데이터 요약 표")

        summary_rows = []
        if not growth_df.empty:
            summary_rows.append([
                "생육 결과 데이터",
                f"{len(growth_df)}개체",
                ", ".join(sorted(set(growth_df.columns))),
            ])
        if not env_df.empty:
            summary_rows.append([
                "환경 데이터",
                f"{len(env_df)}행",
                ", ".join(sorted(set(env_df.columns))),
            ])

        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows, columns=["데이터", "크기", "열 이름"]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("📌 두 CSV 중 하나 이상을 업로드하면 요약 표가 표시됩니다.")

        st.markdown("---")
        st.subheader("📊 주요 지표 카드")

        c1, c2, c3, c4 = st.columns(4)

        if not growth_df.empty and "개체번호" in growth_df.columns:
            c1.metric("총 개체 수", len(growth_df))
        else:
            c1.metric("총 개체 수", "—")

        if not env_df.empty and "temperature" in env_df.columns:
            t_vals = env_df["temperature"].dropna()
            if len(t_vals) >= 2:
                c2.metric("평균 온도", f"{t_vals.mean():.1f} ℃")
            else:
                c2.metric("평균 온도", "데이터 부족")
        else:
            c2.metric("평균 온도", "—")

        if not env_df.empty and "humidity" in env_df.columns:
            h_vals = env_df["humidity"].dropna()
            if len(h_vals) >= 2:
                c3.metric("평균 습도", f"{h_vals.mean():.1f} %")
            else:
                c3.metric("평균 습도", "데이터 부족")
        else:
            c3.metric("평균 습도", "—")

        if not growth_df.empty and "생중량" in growth_df.columns:
            gw_vals = growth_df["생중량"].dropna()
            if len(gw_vals) >= 2:
                c4.metric("평균 생중량", f"{gw_vals.mean():.2f} g")
            else:
                c4.metric("평균 생중량", "데이터 부족")
        else:
            c4.metric("평균 생중량", "—")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  탭 2 — 🌡️ 환경 데이터
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info("📌 환경 데이터 CSV를 사이드바에 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터 분석")

            # ── 환경 요약 표 ──────────────────────────────────────────────
            st.subheader("환경 요약 (평균 · 최소 · 최대)")
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]
                if c in env_df.columns
            ]
            if env_num_cols:
                env_clean = env_df[env_num_cols].dropna()
                if len(env_clean) >= 2:
                    # melt 방식으로 요약 표를 직접 구성 → 축 길이 불일치 방지
                    summ_data = {
                        "항목": [],
                        "평균": [],
                        "최소": [],
                        "최대": [],
                    }
                    unit_map = {
                        "temperature": "℃",
                        "humidity": "%",
                        "ph": "",
                        "ec": "mS/cm",
                        "water_temp": "℃",
                        "co2": "ppm",
                    }
                    for col in env_num_cols:
                        s = env_clean[col].dropna()
                        if len(s) >= 2:
                            summ_data["항목"].append(f"{col} ({unit_map.get(col, '')})".strip())
                            summ_data["평균"].append(round(s.mean(), 2))
                            summ_data["최소"].append(round(s.min(), 2))
                            summ_data["최대"].append(round(s.max(), 2))
                    if summ_data["항목"]:
                        st.dataframe(
                            pd.DataFrame(summ_data),
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("📌 환경 데이터가 부족하여 요약 통계를 계산할 수 없습니다.")
                else:
                    st.info("📌 환경 데이터가 부족하여 요약 통계를 계산할 수 없습니다.")
            else:
                st.info("📌 환경 데이터에 숫자 열(온도·습도·pH·EC 등)이 없어 요약을 표시할 수 없습니다.")

            st.markdown("---")

            # ── 날짜별 시계열 ──────────────────────────────────────────────
            st.subheader("날짜별 환경 변화 (일별 평균)")

            if "time" in env_df.columns:
                env_df["date"] = env_df["time"].dt.date

                # 온도
                if "temperature" in env_df.columns:
                    fig_t = daily_line(env_df, "temperature", "온도", "#FF5722", "℃")
                    if fig_t:
                        st.plotly_chart(fig_t, width="stretch")
                    else:
                        st.info("📌 온도 데이터의 일별 평균을 계산할 수 없습니다.")
                else:
                    st.info("📌 환경 데이터에 온도(temperature) 열이 없습니다.")

                # 습도
                if "humidity" in env_df.columns:
                    fig_h = daily_line(env_df, "humidity", "습도", "#2196F3", "%")
                    if fig_h:
                        st.plotly_chart(fig_h, width="stretch")
                    else:
                        st.info("📌 습도 데이터의 일별 평균을 계산할 수 없습니다.")
                else:
                    st.info("📌 환경 데이터에 습도(humidity) 열이 없습니다.")

                # pH · EC (열이 있을 때만)
                pH_EC_cols = [c for c in ["ph", "ec"] if c in env_df.columns]
                if pH_EC_cols:
                    fig_sub = make_subplots(
                        rows=len(pH_EC_cols), cols=1,
                        subplot_titles=[f"{c.upper()} 일별 평균" for c in pH_EC_cols],
                    )
                    for i, col in enumerate(pH_EC_cols):
                        daily = env_df.groupby("date")[col].mean().reset_index().dropna()
                        if len(daily) >= 2:
                            fig_sub.add_trace(
                                go.Scatter(
                                    x=daily["date"],
                                    y=daily[col],
                                    mode="lines",
                                    name=col.upper(),
                                    line=dict(
                                        color="#9C27B0" if col == "ph" else "#00BCD4",
                                        width=2,
                                    ),
                                    hovertemplate=f"{col.upper()}<br>날짜: %{{x}}<br>값: %{{y:.2f}}<extra></extra>",
                                ),
                                row=i + 1, col=1,
                            )
                    fig_sub.update_layout(
                        margin=dict(t=40, b=40, l=40, r=20),
                        height=200 * len(pH_EC_cols),
                    )
                    style_fig(fig_sub)
                    st.plotly_chart(fig_sub, width="stretch")
                else:
                    st.info("📌 환경 데이터에 pH·EC 열이 없어 pH·EC 그래프를 표시할 수 없습니다.")
            else:
                st.info("📌 환경 데이터에 시간(time) 열이 없어 일별 시계열을 그릴 수 없습니다.")

            st.markdown("---")

            # ── expander: 원본 테이블 + 다운로드 ──────────────────────────
            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                if not env_df.empty:
                    st.dataframe(
                        env_df,
                        width="stretch",
                        hide_index=True,
                    )
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    safe_name = (school_name or "환경데이터") + "_환경데이터.csv"
                    st.download_button(
                        label="📥 환경 데이터 CSV 다운로드",
                        data=csv_bytes,
                        file_name=safe_name,
                        mime="text/csv",
                    )
                else:
                    st.info("📌 표시할 환경 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  탭 3 — 📊 생육 결과
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info("📌 생육 결과 CSV를 사이드바에 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header(f"📊 {school_name} 개체별 생육 결과 분석")

            if "생중량" not in growth_df.columns:
                st.warning("⚠️ 생중량(g) 열이 없어 지상부 질량 + 지하부 질량으로 계산했습니다.")

            gw_col = "생중량"

            # ── 🥇 핵심 결과 카드 ──────────────────────────────────────────
            st.subheader("🥇 핵심 결과")
            kc1, kc2, kc3 = st.columns(3)

            if gw_col in growth_df.columns:
                gw_clean = growth_df[gw_col].dropna()
                if len(gw_clean) >= 2:
                    kc1.metric("평균 생중량", f"{gw_clean.mean():.2f} g")

                    valid_gw = growth_df.dropna(subset=[gw_col])
                    if len(valid_gw) >= 2:
                        idx_max = valid_gw[gw_col].idxmax()
                        idx_min = valid_gw[gw_col].idxmin()
                        max_id = valid_gw.loc[idx_max, "개체번호"]
                        min_id = valid_gw.loc[idx_min, "개체번호"]
                        max_val = valid_gw.loc[idx_max, gw_col]
                        min_val = valid_gw.loc[idx_min, gw_col]
                        kc2.metric("최대 생중량 개체", f"개체 {max_id} ({max_val:.2f} g)")
                        kc3.metric("최소 생중량 개체", f"개체 {min_id} ({min_val:.2f} g)")
                    else:
                        kc2.metric("최대 생중량 개체", "데이터 부족")
                        kc3.metric("최소 생중량 개체", "데이터 부족")
                else:
                    kc1.metric("평균 생중량", "데이터 부족")
                    kc2.metric("최대 생중량 개체", "데이터 부족")
                    kc3.metric("최소 생중량 개체", "데이터 부족")
            else:
                kc1.metric("평균 생중량", "데이터 부족")
                kc2.metric("최대 생중량 개체", "데이터 부족")
                kc3.metric("최소 생중량 개체", "데이터 부족")

            st.markdown("---")

            # ── 개체별 생육 비교 (막대 그래프 서브플롯) ────────────────────
            st.subheader("개체별 생육 비교 (막대 그래프)")

            bar_cols = [c for c in selected_measures if c in growth_df.columns]
            if not bar_cols:
                st.info("📌 업로드한 생육 CSV에 선택한 측정 항목이 없어 막대 그래프를 표시할 수 없습니다.")
            else:
                n_graphs = len(bar_cols)
                rows_needed = (n_graphs + 1) // 2
                fig_grid = make_subplots(
                    rows=rows_needed,
                    cols=2,
                    subplot_titles=bar_cols,
                    vertical_spacing=0.12,
                )

                for i, col in enumerate(bar_cols):
                    tr = make_bar_trace(growth_df, col)
                    if tr is None:
                        continue
                    row_i = (i // 2) + 1
                    col_i = (i % 2) + 1
                    fig_grid.add_trace(tr, row=row_i, col=col_i)

                fig_grid.update_layout(
                    margin=dict(t=60, b=40, l=40, r=20),
                    height=300 * rows_needed,
                    showlegend=False,
                )
                style_fig(fig_grid)
                st.plotly_chart(fig_grid, width="stretch")

                st.markdown("""
                > **막대 그래프 범례**
                > - 🟢 **초록색**: 생중량 상위 25% 개체
                > - 🔴 **빨간색**: 생중량 하위 25% 개체
                > - ⚫ **회색**: 중간 50% 개체
                """)

            st.markdown("---")

            # ── 생중량 분포 (박스플롯) + 통계 표 ───────────────────────────
            if gw_col in growth_df.columns:
                gw_drop = growth_df[[gw_col]].dropna()
                if len(gw_drop) >= 2:
                    fig_box = go.Figure(
                        data=[
                            go.Box(
                                y=gw_drop[gw_col],
                                name="생중량 분포",
                                fillcolor="#4CAF50",
                                line_color="#2E7D32",
                                hovertemplate=f"생중량: %{{y:.2f}} g<extra></extra>",
                            )
                        ]
                    )
                    fig_box.update_layout(
                        title="생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        margin=dict(t=40, b=40, l=40, r=20),
                        height=320,
                    )
                    style_fig(fig_box)
                    st.plotly_chart(fig_box, width="stretch")

                    gw_vals = gw_drop[gw_col]
                    cv = (gw_vals.std() / gw_vals.mean()) if gw_vals.mean() > 0 else 0.0
                    stat_rows = [
                        ["평균", f"{gw_vals.mean():.2f} g",
                         "모든 개체 생중량의 대표값"],
                        ["표준편차", f"{gw_vals.std():.2f} g",
                         "개체 간 차이가 얼마나 큰지 나타냄"],
                        ["CV(변동계수)", f"{cv:.2%}",
                         "단위가 다른 집단과 비교할 때 사용"],
                        ["최소", f"{gw_vals.min():.2f} g",
                         "생중량이 가장 작은 개체"],
                        ["최대", f"{gw_vals.max():.2f} g",
                         "생중량이 가장 큰 개체"],
                        ["상위 25% 기준", f"{gw_vals.quantile(0.75):.2f} g",
                         "여기 이상이면 상위 25%"],
                        ["하위 25% 기준", f"{gw_vals.quantile(0.25):.2f} g",
                         "여기 이하이면 하위 25%"],
                    ]
                    st.dataframe(
                        pd.DataFrame(stat_rows, columns=["지표", "값", "설명"]),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("📌 생중량 데이터가 부족하여 분포 그래프를 표시할 수 없습니다.")
            else:
                st.info("📌 생육 데이터에 생중량 열이 없어 분포 그래프를 표시할 수 없습니다.")

            st.markdown("---")

            # ── 상관관계 분석 (산점도 + 추세선) ────────────────────────────
            st.subheader("상관관계 분석 (산점도 + 추세선)")

            x_candidates = [c for c in ["잎 수", "지상부 길이"] if c in growth_df.columns]
            if gw_col not in growth_df.columns:
                st.info("📌 생중량 열이 없어 상관관계 분석을 수행할 수 없습니다.")
            elif not x_candidates:
                st.info("📌 잎 수·지상부 길이 열이 없어 상관관계 분석을 수행할 수 없습니다.")
            else:
                for x_col in x_candidates:
                    valid = growth_df[[x_col, gw_col]].dropna()
                    if len(valid) < 2:
                        st.info(
                            f"📌 {x_col} vs 생중량 데이터가 부족하여 상관관계를 계산할 수 없습니다."
                        )
                        continue

                    x_vals = valid[x_col].values
                    y_vals = valid[gw_col].values
                    r = np.corrcoef(x_vals, y_vals)[0, 1]

                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    y_line = slope * x_line + intercept

                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="markers",
                            marker=dict(color="#4CAF50", size=7, opacity=0.7),
                            hovertemplate=(
                                f"{x_col}: %{{x:.1f}}<br>생중량: %{{y:.2f}} g<extra></extra>"
                            ),
                            name="개별 개체",
                        )
                    )
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="#F44336", width=2, dash="dash"),
                            name=f"추세선 (r = {r:.3f})",
                            hovertemplate=(
                                f"추세선<br>{x_col}: %{{x:.1f}}<br>"
                                f"예측 생중량: %{{y:.2f}} g<extra></extra>"
                            ),
                        )
                    )
                    fig_scatter.update_layout(
                        title=f"{x_col} vs 생중량  (상관계수 r = {r:.3f})",
                        xaxis_title=x_col,
                        yaxis_title="생중량 (g)",
                        margin=dict(t=40, b=40, l=40, r=20),
                        height=360,
                    )
                    style_fig(fig_scatter)
                    st.plotly_chart(fig_scatter, width="stretch")

                    direction = (
                        "강한 양의 상관" if r > 0.7 else
                        "중간 양의 상관" if r > 0.4 else
                        "약한 상관" if abs(r) < 0.4 else
                        "중간 음의 상관" if r > -0.7 else "강한 음의 상관"
                    )
                    st.info(
                        f"📌 **{x_col} vs 생중량** — 상관계수 r = {r:.3f} ({direction})\n"
                        f"  → 상관계수는 −1에서 1 사이 값으로, 1에 가까울수록 두 값이 함께 증가합니다."
                    )

            st.markdown("---")

            # ── expander: 생육 원본 + 다운로드 ──────────────────────────────
            with st.expander("📋 생육 데이터 원본 + CSV 다운로드"):
                if not growth_df.empty:
                    st.dataframe(
                        growth_df,
                        width="stretch",
                        hide_index=True,
                    )
                    csv_g = growth_df.to_csv(index=False).encode("utf-8-sig")
                    safe_name = (school_name or "생육결과") + "_생육결과데이터.csv"
                    st.download_button(
                        label="📥 생육 결과 CSV 다운로드",
                        data=csv_g,
                        file_name=safe_name,
                        mime="text/csv",
                    )
                else:
                    st.info("📌 표시할 생육 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  푸터 안내
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    f"🌱 데이터는 {school_name} 학생들이 직접 측정한 자료를 바탕으로 합니다.  "
    "평균·표준편차·상관계수 등은 추세를 이해하는 보조 도구로 사용합니다."
)
