import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 앱 및 데이터 설정
# ─────────────────────────────────────────────
APP_TITLE = "영화 유형 나누기"
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/"
    "modudata/main/data/kobis_movies.csv"
)

FEATURES = {
    "로그 스크린 수": "log_first_scrn",
    "로그 누적 관객": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

CLUSTER_COLORS = {
    "㉮": "#E74C3C",
    "㉯": "#3498DB",
    "㉰": "#2ECC71",
    "㉱": "#9B59B6",
    "㉲": "#F39C12",
    "㉳": "#1ABC9C",
    "㉴": "#795548",
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎬",
    layout="wide",
)

st.title(f"🎬 {APP_TITLE}")
st.caption(
    "영화의 흥행 속성을 표준화한 뒤 k-평균으로 비슷한 영화를 묶습니다."
)


# ─────────────────────────────────────────────
# 데이터 불러오기
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    raw_df = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.stop()

total_count = len(raw_df)

required_columns = [
    "movieCd",
    "movieNm",
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
]

missing_columns = [
    column for column in required_columns if column not in raw_df.columns
]

if missing_columns:
    st.error(
        "데이터에서 필요한 열을 찾을 수 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ─────────────────────────────────────────────
# 데이터 전처리
# ─────────────────────────────────────────────
df = raw_df.copy()

numeric_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# 로그 변환을 위해 first_scrn과 total_audi는 양수여야 하며,
# 롱런 지수 계산을 위해 first_week_audi도 양수여야 함
valid_mask = (
    df["first_scrn"].notna()
    & df["first_week_audi"].notna()
    & df["total_audi"].notna()
    & df["days_in_top10"].notna()
    & (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
    & (df["first_week_audi"] > 0)
)

df = df.loc[valid_mask].copy()

df["log_first_scrn"] = df["first_scrn"].map(math.log10)
df["log_total_audi"] = df["total_audi"].map(math.log10)
df["long_run_index"] = (
    df["total_audi"] / df["first_week_audi"]
).clip(upper=20)

df = df.dropna(subset=list(FEATURES.values())).copy()
clustered_count = len(df)

st.markdown(
    f"**전체 영화 {total_count:,}편 · 유형으로 묶은 영화 "
    f"{clustered_count:,}편**"
)

if clustered_count < 7:
    st.error(
        "유효한 영화가 7편보다 적어 요청한 범위의 군집화를 "
        "수행할 수 없습니다."
    )
    st.stop()


# ─────────────────────────────────────────────
# 군집화 설정
# ─────────────────────────────────────────────
st.subheader("1. 군집화 설정")

selected_labels = st.multiselect(
    "묶는 데 사용할 속성을 둘 이상 선택하세요.",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
)

cluster_count = st.slider(
    "묶음 수",
    min_value=2,
    max_value=7,
    value=3,
    step=1,
)

if len(selected_labels) < 2:
    st.warning("k-평균 군집화를 위해 속성을 둘 이상 선택해 주세요.")
    st.stop()

selected_columns = [FEATURES[label] for label in selected_labels]

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[selected_columns])


# ─────────────────────────────────────────────
# 현재 설정으로 k-평균 실행
# ─────────────────────────────────────────────
kmeans = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=10,
)

df["raw_cluster"] = kmeans.fit_predict(scaled_values)

# 누적 관객 평균이 큰 군집부터 ㉮, ㉯ ... 기호 부여
cluster_order = (
    df.groupby("raw_cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

active_symbols = CLUSTER_SYMBOLS[:cluster_count]

cluster_name_map = {
    raw_cluster: active_symbols[index]
    for index, raw_cluster in enumerate(cluster_order)
}

df["묶음"] = df["raw_cluster"].map(cluster_name_map)
df["묶음"] = pd.Categorical(
    df["묶음"],
    categories=active_symbols,
    ordered=True,
)

active_color_map = {
    symbol: CLUSTER_COLORS[symbol]
    for symbol in active_symbols
}


# ─────────────────────────────────────────────
# 2차원 산점도
# ─────────────────────────────────────────────
st.subheader("2. 2차원 산점도")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_label = st.selectbox(
        "가로축",
        options=selected_labels,
        index=0,
        key="scatter_2d_x",
    )

with axis_col2:
    y_label = st.selectbox(
        "세로축",
        options=selected_labels,
        index=1,
        key="scatter_2d_y",
    )

fig_2d = px.scatter(
    df,
    x=FEATURES[x_label],
    y=FEATURES[y_label],
    color="묶음",
    category_orders={"묶음": active_symbols},
    color_discrete_map=active_color_map,
    hover_name="movieNm",
    labels={
        FEATURES[x_label]: x_label,
        FEATURES[y_label]: y_label,
        "묶음": "영화 유형",
    },
)

fig_2d.update_traces(
    marker={"size": 7, "opacity": 0.75},
    hovertemplate="<b>%{hovertext}</b><extra></extra>",
)

fig_2d.update_layout(
    height=600,
    legend_title_text="영화 유형",
)

st.plotly_chart(fig_2d, use_container_width=True)


# ─────────────────────────────────────────────
# 3차원 산점도
# ─────────────────────────────────────────────
st.subheader("3. 3차원 산점도")

if len(selected_labels) < 3:
    st.info(
        "3차원 산점도를 보려면 묶는 데 사용할 속성을 "
        "세 개 이상 선택해 주세요."
    )
else:
    axis_3d_col1, axis_3d_col2, axis_3d_col3 = st.columns(3)

    with axis_3d_col1:
        x3_label = st.selectbox(
            "x축",
            options=selected_labels,
            index=0,
            key="scatter_3d_x",
        )

    with axis_3d_col2:
        y3_label = st.selectbox(
            "y축",
            options=selected_labels,
            index=1,
            key="scatter_3d_y",
        )

    with axis_3d_col3:
        z3_label = st.selectbox(
            "z축",
            options=selected_labels,
            index=2,
            key="scatter_3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=FEATURES[x3_label],
        y=FEATURES[y3_label],
        z=FEATURES[z3_label],
        color="묶음",
        category_orders={"묶음": active_symbols},
        color_discrete_map=active_color_map,
        hover_name="movieNm",
        labels={
            FEATURES[x3_label]: x3_label,
            FEATURES[y3_label]: y3_label,
            FEATURES[z3_label]: z3_label,
            "묶음": "영화 유형",
        },
    )

    fig_3d.update_traces(
        marker={"size": 3, "opacity": 0.75},
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
    )

    fig_3d.update_layout(
        height=700,
        legend_title_text="영화 유형",
        scene={
            "xaxis_title": x3_label,
            "yaxis_title": y3_label,
            "zaxis_title": z3_label,
        },
    )

    st.plotly_chart(fig_3d, use_container_width=True)


# ─────────────────────────────────────────────
# 묶음별 요약 통계
# ─────────────────────────────────────────────
st.subheader("4. 묶음별 특성")

summary = (
    df.groupby("묶음", observed=False)
    .agg(
        편수=("movieCd", "count"),
        평균_스크린_수=("first_scrn", "mean"),
        평균_누적_관객=("total_audi", "mean"),
        평균_10위권_일수=("days_in_top10", "mean"),
        평균_롱런_지수=("long_run_index", "mean"),
    )
    .reset_index()
)

summary.columns = [
    "묶음",
    "편수",
    "스크린 수 평균",
    "누적 관객 평균",
    "10위권 일수 평균",
    "롱런 지수 평균",
]

summary_display = summary.copy()
summary_display["편수"] = summary_display["편수"].map(
    lambda value: f"{int(value):,}"
)
summary_display["스크린 수 평균"] = summary_display[
    "스크린 수 평균"
].map(lambda value: f"{value:,.1f}")
summary_display["누적 관객 평균"] = summary_display[
    "누적 관객 평균"
].map(lambda value: f"{value:,.0f}")
summary_display["10위권 일수 평균"] = summary_display[
    "10위권 일수 평균"
].map(lambda value: f"{value:,.1f}")
summary_display["롱런 지수 평균"] = summary_display[
    "롱런 지수 평균"
].map(lambda value: f"{value:,.2f}")

st.dataframe(
    summary_display,
    hide_index=True,
    use_container_width=True,
)


# ─────────────────────────────────────────────
# 묶음별 누적 관객 상위 5편
# ─────────────────────────────────────────────
st.subheader("5. 묶음별 누적 관객 상위 영화")

top_columns = st.columns(cluster_count)

for index, cluster_symbol in enumerate(active_symbols):
    cluster_df = df[df["묶음"] == cluster_symbol]

    top_movies = (
        cluster_df.sort_values("total_audi", ascending=False)
        .head(5)
    )

    with top_columns[index]:
        st.markdown(f"### {cluster_symbol}")
        st.caption(f"{len(cluster_df):,}편")

        if top_movies.empty:
            st.write("영화가 없습니다.")
        else:
            for rank, row in enumerate(
                top_movies.itertuples(),
                start=1,
            ):
                st.markdown(
                    f"**{rank}. {row.movieNm}**  \n"
                    f"누적 관객 {row.total_audi:,.0f}명"
                )


# ─────────────────────────────────────────────
# 엘보 분석: 묶음 수 1~7의 군집 내 제곱합
# ─────────────────────────────────────────────
st.subheader("6. 묶음 수에 따른 군집 내 제곱합")

inertia_rows = []

for k in range(1, 8):
    elbow_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    elbow_model.fit(scaled_values)

    inertia_rows.append(
        {
            "묶음 수": k,
            "군집 내 제곱합": elbow_model.inertia_,
        }
    )

inertia_df = pd.DataFrame(inertia_rows)
inertia_df["이전 값에서 감소량"] = (
    inertia_df["군집 내 제곱합"].shift(1)
    - inertia_df["군집 내 제곱합"]
)

fig_elbow = go.Figure()

fig_elbow.add_trace(
    go.Scatter(
        x=inertia_df["묶음 수"],
        y=inertia_df["군집 내 제곱합"],
        mode="lines+markers",
        name="군집 내 제곱합",
        line={"color": "#3498DB", "width": 3},
        marker={"size": 9},
        hovertemplate=(
            "묶음 수: %{x}<br>"
            "군집 내 제곱합: %{y:,.2f}"
            "<extra></extra>"
        ),
    )
)

# 현재 선택한 묶음 수 위치에 세로선 표시
fig_elbow.add_vline(
    x=cluster_count,
    line_width=2,
    line_dash="dash",
    line_color="#E74C3C",
    annotation_text=f"현재 선택: {cluster_count}",
    annotation_position="top right",
)

fig_elbow.update_layout(
    height=500,
    xaxis={
        "title": "묶음 수",
        "tickmode": "linear",
        "tick0": 1,
        "dtick": 1,
        "range": [0.7, 7.3],
    },
    yaxis={"title": "군집 내 제곱합"},
    showlegend=False,
)

st.plotly_chart(fig_elbow, use_container_width=True)

inertia_display = inertia_df.copy()

inertia_display["군집 내 제곱합"] = inertia_display[
    "군집 내 제곱합"
].map(lambda value: f"{value:,.2f}")

inertia_display["이전 값에서 감소량"] = inertia_display[
    "이전 값에서 감소량"
].map(
    lambda value: ""
    if pd.isna(value)
    else f"{value:,.2f}"
)

st.dataframe(
    inertia_display,
    hide_index=True,
    use_container_width=True,
)


# ─────────────────────────────────────────────
# 실루엣 점수
# ─────────────────────────────────────────────
st.subheader("7. 선택한 묶음 수의 실루엣 점수")

current_silhouette = silhouette_score(
    scaled_values,
    df["raw_cluster"],
)

st.markdown(
    f"**묶음 수 {cluster_count}의 실루엣 점수: "
    f"{current_silhouette:.4f}** — "
    "점수는 -1에서 1 사이이며, 1에 가까울수록 "
    "묶음이 뚜렷합니다."
)
