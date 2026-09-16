import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────
APP_TITLE = "🎬 영화 유형 나누기"
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

CLUSTER_NAMES = ["㉮", "㉯", "㉰"]
CLUSTER_COLORS = {
    "㉮": "#E74C3C",
    "㉯": "#3498DB",
    "㉰": "#2ECC71",
}

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title(APP_TITLE)
st.caption("영화의 흥행 특성을 선택하고 k-평균으로 세 가지 유형을 찾아봅니다.")


# ─────────────────────────────────────────────
# 데이터 불러오기 및 전처리
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


raw_df = load_data()
total_count = len(raw_df)

numeric_columns = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
]

df = raw_df.copy()

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# 네 속성을 만들 수 없는 행 및 첫 주 관객이 0인 행 제외
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

# 상용로그 및 롱런 지수 계산
df["log_first_scrn"] = df["first_scrn"].map(lambda x: __import__("math").log10(x))
df["log_total_audi"] = df["total_audi"].map(lambda x: __import__("math").log10(x))
df["long_run_index"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)

# 계산 결과가 유효한 행만 유지
model_columns = list(FEATURES.values())
df = df.dropna(subset=model_columns).copy()
clustered_count = len(df)

st.write(
    f"**전체 영화 {total_count:,}편 · 유형으로 묶은 영화 {clustered_count:,}편**"
)

if clustered_count < 3:
    st.error("유효한 영화가 세 편보다 적어 k-평균 군집화를 수행할 수 없습니다.")
    st.stop()


# ─────────────────────────────────────────────
# 군집화 속성 선택
# ─────────────────────────────────────────────
st.subheader("1. 묶는 데 사용할 속성")

selected_labels = st.multiselect(
    "둘 이상의 속성을 선택하세요.",
    options=list(FEATURES.keys()),
    default=list(FEATURES.keys()),
)

if len(selected_labels) < 2:
    st.warning("k-평균 군집화를 위해 속성을 둘 이상 선택해 주세요.")
    st.stop()

selected_columns = [FEATURES[label] for label in selected_labels]

scaled_values = StandardScaler().fit_transform(df[selected_columns])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

df["raw_cluster"] = kmeans.fit_predict(scaled_values)

# 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    df.groupby("raw_cluster")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

cluster_name_map = {
    raw_cluster: CLUSTER_NAMES[index]
    for index, raw_cluster in enumerate(cluster_order)
}

df["묶음"] = df["raw_cluster"].map(cluster_name_map)
df["묶음"] = pd.Categorical(
    df["묶음"],
    categories=CLUSTER_NAMES,
    ordered=True,
)


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
    y_default = 1 if len(selected_labels) > 1 else 0
    y_label = st.selectbox(
        "세로축",
        options=selected_labels,
        index=y_default,
        key="scatter_2d_y",
    )

fig_2d = px.scatter(
    df,
    x=FEATURES[x_label],
    y=FEATURES[y_label],
    color="묶음",
    category_orders={"묶음": CLUSTER_NAMES},
    color_discrete_map=CLUSTER_COLORS,
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
    legend_title_text="영화 유형",
    height=600,
)

st.plotly_chart(fig_2d, use_container_width=True)


# ─────────────────────────────────────────────
# 3차원 산점도
# ─────────────────────────────────────────────
st.subheader("3. 3차원 산점도")

if len(selected_labels) < 3:
    st.info(
        "3차원 산점도를 보려면 묶는 데 사용할 속성을 세 개 이상 선택해 주세요."
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
        category_orders={"묶음": CLUSTER_NAMES},
        color_discrete_map=CLUSTER_COLORS,
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
# 군집별 요약
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
summary_display["편수"] = summary_display["편수"].map(lambda x: f"{int(x):,}")
summary_display["스크린 수 평균"] = summary_display["스크린 수 평균"].map(
    lambda x: f"{x:,.1f}"
)
summary_display["누적 관객 평균"] = summary_display["누적 관객 평균"].map(
    lambda x: f"{x:,.0f}"
)
summary_display["10위권 일수 평균"] = summary_display["10위권 일수 평균"].map(
    lambda x: f"{x:,.1f}"
)
summary_display["롱런 지수 평균"] = summary_display["롱런 지수 평균"].map(
    lambda x: f"{x:,.2f}"
)

st.dataframe(
    summary_display,
    hide_index=True,
    use_container_width=True,
)


# ─────────────────────────────────────────────
# 군집별 누적 관객 상위 5편
# ─────────────────────────────────────────────
st.subheader("5. 묶음별 누적 관객 상위 영화")

top_columns = st.columns(3)

for index, cluster_name in enumerate(CLUSTER_NAMES):
    top_movies = (
        df[df["묶음"] == cluster_name]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with top_columns[index]:
        st.markdown(f"### {cluster_name}")
        st.caption(f"{len(df[df['묶음'] == cluster_name]):,}편")

        if top_movies.empty:
            st.write("영화가 없습니다.")
        else:
            for rank, row in enumerate(top_movies.itertuples(), start=1):
                st.markdown(
                    f"**{rank}. {row.movieNm}**  \n"
                    f"누적 관객 {row.total_audi:,.0f}명"
                )
