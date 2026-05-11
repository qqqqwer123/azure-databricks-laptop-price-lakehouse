from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="筆電價格智能分析平台",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "laptop_price_rf_model.pkl"
MODEL_FEATURES_PATH = MODEL_DIR / "model_features.pkl"

# Analytics tables exported from Databricks Gold layer
BRAND_PRICE_PATH = DATA_DIR / "brand_price_summary.csv"
PRICE_BAND_PATH = DATA_DIR / "price_band_summary.csv"
SPEC_PRICE_PATH = DATA_DIR / "spec_price_summary.csv"
CP_VALUE_PATH = DATA_DIR / "cp_value_ranking.csv"

# ML result tables exported from Databricks Gold layer
MODEL_METRICS_PATH = DATA_DIR / "model_evaluation_metrics.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance_summary.csv"
BRAND_PREMIUM_PATH = DATA_DIR / "brand_premium_residual.csv"
WHAT_IF_PATH = DATA_DIR / "what_if_prediction_results.csv"
ML_METADATA_PATH = DATA_DIR / "ml_pipeline_metadata.csv"

# =========================================================
# Style
# =========================================================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e6e8eb;
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.04);
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
    }
    .info-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 18px;
        margin: 8px 0 16px 0;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Helper Functions
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Failed to load {path.name}: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_model_features():
    if not MODEL_FEATURES_PATH.exists():
        return {
            "numeric_features": ["ram_gb", "harddisk_gb", "screen_size"],
            "categorical_features": ["brand"],
            "target": "price",
        }
    return joblib.load(MODEL_FEATURES_PATH)


def normalize_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalize common Databricks-exported column names for app use."""
    if df is None:
        return None
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def price_band(price: float) -> str:
    if price < 500:
        return "Low"
    if price < 1000:
        return "Mid"
    if price < 1500:
        return "High"
    return "Premium"


def render_header(title: str, subtitle: str):
    st.markdown(f"<div class='main-header'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>{subtitle}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_numeric(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_brand_list(*dfs) -> list:
    brands = set()
    for df in dfs:
        if df is not None and "brand" in df.columns:
            brands.update(df["brand"].dropna().astype(str).str.lower().unique().tolist())
    if not brands:
        brands = {"dell", "hp", "lenovo", "asus", "acer", "apple", "msi", "samsung", "lg", "rokc"}
    return sorted(brands)

# =========================================================
# Load Data
# =========================================================
brand_price_df = normalize_columns(load_csv(BRAND_PRICE_PATH))
price_band_df = normalize_columns(load_csv(PRICE_BAND_PATH))
spec_price_df = normalize_columns(load_csv(SPEC_PRICE_PATH))
cp_value_df = normalize_columns(load_csv(CP_VALUE_PATH))

model_metrics_df = normalize_columns(load_csv(MODEL_METRICS_PATH))
feature_importance_df = normalize_columns(load_csv(FEATURE_IMPORTANCE_PATH))
brand_premium_df = normalize_columns(load_csv(BRAND_PREMIUM_PATH))
what_if_df = normalize_columns(load_csv(WHAT_IF_PATH))
ml_metadata_df = normalize_columns(load_csv(ML_METADATA_PATH))

model = load_model()
model_features = load_model_features()

# Numeric casting for common files
for _df, _cols in [
    (brand_price_df, ["product_count", "avg_price", "min_price", "max_price"]),
    (price_band_df, ["product_count", "avg_price"]),
    (spec_price_df, ["ram_gb", "harddisk_gb", "product_count", "avg_price"]),
    (cp_value_df, ["price", "rating_num", "ram_gb", "harddisk_gb", "screen_size", "cp_score"]),
    (model_metrics_df, ["MAE", "RMSE", "R2", "mae", "rmse", "r2"]),
    (feature_importance_df, ["importance", "rank"]),
    (brand_premium_df, ["avg_actual_price", "avg_predicted_price", "avg_residual", "median_residual", "sample_count"]),
    (what_if_df, ["ram_gb", "harddisk_gb", "screen_size", "predicted_price", "price_change_vs_base", "price_change_pct_vs_base"]),
]:
    if _df is not None:
        safe_numeric(_df, _cols)

# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("筆電價格智能平台")
st.sidebar.caption("Azure Databricks Lakehouse + 機器學習展示")
page = st.sidebar.radio(
    "頁面導覽",
    [
        "🏠 專案總覽",
        "🗂 資料管線",
        "📊 市場分析",
        "🤖 模型洞察",
        "💻 價格模擬器",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**專案資料層級**")
st.sidebar.markdown("Bronze 原始層 → Silver 清理層 → Gold 分析層 → Dashboard → Streamlit")
st.sidebar.markdown("---")
st.sidebar.caption("此作品集展示 Databricks Gold 輸出結果與訓練完成的 Random Forest 模型。")

# =========================================================
# Page 1: Overview
# =========================================================
if page == "🏠 專案總覽":
    render_header(
        "筆電價格智能分析平台",
        "一個端到端的 Azure Databricks Lakehouse 專案，涵蓋價格分析、資料治理與機器學習驅動的價格模擬。",
    )

    st.markdown(
        """
        <div class='info-box'>
        這個應用程式是本專案的資料消費層。資料會先經過 ADF 與 Azure Databricks 處理，整理成 Gold tables 後，再用於儀表板分析與模型驅動的價格模擬。
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    total_products = "N/A"
    avg_price = "N/A"
    brand_count = "N/A"
    highest_brand = "N/A"

    if brand_price_df is not None and not brand_price_df.empty:
        if "product_count" in brand_price_df.columns:
            total_products = f"{int(brand_price_df['product_count'].sum()):,}"
        if "avg_price" in brand_price_df.columns:
            avg_price = f"${brand_price_df['avg_price'].mean():,.0f}"
            if "brand" in brand_price_df.columns:
                highest_row = brand_price_df.sort_values("avg_price", ascending=False).iloc[0]
                highest_brand = str(highest_row["brand"]).title()
        if "brand" in brand_price_df.columns:
            brand_count = f"{brand_price_df['brand'].nunique():,}"

    with c1:
        metric_card("產品總數", total_products)
    with c2:
        metric_card("平均品牌價格", avg_price)
    with c3:
        metric_card("品牌數量", brand_count)
    with c4:
        metric_card("最高平均價格品牌", highest_brand)

    st.markdown("<div class='section-title'>市場快速總覽</div>", unsafe_allow_html=True)
    left, right = st.columns(2)

    if brand_price_df is not None and {"brand", "avg_price"}.issubset(brand_price_df.columns):
        chart_df = brand_price_df.sort_values("avg_price", ascending=False).head(15)
        fig = px.bar(chart_df, x="brand", y="avg_price", title="各品牌平均筆電價格")
        fig.update_layout(xaxis_title="品牌", yaxis_title="平均價格", height=420)
        left.plotly_chart(fig, use_container_width=True)
    else:
        left.info("請加入 data/brand_price_summary.csv 以啟用此圖表。")

    if brand_price_df is not None and {"brand", "product_count"}.issubset(brand_price_df.columns):
        count_df = brand_price_df.sort_values("product_count", ascending=False).head(15)
        fig = px.bar(count_df, x="brand", y="product_count", title="各品牌產品數量")
        fig.update_layout(xaxis_title="品牌", yaxis_title="產品數量", height=420)
        right.plotly_chart(fig, use_container_width=True)
    else:
        right.info("請在 brand_price_summary.csv 中加入 product_count 欄位以啟用此圖表。")

    st.markdown("<div class='section-title'>此專案展示的能力</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            **資料工程範圍**
            - ADF 流程編排
            - Bronze / Silver / Gold 分層架構
            - Delta Lake 資料表
            - Unity Catalog 資料治理
            - Gold 分析資料表
            """
        )
    with b:
        st.markdown(
            """
            **分析與機器學習範圍**
            - 品牌與價格帶分析
            - 規格與價格關係分析
            - 特徵重要性分析
            - 品牌溢價殘差分析
            - 互動式情境價格預測
            """
        )

# =========================================================
# Page 2: Data Pipeline
# =========================================================
elif page == "🗂 資料管線":
    render_header(
        "資料管線與 Lakehouse 架構",
        "說明原始筆電資料如何被轉換成可治理的 Gold tables 與可供模型使用的輸出結果。",
    )

    st.markdown("<div class='section-title'>端到端資料流程</div>", unsafe_allow_html=True)
    st.code(
        """
Kaggle / 原始 CSV
    ↓
Azure Blob Storage
    ↓
Azure Data Factory 流程編排
    ↓
Databricks Bronze Table：laptop_raw_landing
    ↓
Databricks Silver Table：laptop_cleaned
    ↓
Gold 分析資料表 + Gold 機器學習結果表
    ↓
Databricks Dashboard / Streamlit App
        """,
        language="text",
    )

    st.markdown("<div class='section-title'>Medallion 資料分層</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class='metric-card'>
            <div class='metric-value'>Bronze</div>
            <div class='small-note'>原始資料落地層。保留資料來源的原始結構，只做最小程度的轉換。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='metric-card'>
            <div class='metric-value'>Silver</div>
            <div class='small-note'>清理與標準化資料層。處理價格、RAM、儲存容量、螢幕尺寸、空值與重複資料。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class='metric-card'>
            <div class='metric-value'>Gold</div>
            <div class='small-note'>可供商業分析使用的資料表，以及提供儀表板消費的機器學習結果表。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>資料表清單</div>", unsafe_allow_html=True)
    inventory = pd.DataFrame(
        [
            ["Bronze", "laptop_raw_landing", "由 ADF 匯入的原始落地資料表"],
            ["Silver", "laptop_cleaned", "清理後且可供模型使用的筆電資料集"],
            ["Gold - Analytics", "brand_price_summary", "品牌層級的價格與產品數量摘要"],
            ["Gold - Analytics", "price_band_summary", "依價格帶統計的產品分布"],
            ["Gold - Analytics", "spec_price_summary", "規格與平均價格摘要"],
            ["Gold - Analytics", "cp_value_ranking", "性價比排行資料表"],
            ["Gold - ML", "model_evaluation_metrics", "模型表現指標"],
            ["Gold - ML", "feature_importance_summary", "Random Forest 的特徵重要性"],
            ["Gold - ML", "brand_premium_residual", "品牌溢價殘差分析"],
            ["Gold - ML", "what_if_prediction_results", "預先定義的 What-if 模擬情境"],
            ["Gold - ML", "ml_pipeline_metadata", "選用的機器學習可追溯性 metadata"],
        ],
        columns=["資料層級", "資料表", "用途"],
    )
    st.dataframe(inventory, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>資料治理摘要</div>", unsafe_allow_html=True)
    governance = pd.DataFrame(
        [
            ["Data Engineer", "讀取 / 寫入", "讀取 / 寫入", "讀取 / 寫入"],
            ["Data Analyst", "無權限", "僅可讀取", "僅可讀取"],
            ["Business User", "無權限", "無權限", "僅可讀取"],
        ],
        columns=["角色", "Bronze", "Silver", "Gold"],
    )
    st.dataframe(governance, use_container_width=True, hide_index=True)

# =========================================================
# Page 3: Market Analytics
# =========================================================
elif page == "📊 市場分析":
    render_header(
        "市場分析",
        "由 Databricks Gold 分析資料表產生的商業洞察。",
    )

    tab1, tab2, tab3, tab4 = st.tabs(["品牌分析", "價格帶分析", "規格分析", "性價比排行"])

    with tab1:
        st.subheader("品牌分析")
        if brand_price_df is None:
            st.info("請加入 data/brand_price_summary.csv 以啟用此區塊。")
        else:
            col1, col2 = st.columns(2)
            if {"brand", "avg_price"}.issubset(brand_price_df.columns):
                fig = px.bar(
                    brand_price_df.sort_values("avg_price", ascending=False),
                    x="brand",
                    y="avg_price",
                    title="各品牌平均價格",
                )
                fig.update_layout(height=420)
                col1.plotly_chart(fig, use_container_width=True)
            if {"brand", "product_count"}.issubset(brand_price_df.columns):
                fig = px.bar(
                    brand_price_df.sort_values("product_count", ascending=False),
                    x="brand",
                    y="product_count",
                    title="各品牌產品數量",
                )
                fig.update_layout(height=420)
                col2.plotly_chart(fig, use_container_width=True)
            st.dataframe(brand_price_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("價格帶分布")
        if price_band_df is None:
            st.info("請加入 data/price_band_summary.csv 以啟用此區塊。")
        else:
            if {"brand", "price_band", "product_count"}.issubset(price_band_df.columns):
                fig = px.bar(
                    price_band_df,
                    x="brand",
                    y="product_count",
                    color="price_band",
                    title="各品牌價格帶分布",
                )
                fig.update_layout(height=500, barmode="stack")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(price_band_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("規格與價格關係")
        if spec_price_df is None:
            st.info("請加入 data/spec_price_summary.csv 以啟用此區塊。")
        else:
            if {"ram_gb", "avg_price"}.issubset(spec_price_df.columns):
                x_col = "ram_gb"
                y_col = "avg_price"
                size_col = "product_count" if "product_count" in spec_price_df.columns else None
                color_col = "harddisk_gb" if "harddisk_gb" in spec_price_df.columns else None
                fig = px.scatter(
                    spec_price_df,
                    x=x_col,
                    y=y_col,
                    size=size_col,
                    color=color_col,
                    title="RAM 與儲存容量對應的平均價格",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(spec_price_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("性價比排行")
        if cp_value_df is None:
            st.info("請加入 data/cp_value_ranking.csv 以啟用此區塊。")
        else:
            brands = get_brand_list(cp_value_df)
            selected_brands = st.multiselect("依品牌篩選", brands, default=brands[: min(5, len(brands))])
            filtered = cp_value_df.copy()
            if selected_brands and "brand" in filtered.columns:
                filtered = filtered[filtered["brand"].astype(str).str.lower().isin(selected_brands)]
            if "cp_score" in filtered.columns:
                filtered = filtered.sort_values("cp_score", ascending=False)
            st.dataframe(filtered.head(30), use_container_width=True, hide_index=True)

# =========================================================
# Page 4: ML Insights
# =========================================================
elif page == "🤖 模型洞察":
    render_header(
        "機器學習洞察與可解釋性",
        "根據 Gold 機器學習結果表呈現模型表現、特徵重要性與品牌溢價分析。",
    )

    st.markdown("<div class='section-title'>模型表現</div>", unsafe_allow_html=True)
    if model_metrics_df is None:
        st.info("請加入 data/model_evaluation_metrics.csv 以顯示模型表現。")
    else:
        # Flexible metric names
        metrics = model_metrics_df.copy()
        mae_col = "MAE" if "MAE" in metrics.columns else "mae" if "mae" in metrics.columns else None
        rmse_col = "RMSE" if "RMSE" in metrics.columns else "rmse" if "rmse" in metrics.columns else None
        r2_col = "R2" if "R2" in metrics.columns else "r2" if "r2" in metrics.columns else None
        best_row = metrics.iloc[-1]
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("MAE", f"${float(best_row[mae_col]):,.2f}" if mae_col else "N/A")
        with c2:
            metric_card("RMSE", f"${float(best_row[rmse_col]):,.2f}" if rmse_col else "N/A")
        with c3:
            metric_card("R²", f"{float(best_row[r2_col]):.3f}" if r2_col else "N/A")
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>特徵重要性</div>", unsafe_allow_html=True)
    if feature_importance_df is None:
        st.info("請加入 data/feature_importance_summary.csv 以顯示特徵重要性。")
    else:
        if {"feature", "importance"}.issubset(feature_importance_df.columns):
            top_n = st.slider("顯示前幾名特徵", 5, 30, 15)
            top_features = feature_importance_df.sort_values("importance", ascending=False).head(top_n)
            fig = px.bar(
                top_features.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="特徵重要性排行",
            )
            fig.update_layout(height=520, yaxis_title="特徵", xaxis_title="重要性")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>品牌溢價殘差分析</div>", unsafe_allow_html=True)
    if brand_premium_df is None:
        st.info("請加入 data/brand_premium_residual.csv 以顯示品牌溢價分析。")
    else:
        if {"brand", "avg_residual"}.issubset(brand_premium_df.columns):
            filtered = brand_premium_df.copy()
            if "sample_count" in filtered.columns:
                min_count = st.slider("最小樣本數", 1, int(max(1, filtered["sample_count"].max())), min(10, int(max(1, filtered["sample_count"].max()))))
                filtered = filtered[filtered["sample_count"] >= min_count]
            fig = px.bar(
                filtered.sort_values("avg_residual", ascending=False),
                x="brand",
                y="avg_residual",
                title="各品牌平均殘差",
            )
            fig.update_layout(height=480, yaxis_title="實際價格 - 預測價格")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(brand_premium_df, use_container_width=True, hide_index=True)

    with st.expander("方法說明"):
        st.markdown(
            """
            - 訓練資料來源：`databricks0501.silver.laptop_cleaned`
            - 模型：Random Forest Regressor
            - 輸入特徵：品牌、RAM、儲存容量、螢幕尺寸
            - 預測目標：筆電價格
            - 品牌溢價透過不含 brand 特徵的模型估計，再依品牌分析殘差。
            """
        )
        if ml_metadata_df is not None:
            st.dataframe(ml_metadata_df, use_container_width=True, hide_index=True)

# =========================================================
# Page 5: Price Simulator
# =========================================================
elif page == "💻 價格模擬器":
    render_header(
        "互動式價格模擬器",
        "使用訓練完成的 Random Forest 模型，根據品牌與硬體規格估算筆電價格。",
    )

    if model is None:
        st.error("找不到模型檔。請將 laptop_price_rf_model.pkl 放在 streamlit_app/models/ 底下。")
        st.stop()

    brand_list = get_brand_list(brand_price_df, brand_premium_df, cp_value_df)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='section-title'>輸入規格設定</div>", unsafe_allow_html=True)
        brand = st.selectbox("品牌", brand_list, index=brand_list.index("dell") if "dell" in brand_list else 0)
        ram_gb = st.selectbox("RAM（GB）", [4, 8, 16, 32, 64], index=2)
        harddisk_gb = st.selectbox("儲存容量（GB）", [128, 256, 512, 1024, 2048], index=2)
        screen_size = st.selectbox("螢幕尺寸（吋）", [11.6, 13.3, 14.0, 15.6, 16.0, 17.3], index=3)
        predict_clicked = st.button("預測價格", type="primary")

    input_df = pd.DataFrame(
        [
            {
                "ram_gb": ram_gb,
                "harddisk_gb": harddisk_gb,
                "screen_size": screen_size,
                "brand": brand,
            }
        ]
    )

    with right:
        st.markdown("<div class='section-title'>預測結果</div>", unsafe_allow_html=True)
        predicted_price = float(model.predict(input_df)[0])
        band = price_band(predicted_price)

        c1, c2 = st.columns(2)
        with c1:
            metric_card("預測價格", f"${predicted_price:,.2f}")
        with c2:
            metric_card("價格帶", band)

        if brand_price_df is not None and {"brand", "avg_price"}.issubset(brand_price_df.columns):
            brand_avg = brand_price_df[brand_price_df["brand"].astype(str).str.lower() == brand]
            if not brand_avg.empty:
                avg = float(brand_avg.iloc[0]["avg_price"])
                delta = predicted_price - avg
                st.metric("與該品牌平均價格相比", f"${delta:,.2f}")

    st.markdown("<div class='section-title'>規格升級情境比較</div>", unsafe_allow_html=True)
    scenario_df = pd.DataFrame(
        [
            {"brand": brand, "ram_gb": 8, "harddisk_gb": 256, "screen_size": screen_size},
            {"brand": brand, "ram_gb": 16, "harddisk_gb": 512, "screen_size": screen_size},
            {"brand": brand, "ram_gb": 32, "harddisk_gb": 1024, "screen_size": screen_size},
            {"brand": brand, "ram_gb": 64, "harddisk_gb": 2048, "screen_size": screen_size},
        ]
    )
    scenario_df["predicted_price"] = model.predict(scenario_df)
    scenario_df["price_band"] = scenario_df["predicted_price"].apply(price_band)
    scenario_df["price_change_vs_base"] = scenario_df["predicted_price"] - scenario_df["predicted_price"].iloc[0]

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
    with col2:
        fig = px.line(
            scenario_df,
            x="ram_gb",
            y="predicted_price",
            markers=True,
            title=f"{brand.title()} 規格升級情境",
        )
        fig.update_layout(height=360, xaxis_title="RAM（GB）", yaxis_title="預測價格")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>結果解讀</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='info-box'>
        針對目前選擇的規格，模型預估價格為 <b>${predicted_price:,.2f}</b>，所屬價格帶為 <b>{band}</b>。
        這個結果來自使用 Silver 清理資料訓練出的模型，並透過 Streamlit 介面提供 What-if 價格模擬。
        </div>
        """,
        unsafe_allow_html=True,
    )
