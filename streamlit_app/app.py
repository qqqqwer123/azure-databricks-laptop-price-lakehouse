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
    page_title="Laptop Price Intelligence Platform",
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
st.sidebar.title("Laptop Intelligence")
st.sidebar.caption("Azure Databricks Lakehouse + ML Demo")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🗂 Data Pipeline",
        "📊 Market Analytics",
        "🤖 ML Insights",
        "💻 Price Simulator",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Layers**")
st.sidebar.markdown("Bronze → Silver → Gold → Dashboard → Streamlit")
st.sidebar.markdown("---")
st.sidebar.caption("Portfolio demo built from Databricks Gold outputs and a trained Random Forest model.")

# =========================================================
# Page 1: Overview
# =========================================================
if page == "🏠 Overview":
    render_header(
        "Laptop Price Intelligence Platform",
        "An end-to-end Azure Databricks Lakehouse project for pricing analytics, governance, and ML-powered price simulation."
    )

    st.markdown(
        """
        <div class='info-box'>
        This app represents the consumption layer of the project. Data is processed through ADF and Azure Databricks, curated into Gold tables, and then used for dashboarding and model-based price simulation.
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
        metric_card("Total Products", total_products)
    with c2:
        metric_card("Average Brand Price", avg_price)
    with c3:
        metric_card("Brands", brand_count)
    with c4:
        metric_card("Highest Avg Price Brand", highest_brand)

    st.markdown("<div class='section-title'>Quick Market Snapshot</div>", unsafe_allow_html=True)
    left, right = st.columns(2)

    if brand_price_df is not None and {"brand", "avg_price"}.issubset(brand_price_df.columns):
        chart_df = brand_price_df.sort_values("avg_price", ascending=False).head(15)
        fig = px.bar(chart_df, x="brand", y="avg_price", title="Average Laptop Price by Brand")
        fig.update_layout(xaxis_title="Brand", yaxis_title="Average Price", height=420)
        left.plotly_chart(fig, use_container_width=True)
    else:
        left.info("Add data/brand_price_summary.csv to enable this chart.")

    if brand_price_df is not None and {"brand", "product_count"}.issubset(brand_price_df.columns):
        count_df = brand_price_df.sort_values("product_count", ascending=False).head(15)
        fig = px.bar(count_df, x="brand", y="product_count", title="Product Count by Brand")
        fig.update_layout(xaxis_title="Brand", yaxis_title="Product Count", height=420)
        right.plotly_chart(fig, use_container_width=True)
    else:
        right.info("Add product_count to brand_price_summary.csv to enable this chart.")

    st.markdown("<div class='section-title'>What This Project Demonstrates</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            **Data Engineering Scope**
            - ADF orchestration
            - Bronze / Silver / Gold architecture
            - Delta Lake tables
            - Unity Catalog governance
            - Gold analytical tables
            """
        )
    with b:
        st.markdown(
            """
            **Analytics & ML Scope**
            - Brand and price band analysis
            - Specification-price relationship
            - Feature importance
            - Brand premium residual
            - Interactive what-if prediction
            """
        )

# =========================================================
# Page 2: Data Pipeline
# =========================================================
elif page == "🗂 Data Pipeline":
    render_header(
        "Data Pipeline & Lakehouse Architecture",
        "How raw laptop data is transformed into governed Gold tables and model-ready outputs.",
    )

    st.markdown("<div class='section-title'>End-to-End Flow</div>", unsafe_allow_html=True)
    st.code(
        """
Kaggle / Raw CSV
    ↓
Azure Blob Storage
    ↓
Azure Data Factory Orchestration
    ↓
Databricks Bronze Table: laptop_raw_landing
    ↓
Databricks Silver Table: laptop_cleaned
    ↓
Gold Analytical Tables + Gold ML Result Tables
    ↓
Databricks Dashboard / Streamlit App
        """,
        language="text",
    )

    st.markdown("<div class='section-title'>Medallion Layers</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class='metric-card'>
            <div class='metric-value'>Bronze</div>
            <div class='small-note'>Raw landing layer. Preserves original source structure with minimal transformation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='metric-card'>
            <div class='metric-value'>Silver</div>
            <div class='small-note'>Cleaned and standardized data. Handles price, RAM, storage, screen size, nulls, and duplicates.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class='metric-card'>
            <div class='metric-value'>Gold</div>
            <div class='small-note'>Business-ready analytical tables and ML result tables for dashboard consumption.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Table Inventory</div>", unsafe_allow_html=True)
    inventory = pd.DataFrame(
        [
            ["Bronze", "laptop_raw_landing", "Raw landing table from ADF ingestion"],
            ["Silver", "laptop_cleaned", "Cleaned model-ready laptop dataset"],
            ["Gold - Analytics", "brand_price_summary", "Brand-level price and product count summary"],
            ["Gold - Analytics", "price_band_summary", "Product distribution by price band"],
            ["Gold - Analytics", "spec_price_summary", "Specification and average price summary"],
            ["Gold - Analytics", "cp_value_ranking", "Cost-performance ranking table"],
            ["Gold - ML", "model_evaluation_metrics", "Model performance metrics"],
            ["Gold - ML", "feature_importance_summary", "Feature importance from Random Forest"],
            ["Gold - ML", "brand_premium_residual", "Brand premium residual analysis"],
            ["Gold - ML", "what_if_prediction_results", "Predefined what-if scenarios"],
            ["Gold - ML", "ml_pipeline_metadata", "Optional ML traceability metadata"],
        ],
        columns=["Layer", "Table", "Purpose"],
    )
    st.dataframe(inventory, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Governance Summary</div>", unsafe_allow_html=True)
    governance = pd.DataFrame(
        [
            ["Data Engineer", "Read / Write", "Read / Write", "Read / Write"],
            ["Data Analyst", "No access", "Read only", "Read only"],
            ["Business User", "No access", "No access", "Read only"],
        ],
        columns=["Role", "Bronze", "Silver", "Gold"],
    )
    st.dataframe(governance, use_container_width=True, hide_index=True)

# =========================================================
# Page 3: Market Analytics
# =========================================================
elif page == "📊 Market Analytics":
    render_header(
        "Market Analytics",
        "Business-ready insights generated from Databricks Gold analytical tables.",
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Brand Analysis", "Price Band", "Specification Analysis", "Value Ranking"])

    with tab1:
        st.subheader("Brand Analysis")
        if brand_price_df is None:
            st.info("Add data/brand_price_summary.csv to enable this section.")
        else:
            col1, col2 = st.columns(2)
            if {"brand", "avg_price"}.issubset(brand_price_df.columns):
                fig = px.bar(
                    brand_price_df.sort_values("avg_price", ascending=False),
                    x="brand",
                    y="avg_price",
                    title="Average Price by Brand",
                )
                fig.update_layout(height=420)
                col1.plotly_chart(fig, use_container_width=True)
            if {"brand", "product_count"}.issubset(brand_price_df.columns):
                fig = px.bar(
                    brand_price_df.sort_values("product_count", ascending=False),
                    x="brand",
                    y="product_count",
                    title="Product Count by Brand",
                )
                fig.update_layout(height=420)
                col2.plotly_chart(fig, use_container_width=True)
            st.dataframe(brand_price_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Price Band Distribution")
        if price_band_df is None:
            st.info("Add data/price_band_summary.csv to enable this section.")
        else:
            if {"brand", "price_band", "product_count"}.issubset(price_band_df.columns):
                fig = px.bar(
                    price_band_df,
                    x="brand",
                    y="product_count",
                    color="price_band",
                    title="Price Band Distribution by Brand",
                )
                fig.update_layout(height=500, barmode="stack")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(price_band_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Specification vs Price")
        if spec_price_df is None:
            st.info("Add data/spec_price_summary.csv to enable this section.")
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
                    title="Average Price by RAM and Storage",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(spec_price_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Cost-Performance Ranking")
        if cp_value_df is None:
            st.info("Add data/cp_value_ranking.csv to enable this section.")
        else:
            brands = get_brand_list(cp_value_df)
            selected_brands = st.multiselect("Filter by brand", brands, default=brands[: min(5, len(brands))])
            filtered = cp_value_df.copy()
            if selected_brands and "brand" in filtered.columns:
                filtered = filtered[filtered["brand"].astype(str).str.lower().isin(selected_brands)]
            if "cp_score" in filtered.columns:
                filtered = filtered.sort_values("cp_score", ascending=False)
            st.dataframe(filtered.head(30), use_container_width=True, hide_index=True)

# =========================================================
# Page 4: ML Insights
# =========================================================
elif page == "🤖 ML Insights":
    render_header(
        "ML Insights & Explainability",
        "Model performance, feature importance, and brand premium analysis from the Gold ML result tables.",
    )

    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
    if model_metrics_df is None:
        st.info("Add data/model_evaluation_metrics.csv to show model performance.")
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

    st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
    if feature_importance_df is None:
        st.info("Add data/feature_importance_summary.csv to show feature importance.")
    else:
        if {"feature", "importance"}.issubset(feature_importance_df.columns):
            top_n = st.slider("Top features", 5, 30, 15)
            top_features = feature_importance_df.sort_values("importance", ascending=False).head(top_n)
            fig = px.bar(
                top_features.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Top Feature Importance",
            )
            fig.update_layout(height=520, yaxis_title="Feature", xaxis_title="Importance")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Brand Premium Residual Analysis</div>", unsafe_allow_html=True)
    if brand_premium_df is None:
        st.info("Add data/brand_premium_residual.csv to show brand premium analysis.")
    else:
        if {"brand", "avg_residual"}.issubset(brand_premium_df.columns):
            filtered = brand_premium_df.copy()
            if "sample_count" in filtered.columns:
                min_count = st.slider("Minimum sample count", 1, int(max(1, filtered["sample_count"].max())), min(10, int(max(1, filtered["sample_count"].max()))))
                filtered = filtered[filtered["sample_count"] >= min_count]
            fig = px.bar(
                filtered.sort_values("avg_residual", ascending=False),
                x="brand",
                y="avg_residual",
                title="Average Residual by Brand",
            )
            fig.update_layout(height=480, yaxis_title="Actual Price - Predicted Price")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(brand_premium_df, use_container_width=True, hide_index=True)

    with st.expander("Methodology Notes"):
        st.markdown(
            """
            - Training source: `databricks0501.silver.laptop_cleaned`
            - Model: Random Forest Regressor
            - Input features: brand, RAM, storage, screen size
            - Target: laptop price
            - Brand premium is estimated by training a model without brand and analyzing residuals by brand.
            """
        )
        if ml_metadata_df is not None:
            st.dataframe(ml_metadata_df, use_container_width=True, hide_index=True)

# =========================================================
# Page 5: Price Simulator
# =========================================================
elif page == "💻 Price Simulator":
    render_header(
        "Interactive Price Simulator",
        "Estimate laptop price based on brand and hardware configuration using the trained Random Forest model.",
    )

    if model is None:
        st.error("Model file not found. Please place laptop_price_rf_model.pkl under streamlit_app/models/.")
        st.stop()

    brand_list = get_brand_list(brand_price_df, brand_premium_df, cp_value_df)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='section-title'>Input Configuration</div>", unsafe_allow_html=True)
        brand = st.selectbox("Brand", brand_list, index=brand_list.index("dell") if "dell" in brand_list else 0)
        ram_gb = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=2)
        harddisk_gb = st.selectbox("Storage (GB)", [128, 256, 512, 1024, 2048], index=2)
        screen_size = st.selectbox("Screen Size (inch)", [11.6, 13.3, 14.0, 15.6, 16.0, 17.3], index=3)
        predict_clicked = st.button("Predict Price", type="primary")

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
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        predicted_price = float(model.predict(input_df)[0])
        band = price_band(predicted_price)

        c1, c2 = st.columns(2)
        with c1:
            metric_card("Predicted Price", f"${predicted_price:,.2f}")
        with c2:
            metric_card("Price Band", band)

        if brand_price_df is not None and {"brand", "avg_price"}.issubset(brand_price_df.columns):
            brand_avg = brand_price_df[brand_price_df["brand"].astype(str).str.lower() == brand]
            if not brand_avg.empty:
                avg = float(brand_avg.iloc[0]["avg_price"])
                delta = predicted_price - avg
                st.metric("Compared with Brand Average", f"${delta:,.2f}")

    st.markdown("<div class='section-title'>Scenario Comparison</div>", unsafe_allow_html=True)
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
            title=f"Upgrade Scenario for {brand.title()}",
        )
        fig.update_layout(height=360, xaxis_title="RAM (GB)", yaxis_title="Predicted Price")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Interpretation</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='info-box'>
        For the selected configuration, the model estimates a price of <b>${predicted_price:,.2f}</b>, placing it in the <b>{band}</b> band.
        This result is generated from the model trained on the Silver cleaned dataset and served through the Streamlit interface as a what-if pricing simulator.
        </div>
        """,
        unsafe_allow_html=True,
    )
