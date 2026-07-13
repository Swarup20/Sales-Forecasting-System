import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting Dashboard")
st.markdown("### Xylofy AI Internship Project")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()

    return df

df = load_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# ==================================================
# PAGE 1
# ==================================================
if page == "Sales Overview":

    st.header("📊 Sales Overview")

    region = st.sidebar.selectbox(
        "Select Region",
        ["All"] + sorted(df["Region"].unique().tolist())
    )

    category = st.sidebar.selectbox(
        "Select Category",
        ["All"] + sorted(df["Category"].unique().tolist())
    )

    filtered_df = df.copy()

    if region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == region
        ]

    if category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == category
        ]

    # ---------------- KPI ----------------

    total_sales = filtered_df["Sales"].sum()

    total_orders = filtered_df["Order ID"].nunique()

    avg_sales = filtered_df["Sales"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Orders", total_orders)
    c3.metric("Average Sale", f"${avg_sales:.2f}")

    st.divider()

    # ---------------- Sales by Year ----------------

    yearly = (
        filtered_df
        .groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        yearly,
        x="Year",
        y="Sales",
        title="Total Sales by Year",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Monthly Trend ----------------

    monthly = (
        filtered_df
        .groupby("Order Date")["Sales"]
        .sum()
        .resample("ME")
        .sum()
        .reset_index()
    )

    fig2 = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Sales by Category ----------------

    cat = (
        filtered_df
        .groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig3 = px.pie(
        cat,
        names="Category",
        values="Sales",
        title="Sales by Category"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🌍 Sales by Region")

    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig4 = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        color="Region",
        text_auto=True,
        title="Sales by Region"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📄 Dataset Preview")

    st.dataframe(filtered_df.head(10))

    st.download_button(
        "📥 Download Filtered Data",
        filtered_df.to_csv(index=False),
        file_name="filtered_sales.csv",
        mime="text/csv"
    )

# ==================================================
# PAGE 2 - Forecast Explorer
# ==================================================
elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")
    st.write("Forecast future sales using the best-performing XGBoost model.")

    # Select forecast type
    forecast_type = st.selectbox(
        "Select Forecast Type",
        ["Category", "Region"]
    )

    if forecast_type == "Category":
        selected = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )
        data = df[df["Category"] == selected]

    else:
        selected = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )
        data = df[df["Region"] == selected]

    # Forecast horizon
    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    # Monthly Sales
    monthly = (
        data.groupby("Order Date")["Sales"]
        .sum()
        .resample("ME")
        .sum()
        .reset_index()
    )

    monthly.columns = ["Date", "Sales"]

    # Feature Engineering
    monthly["Month"] = monthly["Date"].dt.month
    monthly["Quarter"] = monthly["Date"].dt.quarter

    monthly["Lag1"] = monthly["Sales"].shift(1)
    monthly["Lag2"] = monthly["Sales"].shift(2)
    monthly["Lag3"] = monthly["Sales"].shift(3)
    monthly["RollingMean3"] = monthly["Sales"].rolling(3).mean()

    monthly = monthly.dropna()

    # Features
    features = [
        "Lag1",
        "Lag2",
        "Lag3",
        "RollingMean3",
        "Month",
        "Quarter"
    ]

    X = monthly[features]
    y = monthly["Sales"]

    X_train = X[:-3]
    X_test = X[-3:]

    y_train = y[:-3]
    y_test = y[-3:]

    # Train Model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    # Show only selected horizon
    forecast_df = pd.DataFrame({
        "Date": monthly["Date"].iloc[-3:].values,
        "Actual Sales": y_test.values,
        "Forecast Sales": prediction
    }).head(horizon)

    # Forecast Chart
    fig = px.line(
        forecast_df,
        x="Date",
        y=["Actual Sales", "Forecast Sales"],
        markers=True,
        title=f"{selected} Sales Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    mae = mean_absolute_error(
        y_test.iloc[:horizon],
        prediction[:horizon]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test.iloc[:horizon],
            prediction[:horizon]
        )
    )

    col1, col2 = st.columns(2)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")

    # Forecast Table
    st.subheader("Forecast Results")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    # Download Button
    st.download_button(
        "📥 Download Forecast",
        forecast_df.to_csv(index=False),
        file_name="forecast_results.csv",
        mime="text/csv"
    )

# ==================================================
# PAGE 3 - Anomaly Report
# ==================================================
elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Report")

    from sklearn.ensemble import IsolationForest

    weekly = (
        df.groupby("Order Date")["Sales"]
        .sum()
        .resample("W")
        .sum()
        .reset_index()
    )

    weekly.columns = ["Date", "Sales"]

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly["Anomaly"] = iso.fit_predict(
        weekly[["Sales"]]
    )

    anomalies = weekly[
        weekly["Anomaly"] == -1
    ]

    # Chart

    fig = px.line(
        weekly,
        x="Date",
        y="Sales",
        title="Weekly Sales with Anomalies"
    )

    fig.add_scatter(
        x=anomalies["Date"],
        y=anomalies["Sales"],
        mode="markers",
        marker=dict(
            color="red",
            size=10
        ),
        name="Anomaly"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies[["Date", "Sales"]],
        use_container_width=True
    )

    st.metric(
        "Total Anomalies",
        len(anomalies)
    )

    st.info(
        """
Possible reasons for anomalies:

• Festival sales

• Holiday discounts

• Stock shortages

• Bulk customer purchases

• Seasonal demand spikes
"""
    )

# ==================================================
# PAGE 4 - Demand Segments
# ==================================================
elif page == "Demand Segments":

    st.header("📦 Product Demand Segments")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import plotly.express as px

    # -------- Feature Engineering --------

    total_sales = df.groupby("Sub-Category")["Sales"].sum()

    avg_order = df.groupby("Sub-Category")["Sales"].mean()

    monthly = (
        df.groupby([
            "Sub-Category",
            pd.Grouper(key="Order Date", freq="ME")
        ])["Sales"]
        .sum()
        .reset_index()
    )

    volatility = monthly.groupby("Sub-Category")["Sales"].std()

    yearly = (
        df.groupby(["Sub-Category", "Year"])["Sales"]
        .sum()
        .reset_index()
    )

    yearly["Growth"] = yearly.groupby("Sub-Category")["Sales"].pct_change()

    growth_rate = yearly.groupby("Sub-Category")["Growth"].mean()

    cluster_df = pd.DataFrame({
        "Total Sales": total_sales,
        "Growth Rate": growth_rate,
        "Volatility": volatility,
        "Average Order Value": avg_order
    })

    cluster_df = cluster_df.fillna(0)

    # -------- Scaling --------

    scaler = StandardScaler()

    scaled = scaler.fit_transform(cluster_df)

    # -------- KMeans --------

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    cluster_df["Cluster"] = kmeans.fit_predict(scaled)

    # -------- PCA --------

    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled)

    cluster_df["PCA1"] = components[:,0]
    cluster_df["PCA2"] = components[:,1]

    labels = {
        0: "High Value, High Volatility",
        1: "Low Volume, Stable Demand",
        2: "High Volume, Stable Demand"
    }

    cluster_df["Demand Segment"] = cluster_df["Cluster"].map(labels)

    # -------- Plot --------

    fig = px.scatter(
        cluster_df,
        x="PCA1",
        y="PCA2",
        color="Demand Segment",
        text=cluster_df.index,
        title="Product Demand Segments"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Demand Segment Table")

    st.dataframe(
        cluster_df[[
            "Total Sales",
            "Growth Rate",
            "Volatility",
            "Average Order Value",
            "Demand Segment"
        ]],
        use_container_width=True
    )

    st.subheader("📦 Recommended Stocking Strategy")

    st.success("""
**High Volume, Stable Demand**
- Maintain higher inventory levels.
- Replenish stock frequently.

**High Value, High Volatility**
- Keep safety stock.
- Monitor demand regularly.

**Low Volume, Stable Demand**
- Maintain limited inventory.
- Reorder only when required.
""")

st.markdown("---")
st.caption("Developed by Swarup Karthik S | Xylofy AI Internship Project 2026")