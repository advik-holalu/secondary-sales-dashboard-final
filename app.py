import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Secondary Sales Dashboard", layout="wide")

# ------------------------------------------------------------
# TAB STYLE OVERRIDE (User Guide tab)
# ------------------------------------------------------------
st.markdown("""
<style>
    button[data-baseweb="tab"]:nth-child(6) {
        color: #1f77b4 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DASHBOARD TITLE (GLOBAL)
# ------------------------------------------------------------
st.markdown(
    """
    <div style="
        background-color:#f6892b;
        padding:18px 24px;
        border-radius:10px;
        margin-bottom:20px;
    ">
        <h1 style="
            color:white;
            margin:0;
            font-size:32px;
            font-weight:700;
        ">
            Secondary Sales Dashboard
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# LOAD PRE-AGGREGATED DATA 
# ------------------------------------------------------------

#tab 1 chart 1 load
@st.cache_data
def load_tab1_data():
    return pd.read_parquet("data_agg/tab1_category_trend.parquet")

df = load_tab1_data()

#tab 1 chart 2 load - donuts
@st.cache_data
def load_tab1_donut_data():
    return pd.read_parquet("data_agg/tab1_quarter_donut.parquet")

donut_df = load_tab1_donut_data()

#tab 1 chart 3 load - top skus
@st.cache_data
def load_tab1_top_sku_data():
    return pd.read_parquet("data_agg/tab1_top_skus_quarter.parquet")

sku_df = load_tab1_top_sku_data()

#tab 1 chart 4 load - state quarter
@st.cache_data
def load_tab1_state_quarter_data():
    return pd.read_parquet("data_agg/tab1_state_quarter.parquet")

state_q_df = load_tab1_state_quarter_data()

#tab 2 chart 1 load - 70% state month trend
@st.cache_data
def load_tab2_state_month_data():
    return pd.read_parquet("data_agg/tab2_state_month_trend.parquet")

tab2_df = load_tab2_state_month_data()

#tab 3 chart 1 load - gorwth vs laggards monthly
@st.cache_data
def load_tab3_monthly():
    return pd.read_parquet("data_agg/tab3_state_quarter_month_avg.parquet")

tab3_month_df = load_tab3_monthly()

#tab 3 chart 2 load - total all time
@st.cache_data
def load_tab3_totals():
    return pd.read_parquet("data_agg/tab3_state_total_alltime.parquet")

tab3_total_df = load_tab3_totals()

# TAB 4 LOADS — INDSUTRY VIEW
@st.cache_data
def load_tab4_data():
    return pd.read_parquet("data_agg/tab4_trend.parquet")

tab4_df = load_tab4_data()

# TAB 5 LOADS — PRODUCT TYPE DEEP DIVE
@st.cache_data
def load_tab5_ptype_month():
    return pd.read_parquet("data_agg/tab5_ptype_month.parquet")

@st.cache_data
def load_tab5_ptype_variant():
    return pd.read_parquet("data_agg/tab5_ptype_variant_month.parquet")

@st.cache_data
def load_tab5_ptype_base():
    return pd.read_parquet("data_agg/tab5_ptype_variant_month.parquet")

pt_df = load_tab5_ptype_base()
tab5_month_df = load_tab5_ptype_month()
tab5_variant_df = load_tab5_ptype_variant()

# Load once at top of file (not inside tab)
tab5_df = pd.read_parquet("data_agg/tab5_ptype_final.parquet")

# ------------------------------------------------------------
# GLOBAL FY FILTER (CANONICAL TIME MODEL)
# ------------------------------------------------------------
def _get_all_financial_years(*dfs):
    fys = set()
    for _df in dfs:
        if _df is None or len(_df) == 0:
            continue
        if "FinancialYear" in _df.columns:
            fys.update(_df["FinancialYear"].dropna().astype(str).unique().tolist())
    return sorted(fys)

ALL_FY = _get_all_financial_years(
    df,
    donut_df,
    sku_df,
    state_q_df,
    tab2_df,
    tab3_month_df,
    tab3_total_df,
    pt_df,
    tab5_month_df,
    tab5_variant_df,
)

if not ALL_FY:
    st.error("No FinancialYear values found in loaded parquet files.")
    st.stop()

# ------------------------------------------------------------
# GLOBAL FY FILTER (MAX 2)
# ------------------------------------------------------------
with st.sidebar:
    selected_fys = st.multiselect(
        "Financial Year",
        options=ALL_FY,
        default=[ALL_FY[-1]],
        key="global_financial_year",
    )

    # ---- POPUP STYLE (ONLY WHEN WRONG) ----
    if len(selected_fys) > 2:
        st.toast("Select maximum 2 Financial Years", icon="⚠️")
        st.stop()

    if not selected_fys:
        st.toast("Select at least 1 Financial Year", icon="⚠️")
        st.stop()

# ------------------------------------------------------------
# FY-SAFE HELPERS (USE EVERYWHERE)
# ------------------------------------------------------------
def filter_by_fy(dfin: pd.DataFrame, fys: list) -> pd.DataFrame:
    if dfin is None or dfin.empty:
        return dfin
    if "FinancialYear" not in dfin.columns:
        return dfin
    return dfin[dfin["FinancialYear"].isin(fys)].copy()

def get_month_order(dfin: pd.DataFrame):
    """
    Returns MonthLabel list sorted by FYMonthOrder (Apr->Mar).
    """
    if dfin is None or dfin.empty:
        return []
    if "FYMonthOrder" not in dfin.columns or "MonthLabel" not in dfin.columns:
        return []
    return (
        dfin[["FYMonthOrder", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("FYMonthOrder")["MonthLabel"]
        .astype(str)
        .tolist()
    )

def enforce_month_order(dfin: pd.DataFrame, month_order: list):
    """
    Converts MonthLabel to ordered categorical for plotting.
    """
    if dfin is None or dfin.empty or not month_order:
        return dfin
    if "MonthLabel" not in dfin.columns:
        return dfin
    out = dfin.copy()
    out["MonthLabel"] = pd.Categorical(
        out["MonthLabel"].astype(str),
        categories=month_order,
        ordered=True
    )
    return out

def add_fy_month_axis(dfin: pd.DataFrame) -> pd.DataFrame:
    if dfin is None or dfin.empty:
        return dfin

    out = dfin.copy()

    out["FYMonthKey"] = (
        out["FinancialYear"].astype(str)
        + "_"
        + out["FYMonthOrder"].astype(int).astype(str).str.zfill(2)
    )

    out["FYMonthLabel"] = (
        out["MonthLabel"].astype(str)
        + " "
        + out["FinancialYear"].astype(str)
    )

    return out

# ------------------------------------------------------------
# GLOBAL NUMBER FORMATTERS (INDIAN: Lakhs / Crores)
# ------------------------------------------------------------
def format_indian(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "0"
    try:
        v = float(v)
    except Exception:
        return "0"

    if v >= 1e7:
        return f"{v/1e7:.{decimals}f} Cr"
    elif v >= 1e5:
        return f"{v/1e5:.{decimals}f} L"
    elif v >= 1e3:
        return f"{v/1e3:.{decimals}f} K"
    else:
        return f"{v:.{decimals}f}"

def format_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "0.0%"
    try:
        v = float(v)
    except Exception:
        return "0.0%"
    return f"{v:.{decimals}f}%"

# ------------------------------------------------------------
# GLOBAL LINE-CHART LABEL STYLING
# ------------------------------------------------------------
def apply_line_label_style(fig, text_size=13):
    print("🔥 APPLY_LINE_LABEL_STYLE CALLED")

    for trace in fig.data:
        print("  → trace:", trace.name)

        trace.textfont = dict(
            color=trace.line.color,
            size=18  # VERY BIG on purpose
        )
        trace.textposition = "top center"
        trace.hoverinfo = "skip"

    return fig

# ------------------------------------------------------------
# GLOBAL FY DIVIDER (APPLIES TO ALL CHARTS)
# ------------------------------------------------------------
def add_fy_divider(fig, df, selected_fys):
    if len(selected_fys) != 2:
        return fig

    if "FYMonthKey" not in df.columns or "FinancialYear" not in df.columns:
        return fig

    second_fy = sorted(selected_fys)[1]

    x_order = df["FYMonthKey"].drop_duplicates().tolist()

    second_fy_points = df[df["FinancialYear"] == second_fy]["FYMonthKey"]

    if second_fy_points.empty:
        return fig

    first_point_second_fy = second_fy_points.iloc[0]

    if first_point_second_fy not in x_order:
        return fig

    idx = x_order.index(first_point_second_fy)
    divider_x = idx - 0.5

    fig.add_vline(
        x=divider_x,
        line_width=0.6,
        line_dash="dash",
        line_color="white",
        opacity=0.5
    )

    return fig

# ============================================================
# DEFINE DASHBOARD TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sales Overview",
    "Top Markets",
    "Growth vs Laggards",
    "Industry View",
    "Product Type Deep Dive",
    "User Guide"
])

# ============================================================
# TAB 1 — SALES OVERVIEW (FY SAFE + MULTI FY FIXED)
# ============================================================
with tab1:

    # ------------------------------------------------------------
    # SIDEBAR FILTERS
    # ------------------------------------------------------------
    with st.sidebar:
        st.header("Sales Overview Filters")

        metric = st.radio("Metric", ["Revenue", "GMV"], index=0)

        region_sel = st.multiselect(
            "Region",
            sorted(df["Region Name"].dropna().unique())
        )

        if region_sel:
            state_opts = sorted(
                df[df["Region Name"].isin(region_sel)]["State Name"].dropna().unique()
            )
        else:
            state_opts = sorted(df["State Name"].dropna().unique())

        state_sel = st.multiselect("State", state_opts)

        parent_cat_sel = st.multiselect(
            "Parent Category",
            sorted(df["Parent Category"].dropna().unique())
        )

        if parent_cat_sel:
            l1_opts = sorted(
                df[df["Parent Category"].isin(parent_cat_sel)]["L1 Category"].dropna().unique()
            )
        else:
            l1_opts = sorted(df["L1 Category"].dropna().unique())

        l1_sel = st.multiselect("L1 Category", l1_opts)

        if l1_sel:
            l2_opts = sorted(
                df[df["L1 Category"].isin(l1_sel)]["L2 Category"].dropna().unique()
            )
        else:
            l2_opts = sorted(df["L2 Category"].dropna().unique())

        l2_sel = st.multiselect("L2 Category", l2_opts)

        if l2_sel:
            l3_opts = sorted(
                df[df["L2 Category"].isin(l2_sel)]["L3 Category"].dropna().unique()
            )
        else:
            l3_opts = sorted(df["L3 Category"].dropna().unique())

        l3_sel = st.multiselect("L3 Category", l3_opts)

        platform_sel = st.multiselect(
            "Platform",
            sorted(df["Platform"].dropna().unique())
        )

    # ------------------------------------------------------------
    # FILTER FUNCTION
    # ------------------------------------------------------------
    def apply_tab1_filters(dfin):
        out = dfin.copy()

        if region_sel:
            out = out[out["Region Name"].isin(region_sel)]
        if state_sel:
            out = out[out["State Name"].isin(state_sel)]
        if parent_cat_sel:
            out = out[out["Parent Category"].isin(parent_cat_sel)]
        if l1_sel:
            out = out[out["L1 Category"].isin(l1_sel)]
        if l2_sel:
            out = out[out["L2 Category"].isin(l2_sel)]
        if l3_sel:
            out = out[out["L3 Category"].isin(l3_sel)]
        if platform_sel:
            out = out[out["Platform"].isin(platform_sel)]

        return out

    # ------------------------------------------------------------
    # APPLY FILTERS
    # ------------------------------------------------------------
    df_filt = apply_tab1_filters(df)
    df_filt = df_filt[df_filt["FinancialYear"].isin(selected_fys)]

    if df_filt.empty:
        st.warning("No data for selected filters.")
        st.stop()

    month_order = get_month_order(df_filt)

    # ============================================================
    # 1. CATEGORY TREND
    # ============================================================
    st.title("Secondary Sales Overview")
    st.subheader("Parent Category-wise Trend")

    timeline = (
        df_filt
        .groupby(
            ["FinancialYear", "FYMonthOrder", "MonthLabel", "Parent Category"],
            as_index=False
        )[metric]
        .sum()
    )

    timeline = enforce_month_order(timeline, month_order)
    timeline = add_fy_month_axis(timeline)
    timeline = timeline.sort_values("FYMonthKey")

    fig = px.line(
        timeline,
        x="FYMonthKey",
        y=metric,
        color="Parent Category",
        markers=True,
        text=timeline[metric].apply(format_indian)
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=timeline["FYMonthKey"],
        ticktext=timeline["MonthLabel"]
    )

    y_max = timeline[metric].max()
    y_ticks = np.linspace(0, y_max, 6)

    fig.update_yaxes(
        tickvals=y_ticks,
        ticktext=[format_indian(v, 0) for v in y_ticks]
    )

    fig = add_fy_divider(fig, timeline, selected_fys)

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 1B. YoY COMPARISON (MONTH-WISE)
    # ============================================================
    st.subheader("YoY Comparison — Month-wise")

    # ----------------------------
    # Month selector
    # ----------------------------
    month_options = (
        timeline[["FYMonthOrder", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("FYMonthOrder")
    )

    selected_month = st.selectbox(
        "Select Month",
        options=month_options["MonthLabel"].tolist(),
        index=0,
        key="yoy_month_selector"
    )

    # ----------------------------
    # Filter ONLY that month
    # ----------------------------
    yoy_df = timeline[timeline["MonthLabel"] == selected_month]

    if yoy_df.empty:
        st.info("No data available for selected month.")
    else:

        # ----------------------------
        # Aggregate FY-wise
        # ----------------------------
        yoy_df = (
            yoy_df
            .groupby(["FinancialYear", "Parent Category"], as_index=False)[metric]
            .sum()
        )

        # ----------------------------
        # Plot
        # ----------------------------
        fig = px.line(
            yoy_df,
            x="FinancialYear",
            y=metric,
            color="Parent Category",
            markers=True,
            text=yoy_df[metric].apply(format_indian)
        )

        fig.update_traces(textposition="top center")

        # ----------------------------
        # Y-axis formatting
        # ----------------------------
        y_max = yoy_df[metric].max()
        y_ticks = np.linspace(0, y_max, 6)

        fig.update_yaxes(
            tickvals=y_ticks,
            ticktext=[format_indian(v, 0) for v in y_ticks]
        )

        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 2. DONUTS (FINAL — MULTI FY + PARTIAL QUARTER SAFE)
    # ============================================================
    st.subheader("Sales Distribution — Quarter-wise")

    # ------------------------
    # Apply filters
    # ------------------------
    donut_base = apply_tab1_filters(donut_df)
    donut_base = donut_base[donut_base["FinancialYear"].isin(selected_fys)]

    # ------------------------
    # ALWAYS derive quarters from FULL data (not filtered)
    # ------------------------
    fy_quarter_pairs = (
        donut_df[donut_df["FinancialYear"].isin(selected_fys)]
        [["FinancialYear", "FYQuarter"]]
        .dropna()
        .drop_duplicates()
    )

    # correct ordering
    order_map = {"Q1":1, "Q2":2, "Q3":3, "Q4":4}
    fy_quarter_pairs["sort"] = fy_quarter_pairs["FYQuarter"].map(order_map)

    fy_quarter_pairs = fy_quarter_pairs.sort_values(
        ["FinancialYear", "sort"]
    )

    # ------------------------
    # Render donuts
    # ------------------------
    for _, row in fy_quarter_pairs.iterrows():

        fy = row["FinancialYear"]
        q = row["FYQuarter"]

        # try filtered data first
        dfq = donut_base[
            (donut_base["FinancialYear"] == fy) &
            (donut_base["FYQuarter"] == q)
        ]

        # fallback to full data if filters remove it
        if dfq.empty:
            dfq = donut_df[
                (donut_df["FinancialYear"] == fy) &
                (donut_df["FYQuarter"] == q)
            ]

        # if still empty → skip (true no-data case)
        if dfq.empty:
            continue

        st.markdown(f"### {fy} — {q}")

        col1, col2 = st.columns(2)

        # ------------------------
        # Region donut
        # ------------------------
        with col1:
            reg = (
                dfq.groupby("Region Name", as_index=False)[metric]
                .sum()
                .sort_values(metric, ascending=False)
            )

            fig_reg = px.pie(
                reg,
                names="Region Name",
                values=metric,
                hole=0.5,
            )

            fig_reg.update_traces(
                textinfo="percent+label",
                textfont_size=12
            )

            st.plotly_chart(fig_reg, use_container_width=True)

        # ------------------------
        # State donut
        # ------------------------
        with col2:
            stt = (
                dfq.groupby("State Name", as_index=False)[metric]
                .sum()
                .sort_values(metric, ascending=False)
            )

            fig_state = px.pie(
                stt,
                names="State Name",
                values=metric,
                hole=0.5,
            )

            fig_state.update_traces(
                textinfo="percent",
                textfont_size=11
            )

            st.plotly_chart(fig_state, use_container_width=True)

        st.markdown("---")

    # ============================================================
    # 3. TOP SKUS (FIXED MULTI FY)
    # ============================================================
    st.subheader(f"Top 10 SKUs — Quarter-wise ({metric})")

    sku_base = apply_tab1_filters(sku_df)
    sku_base = sku_base[sku_base["FinancialYear"].isin(selected_fys)]

    fy_quarter_pairs = (
        sku_base[["FinancialYear","FYQuarter"]]
        .dropna()
        .drop_duplicates()
    )

    fy_quarter_pairs["sort"] = fy_quarter_pairs["FYQuarter"].map(order_map)
    fy_quarter_pairs = fy_quarter_pairs.sort_values(["FinancialYear","sort"])

    for _, row in fy_quarter_pairs.iterrows():

        fy = row["FinancialYear"]
        q = row["FYQuarter"]

        dfq = sku_base[
            (sku_base["FinancialYear"] == fy) &
            (sku_base["FYQuarter"] == q)
        ]

        if dfq.empty:
            continue

        st.markdown(f"### {fy} — {q}")

        dfq = (
            dfq.groupby(
                ["L3 Category","Parent Category","L1 Category"],
                as_index=False
            )[metric]
            .sum()
            .sort_values(metric, ascending=False)
            .head(10)
        )

        dfq = dfq.rename(columns={"L3 Category":"Normalised Item Name"})
        dfq[f"{metric} (₹)"] = dfq[metric].apply(format_indian)

        st.dataframe(
            dfq[
                ["Normalised Item Name","Parent Category","L1 Category",f"{metric} (₹)"]
            ],
            use_container_width=True
        )

        st.markdown("---")

    # ============================================================
    # 4. STATE PERFORMANCE (SINGLE TABLE — MULTI FY)
    # ============================================================
    st.subheader("State Performance — Quarter-wise")

    df_state = apply_tab1_filters(state_q_df)
    df_state = df_state[df_state["FinancialYear"].isin(selected_fys)]

    if df_state.empty:
        st.info("No data available for selected filters.")
        st.stop()

    # --- Create combined column: FY + Quarter ---
    df_state["FY_Q"] = df_state["FinancialYear"] + " " + df_state["FYQuarter"]

    # --- Define correct order ---
    order_map = {"Q1":1,"Q2":2,"Q3":3,"Q4":4}

    df_state["q_order"] = df_state["FYQuarter"].map(order_map)

    # sort properly
    df_state = df_state.sort_values(["FinancialYear", "q_order"])

    # --- Pivot ---
    pivot = (
        df_state
        .groupby(["State Name", "FY_Q"], as_index=False)[metric]
        .sum()
        .pivot(index="State Name", columns="FY_Q", values=metric)
        .fillna(0)
    )

    # --- Ensure correct column order ---
    ordered_cols = (
        df_state[["FY_Q","FinancialYear","q_order"]]
        .drop_duplicates()
        .sort_values(["FinancialYear","q_order"])["FY_Q"]
        .tolist()
    )

    pivot = pivot[ordered_cols]

    # --- Format ---
    for col in pivot.columns:
        pivot[col] = pivot[col].apply(format_indian)

    # --- Final table ---
    pivot = pivot.reset_index()
    pivot.insert(0, "Sl. No.", range(1, len(pivot)+1))

    st.dataframe(pivot, use_container_width=True)
    
# ============================================================
# TAB 2 — TOP MARKETS (FY SAFE)
# ============================================================
with tab2:
    st.title("Top Markets — State Trends (Top 70% Contribution)")

    # --------------------------------------------------------
    # SIDEBAR FILTERS (TAB 2)
    # --------------------------------------------------------
    with st.sidebar:
        st.header("Top Markets Filters")

        metric_tab2 = st.radio(
            "Metric (Tab 2)",
            ["Revenue", "GMV"],
            index=0,
            key="metric_tab2"
        )

        parent_cat_sel = st.multiselect(
            "Parent Category (Tab 2)",
            sorted(tab2_df["Parent Category"].dropna().unique()),
            key="parent_tab2"
        )

        if parent_cat_sel:
            l1_opts = sorted(
                tab2_df[tab2_df["Parent Category"].isin(parent_cat_sel)]["L1 Category"]
                .dropna().unique()
            )
        else:
            l1_opts = sorted(tab2_df["L1 Category"].dropna().unique())

        l1_sel = st.multiselect("L1 Category (Tab 2)", l1_opts, key="l1_tab2")

        if l1_sel:
            l2_opts = sorted(
                tab2_df[tab2_df["L1 Category"].isin(l1_sel)]["L2 Category"]
                .dropna().unique()
            )
        else:
            l2_opts = sorted(tab2_df["L2 Category"].dropna().unique())

        l2_sel = st.multiselect("L2 Category (Tab 2)", l2_opts, key="l2_tab2")

        if l2_sel:
            l3_opts = sorted(
                tab2_df[tab2_df["L2 Category"].isin(l2_sel)]["L3 Category"]
                .dropna().unique()
            )
        else:
            l3_opts = sorted(tab2_df["L3 Category"].dropna().unique())

        l3_sel = st.multiselect("L3 Category (Tab 2)", l3_opts, key="l3_tab2")

        platform_sel = st.multiselect(
            "Platform (Tab 2)",
            sorted(tab2_df["Platform"].dropna().unique()),
            key="platform_tab2"
        )

        baseline_q = st.selectbox(
            "Baseline Quarter",
            sorted(tab2_df["FYQuarter"].dropna().unique()),
            key="baseline_q_tab2"
        )

    # --------------------------------------------------------
    # APPLY FILTERS + FY FILTER
    # --------------------------------------------------------
    df2 = tab2_df.copy()
    df2 = df2[df2["FinancialYear"].isin(selected_fys)]

    if parent_cat_sel:
        df2 = df2[df2["Parent Category"].isin(parent_cat_sel)]
    if l1_sel:
        df2 = df2[df2["L1 Category"].isin(l1_sel)]
    if l2_sel:
        df2 = df2[df2["L2 Category"].isin(l2_sel)]
    if l3_sel:
        df2 = df2[df2["L3 Category"].isin(l3_sel)]
    if platform_sel:
        df2 = df2[df2["Platform"].isin(platform_sel)]

    if df2.empty:
        st.warning("No data available for selected filters.")
        st.stop()

    # --------------------------------------------------------
    # TOP 70% STATES
    # --------------------------------------------------------
    def get_top70_states(dfin, metric):
        state_tot = (
            dfin.groupby("State Name", as_index=False, observed=False)[metric]
            .sum()
            .sort_values(metric, ascending=False)
        )
        total = state_tot[metric].sum()
        state_tot["CumShare"] = state_tot[metric].cumsum() / total * 100
        top_states = state_tot[state_tot["CumShare"] <= 70]["State Name"].tolist()
        return top_states or state_tot.head(1)["State Name"].tolist()

    baseline_df = df2[df2["FYQuarter"] == baseline_q]
    top_states = get_top70_states(baseline_df, metric_tab2)

    plot_df = df2[df2["State Name"].isin(top_states)]

    # --------------------------------------------------------
    # MONTH ORDER
    # --------------------------------------------------------
    month_order = get_month_order(plot_df)

    trend_df = (
        plot_df
        .groupby(
            ["FinancialYear", "FYMonthOrder", "MonthLabel", "State Name"],
            as_index=False,
            observed=False
        )[metric_tab2]
        .sum()
    )

    trend_df = enforce_month_order(trend_df, month_order)
    trend_df = add_fy_month_axis(trend_df)

    # ✅ FIX: correct ordering
    trend_df = trend_df.sort_values("FYMonthKey")

    # --------------------------------------------------------
    # LEGEND ORDER
    # --------------------------------------------------------
    state_order = (
        trend_df
        .groupby("State Name", as_index=False, observed=False)[metric_tab2]
        .sum()
        .sort_values(metric_tab2, ascending=False)["State Name"]
        .tolist()
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------
    fig = px.line(
        trend_df,
        x="FYMonthKey",
        y=metric_tab2,
        color="State Name",
        markers=True,
        text=trend_df[metric_tab2].apply(format_indian),
        category_orders={
            "FYMonthKey": trend_df["FYMonthKey"].drop_duplicates().tolist(),
            "State Name": state_order
        }
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=trend_df["FYMonthKey"],
        ticktext=trend_df["MonthLabel"]
    )

    fig.update_traces(textposition="top center")

    y_max = trend_df[metric_tab2].max()
    y_ticks = np.linspace(0, y_max, 6)

    fig.update_yaxes(
        tickvals=y_ticks,
        ticktext=[format_indian(v, decimals=0) for v in y_ticks]
    )

    fig = add_fy_divider(fig, trend_df, selected_fys)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "States are selected based on the chosen baseline quarter contributing the first "
        "70% of sales and are tracked consistently across all months."
    )

# ============================================================
# TAB 3 — GROWTH VS LAGGARDS (FY SAFE)
# ============================================================
with tab3:
    st.title("Growth vs Laggard Markets (Top 70% Contribution)")

    # ----------------------------
    # SIDEBAR FILTERS (TAB 3)
    # ----------------------------
    with st.sidebar:
        st.header("Growth vs Laggards Filters")

        metric = st.radio("Metric (Tab 3)", ["Revenue", "GMV"], index=0)

        available_quarters = sorted(tab3_month_df["FYQuarter"].dropna().unique())

        compare_q = st.selectbox(
            "Compare Quarter",
            available_quarters,
            index=len(available_quarters) - 1
        )

        baseline_q = st.selectbox(
            "Baseline Quarter",
            [q for q in available_quarters if q != compare_q],
            index=0
        )

        parent_sel = st.multiselect(
            "Parent Category (Tab 3)",
            sorted(tab3_month_df["Parent Category"].dropna().unique())
        )

        l1_sel = st.multiselect(
            "L1 Category (Tab 3)",
            sorted(tab3_month_df["L1 Category"].dropna().unique())
        )

        l2_sel = st.multiselect(
            "L2 Category (Tab 3)",
            sorted(tab3_month_df["L2 Category"].dropna().unique())
        )

        l3_sel = st.multiselect(
            "L3 Category (Tab 3)",
            sorted(tab3_month_df["L3 Category"].dropna().unique())
        )

        platform_sel = st.multiselect(
            "Platform (Tab 3)",
            sorted(tab3_month_df["Platform"].dropna().unique())
        )

    # ----------------------------
    # APPLY FILTERS + FY FILTER
    # ----------------------------
    df_month = tab3_month_df[tab3_month_df["FinancialYear"].isin(selected_fys)].copy()
    df_total = tab3_total_df[tab3_total_df["FinancialYear"].isin(selected_fys)].copy()

    if parent_sel:
        df_month = df_month[df_month["Parent Category"].isin(parent_sel)]
        df_total = df_total[df_total["Parent Category"].isin(parent_sel)]
    if l1_sel:
        df_month = df_month[df_month["L1 Category"].isin(l1_sel)]
        df_total = df_total[df_total["L1 Category"].isin(l1_sel)]
    if l2_sel:
        df_month = df_month[df_month["L2 Category"].isin(l2_sel)]
        df_total = df_total[df_total["L2 Category"].isin(l2_sel)]
    if l3_sel:
        df_month = df_month[df_month["L3 Category"].isin(l3_sel)]
        df_total = df_total[df_total["L3 Category"].isin(l3_sel)]
    if platform_sel:
        df_month = df_month[df_month["Platform"].isin(platform_sel)]
        df_total = df_total[df_total["Platform"].isin(platform_sel)]

    if df_month.empty or df_total.empty:
        st.warning("No data for selected filters.")
        st.stop()

    # ----------------------------
    # TOP 70% STATES
    # ----------------------------
    baseline_df = df_month[df_month["FYQuarter"] == baseline_q]

    state_rank = (
        baseline_df
        .groupby("State Name", as_index=False, observed=False)[metric]
        .sum()
        .sort_values(metric, ascending=False)
    )

    total_val = state_rank[metric].sum()
    state_rank["CumPct"] = state_rank[metric].cumsum() / total_val

    top_states = state_rank[state_rank["CumPct"] <= 0.70]["State Name"].tolist()
    if not top_states:
        top_states = state_rank.head(1)["State Name"].tolist()

    df_month = df_month[df_month["State Name"].isin(top_states)]

    # ----------------------------
    # BASELINE vs COMPARE
    # ----------------------------
    base_avg = (
        df_month[df_month["FYQuarter"] == baseline_q]
        .groupby("State Name", as_index=False, observed=False)[metric]
        .mean()
        .rename(columns={metric: "Baseline"})
    )

    comp_avg = (
        df_month[df_month["FYQuarter"] == compare_q]
        .groupby("State Name", as_index=False, observed=False)[metric]
        .mean()
        .rename(columns={metric: "Compare"})
    )

    growth_df = base_avg.merge(comp_avg, on="State Name", how="outer").fillna(0)

    growth_df["Growth %"] = (
        (growth_df["Compare"] - growth_df["Baseline"])
        / growth_df["Baseline"].replace(0, np.nan) * 100
    )

    # ----------------------------
    # SPLIT
    # ----------------------------
    growth_pos = (
        growth_df[growth_df["Growth %"] > 0]
        .sort_values("Growth %", ascending=False)
        .head(5)
    )

    growth_neg = (
        growth_df[growth_df["Growth %"] < 0]
        .sort_values("Growth %")
        .head(5)
    )

    # ----------------------------
    # VISUALS
    # ----------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"Top Growth — {compare_q} vs {baseline_q}")
        if not growth_pos.empty:
            fig = px.bar(
                growth_pos,
                x="Growth %",
                y="State Name",
                orientation="h",
                text=growth_pos["Growth %"].apply(format_pct)
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader(f"Top Laggards — {compare_q} vs {baseline_q}")
        if not growth_neg.empty:
            fig = px.bar(
                growth_neg,
                x="Growth %",
                y="State Name",
                orientation="h",
                text=growth_neg["Growth %"].apply(format_pct)
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # DRILL-DOWN
    # ----------------------------
    st.markdown("---")
    st.subheader("Monthly Trends — Drill-down")

    month_order = get_month_order(df_month)

    # -------- GROWTH --------
    st.markdown("### Growth States — Monthly Trend")

    growth_states = growth_pos["State Name"].tolist()

    if growth_states:
        selected_growth_states = st.multiselect(
            "Select Growth States",
            options=growth_states,
            default=growth_states,
            key="growth_state_selector"
        )

        growth_trend = (
            df_month[df_month["State Name"].isin(selected_growth_states)]
            .groupby(
                ["FinancialYear", "FYMonthOrder", "MonthLabel", "State Name"],
                as_index=False,
                observed=False
            )[metric]
            .sum()
        )

        growth_trend = enforce_month_order(growth_trend, month_order)
        growth_trend = add_fy_month_axis(growth_trend)
        growth_trend = growth_trend.sort_values("FYMonthKey")

        fig = px.line(
            growth_trend,
            x="FYMonthKey",
            y=metric,
            color="State Name",
            markers=True,
            text=growth_trend[metric].apply(format_indian),
            category_orders={"FYMonthKey": growth_trend["FYMonthKey"].drop_duplicates().tolist()}
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=growth_trend["FYMonthKey"],
            ticktext=growth_trend["MonthLabel"]
        )

        fig = add_fy_divider(fig, growth_trend, selected_fys)
        st.plotly_chart(fig, use_container_width=True)

    # -------- LAGGARD --------
    st.markdown("### Laggard States — Monthly Trend")

    laggard_states = growth_neg["State Name"].tolist()

    if laggard_states:
        selected_laggard_states = st.multiselect(
            "Select Laggard States",
            options=laggard_states,
            default=laggard_states,
            key="laggard_state_selector"
        )

        laggard_trend = (
            df_month[df_month["State Name"].isin(selected_laggard_states)]
            .groupby(
                ["FinancialYear", "FYMonthOrder", "MonthLabel", "State Name"],
                as_index=False,
                observed=False
            )[metric]
            .sum()
        )

        laggard_trend = enforce_month_order(laggard_trend, month_order)
        laggard_trend = add_fy_month_axis(laggard_trend)
        laggard_trend = laggard_trend.sort_values("FYMonthKey")

        fig = px.line(
            laggard_trend,
            x="FYMonthKey",
            y=metric,
            color="State Name",
            markers=True,
            text=laggard_trend[metric].apply(format_indian),
            category_orders={"FYMonthKey": laggard_trend["FYMonthKey"].drop_duplicates().tolist()}
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=laggard_trend["FYMonthKey"],
            ticktext=laggard_trend["MonthLabel"]
        )

        fig = add_fy_divider(fig, laggard_trend, selected_fys)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4 — INDUSTRY VIEW (FIXED FOR MULTI FY)
# ============================================================
with tab4:

    st.title("Industry View")

    # ----------------------------
    # SIDEBAR FILTERS
    # ----------------------------
    with st.sidebar:
        st.header("Industry Filters")

        metric_tab4 = st.radio(
            "Metric",
            ["GMV", "SP"],
            index=0,
            key="tab4_metric"
        )

        platform_options = sorted(tab4_df["Platform"].dropna().unique())
        city_options = sorted(tab4_df["City Name"].dropna().unique())
        category_options = sorted(tab4_df["Parent Category"].dropna().unique())

        default_platform = ["Blinkit"] if "Blinkit" in platform_options else []
        default_cities = [c for c in ["PAN India", "Bengaluru-Metro", "Mumbai-Metro"] if c in city_options]
        default_category = ["Indian Sweets"] if "Indian Sweets" in category_options else []

        platform_sel = st.multiselect("Platform", platform_options, default=default_platform)
        city_sel = st.multiselect("Region", city_options, default=default_cities)
        category_sel = st.multiselect("Parent Category", category_options, default=default_category)

    # ----------------------------
    # APPLY FILTERS
    # ----------------------------
    df4 = tab4_df.copy()
    df4 = df4[df4["FinancialYear"].isin(selected_fys)]

    if platform_sel:
        df4 = df4[df4["Platform"].isin(platform_sel)]
    if city_sel:
        df4 = df4[df4["City Name"].isin(city_sel)]
    if category_sel:
        df4 = df4[df4["Parent Category"].isin(category_sel)]

    if df4.empty:
        st.warning("No data for selected filters.")
        st.stop()

    month_order = get_month_order(df4)

    # ----------------------------
    # METRIC SWITCH
    # ----------------------------
    if metric_tab4 == "GMV":
        industry_col = "Industry_Size_GMV"
        godesi_col = "GO_DESI_GMV"
        share_col = "Market_Share_GMV"
    else:
        industry_col = "Industry_Size_SP"
        godesi_col = "GO_DESI_SP"
        share_col = "Market_Share_SP"

    # ============================================================
    # GRAPH 1 — MARKET SHARE
    # ============================================================
    st.subheader("GO DESi Market Share Trend (%)")

    share_df = (
        df4
        .groupby(["FinancialYear","FYMonthOrder","MonthLabel","City Name"], as_index=False)
        [[godesi_col, industry_col]]
        .sum()
    )

    share_df[share_col] = np.where(
        share_df[industry_col] > 0,
        (share_df[godesi_col] / share_df[industry_col]) * 100,
        0
    )

    share_df = enforce_month_order(share_df, month_order)
    share_df = add_fy_month_axis(share_df)
    share_df = share_df.sort_values("FYMonthKey")

    fig = px.line(
        share_df,
        x="FYMonthKey",
        y=share_col,
        color="City Name",
        markers=True,
        text=share_df[share_col].apply(format_pct),
        category_orders={"FYMonthKey": share_df["FYMonthKey"].drop_duplicates().tolist()}
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=share_df["FYMonthKey"],
        ticktext=share_df["MonthLabel"]
    )

    fig = add_fy_divider(fig, share_df, selected_fys)
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # GRAPH 2 — BAR (NO CHANGE)
    # ============================================================
    st.subheader("Industry Size vs GO DESi")

    bar_df = (
        df4
        .groupby(["FinancialYear","FYMonthOrder","MonthLabel"], as_index=False)
        [[industry_col, godesi_col]]
        .sum()
    )

    bar_df = enforce_month_order(bar_df, month_order)
    bar_df = add_fy_month_axis(bar_df)
    bar_df = bar_df.sort_values("FYMonthKey")

    bar_df_melt = bar_df.melt(
        id_vars=["FYMonthKey","MonthLabel"],
        value_vars=[industry_col, godesi_col],
        var_name="Type",
        value_name="Value"
    )

    bar_df_melt["Type"] = bar_df_melt["Type"].map({
        industry_col: "Industry Size",
        godesi_col: "GO DESi"
    })

    fig = px.bar(
        bar_df_melt,
        x="FYMonthKey",
        y="Value",
        color="Type",
        barmode="group",
        text=bar_df_melt["Value"].apply(format_indian),
        category_orders={"FYMonthKey": bar_df["FYMonthKey"].drop_duplicates().tolist()}
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=bar_df["FYMonthKey"],
        ticktext=bar_df["MonthLabel"]
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # GRAPH 3 — INDUSTRY TREND
    # ============================================================
    st.subheader("Industry Size Trend")

    ind_df = (
        df4
        .groupby(["FinancialYear","FYMonthOrder","MonthLabel","City Name"], as_index=False)[industry_col]
        .sum()
    )

    ind_df = enforce_month_order(ind_df, month_order)
    ind_df = add_fy_month_axis(ind_df)
    ind_df = ind_df.sort_values("FYMonthKey")

    fig = px.line(
        ind_df,
        x="FYMonthKey",
        y=industry_col,
        color="City Name",
        markers=True,
        text=ind_df[industry_col].apply(format_indian),
        category_orders={"FYMonthKey": ind_df["FYMonthKey"].drop_duplicates().tolist()}
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=ind_df["FYMonthKey"],
        ticktext=ind_df["MonthLabel"]
    )

    fig = add_fy_divider(fig, ind_df, selected_fys)
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # GRAPH 4 — GO DESi TREND
    # ============================================================
    st.subheader("GO DESi Sales Trend")

    gd_df = (
        df4
        .groupby(["FinancialYear","FYMonthOrder","MonthLabel","City Name"], as_index=False)[godesi_col]
        .sum()
    )

    gd_df = enforce_month_order(gd_df, month_order)
    gd_df = add_fy_month_axis(gd_df)
    gd_df = gd_df.sort_values("FYMonthKey")

    fig = px.line(
        gd_df,
        x="FYMonthKey",
        y=godesi_col,
        color="City Name",
        markers=True,
        text=gd_df[godesi_col].apply(format_indian),
        category_orders={"FYMonthKey": gd_df["FYMonthKey"].drop_duplicates().tolist()}
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=gd_df["FYMonthKey"],
        ticktext=gd_df["MonthLabel"]
    )

    fig = add_fy_divider(fig, gd_df, selected_fys)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # CITY MAPPINGS (FIXED - NO NESTING)
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("City Mappings")

    # ---------- PLATFORM SELECT ----------
    platform_map_choice = st.selectbox(
        "Select Platform",
        ["Blinkit", "Instamart", "Zepto"]
    )

    # ---------- DATA ----------
    city_maps = {
        "Blinkit": {
            "Bengaluru-Metro": ["Bangalore", "Bengaluru"],
            "South-T2": ["Kochi", "Vijayawada", "Visakhapatnam", "Vizag", "Guntur"],
            "West-T2": ["Bhopal", "Goa", "Gwalior", "Indore", "Jaipur", "Jodhpur", "Kota", "Rajkot", "Surat", "Vadodara"],
            "North-T2": ["Agra", "Amritsar", "Bareilly", "Chandigarh", "Dehradun", "Faridabad", "Ghaziabad", "Gurgaon", "Gurugram", "Jalandhar", "Kanpur", "Lucknow", "Ludhiana", "Meerut", "Mohali", "Noida", "Patiala", "Varanasi"],
            "East-T2": ["Durgapur", "Jamshedpur", "Ranchi"],
            "North-T3": ["Bahadurgarh", "Bathinda", "Haridwar", "Kharar", "Panchkula", "Phagwara", "Rohtak", "Roorkee", "Sonipat", "Zirakpur"],
            "Others": ["Bombay", "HR-NCR", "NorthGoa", "SouthGoa", "UP-NCR"]
        },
        "Instamart": {
            "Bengaluru-Metro": ["Bangalore", "Bengaluru"],
            "South-T2": ["Coimbatore", "Guntur", "Kochi", "Kozhikode", "Pondicherry", "Thiruvananthapuram", "Vijayawada", "Vizag"],
            "West-T2": ["Bhopal", "Central Goa", "Indore", "Jaipur", "Nagpur", "Nashik", "Rajkot", "Surat", "Vadodara", "Goa"],
            "North-T2": ["Amritsar", "Chandigarh", "Dehradun", "Faridabad", "Ghaziabad", "Gurgaon", "Gurugram", "Kanpur", "Lucknow", "Ludhiana", "Noida", "Noida 1", "Varanasi", "Mohali"],
            "South-T3": ["Mangaluru", "Mysore", "Salem", "Thrissur", "Tirupati", "Trichy", "Warangal"],
            "East-T2": ["Bhubaneswar", "Ranchi"],
            "North-T3": ["Panchkula", "Zirakpur"]
        },
        "Zepto": {
            "Bengaluru-Metro": ["Bangalore", "Bengaluru"],
            "South-T2": ["Coimbatore", "Kochi"],
            "West-T2": ["Jaipur", "Nashik"],
            "North-T2": ["Chandigarh", "Mohali", "Faridabad", "Ghaziabad", "Gurgaon", "Gurugram", "Lucknow", "Noida", "SAS Nagar"]
        }
    }

    # ---------- DISPLAY (CLEAN UI) ----------
    selected_map = city_maps.get(platform_map_choice, {})

    # 3 columns layout for neat grid
    cols = st.columns(3)

    i = 0
    for grouping, cities in selected_map.items():
        if not cities:
            continue

        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    background-color:#111827;
                    padding:12px;
                    border-radius:10px;
                    margin-bottom:12px;
                    border:1px solid #1f2937;
                ">
                    <div style="
                        font-weight:600;
                        font-size:14px;
                        color:#f97316;
                        margin-bottom:6px;
                    ">
                        {grouping}
                    </div>
                    <div style="
                        font-size:13px;
                        color:#d1d5db;
                        line-height:1.5;
                    ">
                        {", ".join(cities)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        i += 1

# ------------------------------------------------------------
# TAB 5 — P-TYPE DEEP DIVE (CLOUD SAFE)
# ------------------------------------------------------------

def render_ptype_section(
    df,
    ptype,
    selected_platforms,
    selected_cities,
    selected_fy,
    key_suffix=""
):

    subset = df.copy()

    # ---------------------------
    # FILTERS
    # ---------------------------
    subset = subset[subset["FinancialYear"] == selected_fy]
    subset = subset[subset["P Type"] == ptype]

    if selected_platforms:
        subset = subset[subset["Platform"].isin(selected_platforms)]

    if selected_cities:
        subset = subset[subset["City"].isin(selected_cities)]

    if subset.empty:
        st.info(f"No data for {ptype} with current filters.")
        return

    # ---------------------------
    # VARIANT FILTER
    # ---------------------------
    variants = sorted(subset["Variant"].dropna().unique())
    variant_key = f"variants_{ptype}_{key_suffix}"

    if variants:
        selected_variants = st.multiselect(
            f"Variants for {ptype}",
            options=variants,
            default=variants,
            key=variant_key
        )
        subset = subset[subset["Variant"].isin(selected_variants)]

    if subset.empty:
        st.info(f"No data for {ptype} after variant filter.")
        return

    # ---------------------------
    # MONTH AGGREGATION (LIGHT)
    # ---------------------------
    monthly = (
        subset.groupby(
            ["FYMonthOrder", "MonthLabel"],
            as_index=False
        )
        .agg(
            Industry_Absolute=("Industry_Absolute", "sum"),
            GODESI_Absolute=("GODESI_Absolute", "sum")
        )
        .sort_values("FYMonthOrder")
    )

    if monthly.empty:
        st.info(f"No monthly data for {ptype}.")
        return

    # ---------------------------
    # CALCULATIONS
    # ---------------------------
    monthly["Industry_Cr"] = monthly["Industry_Absolute"] / 1e7

    monthly["Share_Pct"] = np.where(
        monthly["Industry_Absolute"] > 0,
        (monthly["GODESI_Absolute"] / monthly["Industry_Absolute"]) * 100,
        np.nan
    )

    # ---------------------------
    # PLOT
    # ---------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=monthly["MonthLabel"],
            y=monthly["Industry_Cr"],
            name="Industry Size (₹ Cr)",
            mode="lines+markers"
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=monthly["MonthLabel"],
            y=monthly["Share_Pct"],
            name="GO DESI Share (%)",
            mode="lines+markers",
            line=dict(dash="dot")
        ),
        secondary_y=False
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )

    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="GO DESI Share (%)", secondary_y=False)
    fig.update_yaxes(title_text="Industry Size (₹ Cr)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 5 UI
# ------------------------------------------------------------
with tab5:

    st.title("P-Type Deep Dive — Industry vs GO DESi")

    # --------------------------
    # SIDEBAR FILTERS (TAB 5 ONLY)
    # --------------------------
    with st.sidebar:

        st.markdown("### Tab 5 Filters")

        fy_options = sorted(tab5_df["FinancialYear"].unique())

        selected_fy = st.selectbox(
            "Financial Year (Tab 5)",
            options=fy_options,
            index=len(fy_options) - 1,
            key="tab5_fy"
        )

        platforms_tab5 = st.multiselect(
            "Platform (Tab 5)",
            sorted(tab5_df["Platform"].dropna().unique()),
            default=[],
            key="tab5_platform"
        )

        cities_tab5 = st.multiselect(
            "City (Tab 5)",
            sorted(tab5_df["City"].dropna().unique()),
            default=[],
            key="tab5_city"
        )

    sweets_tab, candy_tab = st.tabs(["Indian Sweets", "Candies & Gum"])

    # ---------- Indian Sweets ----------
    with sweets_tab:

        st.subheader("Indian Sweets — P Type Trends")

        sweets_ptypes = [
            "Barfi", "Katli", "Laddu",
            "Peda", "Chikki", "Gajak", "Mysore Pak"
        ]

        for ptype in sweets_ptypes:
            st.markdown(f"### {ptype}")
            render_ptype_section(
                tab5_df,
                ptype,
                platforms_tab5,
                cities_tab5,
                selected_fy,
                key_suffix="sweets"
            )
            st.markdown("---")

    # ---------- Candies & Gum ----------
    with candy_tab:

        st.subheader("Candies & Gum — P Type Trends")

        candy_ptypes = ["Candy", "Gum", "Mint"]

        for ptype in candy_ptypes:
            st.markdown(f"### {ptype}")
            render_ptype_section(
                tab5_df,
                ptype,
                platforms_tab5,
                cities_tab5,
                selected_fy,
                key_suffix="candy"
            )
            st.markdown("---")

# ============================================================
# TAB 6 — USER GUIDE
# ============================================================
with tab6:

    st.title("User Guide")
    st.caption("Expand sections below to understand each dashboard view.")

    with st.expander("Sales Overview", expanded=False):
        st.markdown("""
        ### Purpose
        Provides a consolidated view of secondary sales across time, categories, regions, states, SKUs, and quarters.

        ---

        ### Filters (Left Panel)

        **Metric**  
        Select the metric for analysis.  
        - Revenue  
        - GMV  

        **Region**  
        Filters data to a specific geographic region/regions.

        **State**  
        Filters data to a single state/multiple states.

        **Category**  
        Filters data by primary product category/categories.

        **Platform**  
        Filters data by platform/platforms.

        ---

        ### Category-wise Trend
        - Month-wise secondary sales trend
        - Separate line per category
        - Q1 average shown as a reference line

        ---

        ### Sales Distribution (Q1, Q2, Q3)

        **Region-wise Distribution**  
        - Regional share within each quarter

        **State-wise Distribution**  
        - State-level share within each quarter

        ---

        ### Top 10 SKUs — Quarter-wise

        **Table Columns**
        - Item Name: SKU description  
        - Parent Category: Product category  
        - Revenue: Sales value for the quarter  
        - % of Total: Percentage contribution of the SKU within the quarter

        ---

        ### State Performance — Q1 vs Q2 vs Q3

        **Table Columns**
        - Q1: Revenue in Q1  
        - Q2: Revenue in Q2  
        - Q3: Revenue in Q3  
        - Q2 % vs Q1: Growth from Q1 to Q2  
        - Q3 % vs Q2: Growth from Q2 to Q3  
        - Share % (Q3): State contribution in Q3
                        
        """)

    with st.expander("Top Markets", expanded=False):
        st.markdown("""
        ### Purpose
        Displays monthly secondary sales trends for the top-performing states contributing to 70% of total sales, based on a selected baseline quarter.

        ---

        ### Internal Filters

        **Metric**  
        Select the metric for analysis.  
        - Revenue  
        - GMV  

        **Category**  
        Filters data by primary product category/categories.

        **Platform**  
        Filters data by platform/platforms.

        **Baseline Quarter**  
        Select the quarter used to identify the top contributing states.  
        The same set of states is tracked across subsequent months.

        ---

        ### Definition: Top 70% States
        States are ranked by total sales in the selected baseline quarter.  
        The top states whose cumulative contribution reaches 70% of total sales are included in the analysis.  
        Only these states appear in the trend chart.

        ---

        ### Top Markets Trend Chart

        **Description**
        - Month-wise sales trend for the top 70% contributing states  
        - Each line represents a state  
        - X-axis: Month  
        - Y-axis: Selected metric value  
        """)

    with st.expander("Growth vs Laggards", expanded=False):
        st.markdown("""
        ### Purpose
        Identifies high-growth and lagging markets by comparing performance between two selected quarters, limited to the top 70% contributing states.

        ---

        ### Scope Definition
        All growth and laggard analysis in this section is restricted to the **Top 70% contributing states**, determined based on the selected baseline quarter.  
        States outside this contribution band are excluded from all calculations and visuals.

        ---

        ### Filters

        **Metric**  
        Select the metric for comparison.  
        - Revenue  
        - GMV  

        **Compare Quarter**  
        Quarter whose performance is evaluated for growth or decline.

        **Baseline Quarter**  
        Reference quarter used to calculate percentage change and determine the top 70% contributing states.

        **Category**  
        Filters data by primary product category/categories.

        **Platform**  
        Filters data by platform/platforms.

        ---

        ### Top Growth — Compare Quarter vs Baseline

        **Description**
        - Horizontal bar chart showing percentage growth by state  
        - Includes only states within the top 70% contribution band  
        - Growth is calculated relative to the selected baseline quarter  

        ---

        ### Top Laggards — Compare Quarter vs Baseline

        **Description**
        - Identifies states with negative or lowest growth within the same top 70% set  
        - If no states meet laggard criteria, a placeholder message is displayed  

        ---

        ### Benchmark Shift — Baseline vs Compare Quarter

        **Description**
        - Visual comparison of absolute values between baseline and compare quarters  
        - Shows how the benchmark has shifted for each included state  

        ---

        ### Drill-down — Monthly Trends

        **Growth States**
        - Multi-select allows drilling into monthly trends for growth states only  

        **Laggard States**
        - Multi-select allows drilling into monthly trends for laggard states only  

        Both drill-down views reflect monthly performance for the selected states and active filters.

        """)

    with st.expander("Industry View", expanded=False):
        st.markdown("""
        ### Purpose
        Compares overall industry size with GO DESi performance across regions over time.

        ---

        ### Filters

        **Metric**
        Select comparison basis:
        - GMV (Gross Merchandise Value)
        - SP (Selling Price)

        **Platform**
        Filters data by platform/platforms.

        **Region**
        Filters data by selected city buckets.

        **Category**
        Filters data by parent category/categories.

        ---

        ### GO DESi Market Share Trend (%)

        **Description**
        - Month-wise market share trend for GO DESi  
        - Each line represents a region  
        - Market Share = GO DESi Sales / Industry Size  

        ---

        ### Industry Size vs GO DESi

        **Description**
        - Clustered bar chart comparing:
            - Total Industry Size  
            - GO DESi Sales  
        - Helps understand absolute scale difference  

        ---

        ### Industry Size Trend

        **Description**
        - Multi-line monthly trend of industry size  
        - Each line represents a region  
        - Shows market expansion or contraction  

        ---

        ### GO DESi Sales Trend

        **Description**
        - Multi-line monthly trend of GO DESi sales  
        - Each line represents a region  
        - Shows performance movement across regions  

        ---

        ### How to Use
        - Compare share vs absolute growth  
        - Identify regions with strong penetration  
        - Spot markets where industry is growing but share is flat  

        """)


    with st.expander("Product Type Deep Dive", expanded=False):
        st.markdown("""
        ### Purpose
        Provides a detailed comparison of industry size and GO DESi performance across product types (P-Types), enabling granular analysis within major categories.

        ---

        ### Scope
        This view is structured by:
        - Category (e.g., Indian Sweets, Candies & Gum)
        - Product Type (P-Type) within each category

        Each P-Type is analyzed independently.

        ---

        ### Filters

        **Category**
        Switches between high-level product categories.

        **Product Type (P-Type)**
        Displays trends for each P-Type within the selected category.

        **Variant (Optional)**
        Allows narrowing analysis to selected variants within a P-Type.
        Variant selection affects only the corresponding P-Type chart.

        ---

        ### P-Type Trend — Industry vs GO DESi

        **Description**
        - Dual-axis monthly trend chart per P-Type:
        - Industry Size (₹ Cr)
        - GO DESi Share (%)
        - X-axis: Month
        - Left Y-axis: GO DESi Share (%)
        - Right Y-axis: Industry Size (₹ Cr)

        Each P-Type chart reflects:
        - Market expansion or contraction
        - GO DESi penetration movement within that market

        ---

        ### How to Use
        - Compare industry growth against GO DESi share movement
        - Identify P-Types where share increases despite flat industry growth
        - Detect P-Types with expanding markets but stagnant share

        """)
