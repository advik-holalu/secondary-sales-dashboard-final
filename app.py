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

#TAB 4 LOADS — METRO INDUSTRY VIEW
@st.cache_data
def load_tab4_industry_core():
    return pd.read_parquet("data_agg/tab4_industry_size_core_cities.parquet")

@st.cache_data
def load_tab4_industry_universe():
    return pd.read_parquet("data_agg/tab4_industry_size_universe.parquet")

@st.cache_data
def load_tab4_godesi_gmv():
    return pd.read_parquet("data_agg/tab4_godesi_gmv_core_cities.parquet")


tab4_industry_core = load_tab4_industry_core()
tab4_industry_universe = load_tab4_industry_universe()
tab4_godesi_gmv = load_tab4_godesi_gmv()

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
    tab4_industry_core,
    tab4_industry_universe,
    tab4_godesi_gmv,
    pt_df,
    tab5_month_df,
    tab5_variant_df,
)

if not ALL_FY:
    st.error("No FinancialYear values found in loaded parquet files.")
    st.stop()

# Single global FY selector (used by all tabs)
with st.sidebar:
    selected_fy = st.selectbox(
        "Financial Year",
        options=ALL_FY,
        index=len(ALL_FY) - 1,  # default = latest FY
        key="global_financial_year",
    )

# ------------------------------------------------------------
# FY-SAFE HELPERS (USE EVERYWHERE)
# ------------------------------------------------------------
def filter_by_fy(dfin: pd.DataFrame, fy: str) -> pd.DataFrame:
    if dfin is None or dfin.empty:
        return dfin
    if "FinancialYear" not in dfin.columns:
        return dfin
    return dfin[dfin["FinancialYear"].astype(str) == str(fy)].copy()

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



# ============================================================
# DEFINE DASHBOARD TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sales Overview",
    "Top Markets",
    "Growth vs Laggards",
    "Metro Industry View",
    "Product Type Deep Dive",
    "User Guide"
])

# ============================================================
# TAB 1 — SALES OVERVIEW (FY SAFE)
# ============================================================
with tab1:

    # ------------------------------------------------------------
    # SIDEBAR FILTERS (TAB 1)
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
    # SHARED FILTER HELPER
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
    # APPLY FILTERS + FY FILTER
    # ------------------------------------------------------------
    df_filt = apply_tab1_filters(df)
    df_filt = filter_by_fy(df_filt, selected_fy)

    if df_filt.empty:
        st.warning("No data for selected filters.")
        st.stop()

    month_order = get_month_order(df_filt)

    # ============================================================
    # SECTION 1 — CATEGORY TREND (PARENT CATEGORY)
    # ============================================================
    st.title("Secondary Sales Overview")
    st.subheader("Parent Category-wise Trend")

    timeline = (
        df_filt
        .groupby(
            ["FinancialYear", "FYMonthOrder", "MonthLabel", "Parent Category"],
            as_index=False,
            observed=False
        )[metric]
        .sum()
        .sort_values("FYMonthOrder")
    )

    timeline = enforce_month_order(timeline, month_order)

    fig = px.line(
        timeline,
        x="MonthLabel",
        y=metric,
        color="Parent Category",
        markers=True,
        text=timeline[metric].apply(format_indian),
        category_orders={"MonthLabel": month_order}
    )

    y_max = timeline[metric].max()
    y_ticks = np.linspace(0, y_max, 6)

    fig.update_yaxes(
        tickvals=y_ticks,
        ticktext=[format_indian(v, decimals=0) for v in y_ticks]
    )

    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # SECTION 2 — DONUTS (FY QUARTERS)
    # ============================================================
    st.subheader("Sales Distribution — Q1, Q2, Q3")

    donut_base = apply_tab1_filters(donut_df)
    donut_base = filter_by_fy(donut_base, selected_fy)

    def donut_pair(fy_quarter):
        dfq = donut_base[donut_base["FYQuarter"] == fy_quarter]

        if dfq.empty:
            return

        # Quarter heading
        st.markdown(
            f"### {fy_quarter}",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            reg = dfq.groupby("Region Name", as_index=False, observed=False)[metric].sum()
            fig = px.pie(
                reg,
                names="Region Name",
                values=metric,
                hole=0.45,
                title="Region-wise Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            stt = dfq.groupby("State Name", as_index=False, observed=False)[metric].sum()
            fig = px.pie(
                stt,
                names="State Name",
                values=metric,
                hole=0.45,
                title="State-wise Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

    available_quarters = (
        donut_base["FYQuarter"]
        .dropna()
        .unique()
        .tolist()
    )

    # Sort FY quarters in correct order
    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    available_quarters = [q for q in quarter_order if q in available_quarters]

    for q in available_quarters:
        donut_pair(q)

    # ============================================================
    # SECTION 3 — TOP SKUs (FY QUARTERS)
    # ============================================================
    st.subheader(f"Top 10 SKUs — Quarter-wise ({metric})")

    sku_base = apply_tab1_filters(sku_df)
    sku_base = filter_by_fy(sku_base, selected_fy)

    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    available_quarters = [
        q for q in quarter_order
        if q in sku_base["FYQuarter"].unique()
    ]

    def render_top_skus_table(fy_quarter):
        dfq = sku_base[sku_base["FYQuarter"] == fy_quarter]

        if dfq.empty:
            return

        st.markdown(f"### {fy_quarter}")

        dfq = (
            dfq.sort_values(metric, ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        dfq[f"{metric} (₹)"] = dfq[metric].apply(format_indian)

        st.dataframe(
            dfq[
                [
                    "Item Name",
                    "Parent Category",
                    "L1 Category",
                    f"{metric} (₹)",
                ]
            ].reset_index(drop=True),
            use_container_width=True
        )

        st.markdown("---")

    for q in available_quarters:
        render_top_skus_table(q)


    # ============================================================
    # SECTION 4 — STATE PERFORMANCE (FY QUARTERS)
    # ============================================================
    st.subheader("State Performance — Quarter-wise")

    df_state = apply_tab1_filters(state_q_df)
    df_state = filter_by_fy(df_state, selected_fy)

    if df_state.empty:
        st.info("No data available for selected filters.")
        st.stop()

    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    available_quarters = [
        q for q in quarter_order
        if q in df_state["FYQuarter"].unique()
    ]

    pivot = (
        df_state
        .groupby(["State Name", "FYQuarter"], as_index=False, observed=False)[metric]
        .sum()
        .pivot(index="State Name", columns="FYQuarter", values=metric)
        .fillna(0)
    )

    pivot = pivot[available_quarters]

    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).drop(columns="Total")

    for q in available_quarters:
        pivot[q] = pivot[q].apply(format_indian)

    pivot = pivot.reset_index()
    pivot.insert(0, "S.No", range(1, len(pivot) + 1))

    st.dataframe(
        pivot,
        use_container_width=True
    )

    
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
    df2 = filter_by_fy(df2, selected_fy)

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
    # HELPER — TOP 70% STATES (BASED ON BASELINE FY QUARTER)
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
    # MONTH ORDER (FY SAFE)
    # --------------------------------------------------------
    month_order = get_month_order(plot_df)

    trend_df = (
        plot_df
        .groupby(
            ["State Name", "FYMonthOrder", "MonthLabel"],
            as_index=False,
            observed=False
        )[metric_tab2]
        .sum()
        .sort_values("FYMonthOrder")
    )

    trend_df = enforce_month_order(trend_df, month_order)

    # --------------------------------------------------------
    # LEGEND ORDER (BY TOTAL CONTRIBUTION)
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
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True,
        text=trend_df[metric_tab2].apply(format_indian),
        category_orders={
            "MonthLabel": month_order,
            "State Name": state_order
        }
    )

    fig.update_traces(textposition="top center")

    y_max = trend_df[metric_tab2].max()
    y_ticks = np.linspace(0, y_max, 6)

    fig.update_yaxes(
        tickvals=y_ticks,
        ticktext=[format_indian(v, decimals=0) for v in y_ticks]
    )

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
    df_month = filter_by_fy(tab3_month_df.copy(), selected_fy)
    df_total = filter_by_fy(tab3_total_df.copy(), selected_fy)

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
    # TOP 70% STATES (BASELINE FY QUARTER)
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
    # BASELINE vs COMPARE (MONTH AVG)
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
    # SPLIT GROWTH / LAGGARDS
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
        if growth_pos.empty:
            st.info("No growth states.")
        else:
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
        if growth_neg.empty:
            st.info("No laggard states.")
        else:
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
    # DRILL-DOWN — MONTHLY TRENDS
    # ----------------------------
    st.markdown("---")
    st.subheader("Monthly Trends — Drill-down")

    month_order = get_month_order(df_month)
    df_month = enforce_month_order(df_month, month_order)

    # ----------------------------
    # GROWTH CHART
    # ----------------------------

    st.markdown("### Growth States — Monthly Trend")

    growth_states = growth_pos["State Name"].tolist()

    if not growth_states:
        st.info("No growth states to display.")
    else:
        selected_growth_states = st.multiselect(
            "Select Growth States",
            options=growth_states,
            default=growth_states,
            key="growth_state_selector"
        )

        if not selected_growth_states:
            st.info("Select at least one state to view the trend.")
        else:
            growth_trend = (
                df_month[df_month["State Name"].isin(selected_growth_states)]
                .groupby(
                    ["State Name", "FYMonthOrder", "MonthLabel"],
                    as_index=False,
                    observed=False
                )[metric]
                .sum()
                .sort_values("FYMonthOrder")
            )

            # Remove zero months
            growth_trend = growth_trend[growth_trend[metric] > 0]

            fig = px.line(
                growth_trend,
                x="MonthLabel",
                y=metric,
                color="State Name",
                markers=True,
                text=growth_trend[metric].apply(format_indian),
                category_orders={"MonthLabel": month_order}
            )

            fig.update_traces(
                textposition="top center",
                textfont=dict(size=11)
            )

            y_max = growth_trend[metric].max()
            fig.update_yaxes(
                tickvals=np.linspace(0, y_max, 6),
                ticktext=[format_indian(v, 0) for v in np.linspace(0, y_max, 6)]
            )

            st.plotly_chart(fig, use_container_width=True)

    
    # ----------------------------
    # LAGGARD CHART
    # ----------------------------

    st.markdown("### Laggard States — Monthly Trend")

    laggard_states = growth_neg["State Name"].tolist()

    if not laggard_states:
        st.info("No laggard states for the selected comparison.")
    else:
        selected_laggard_states = st.multiselect(
            "Select Laggard States",
            options=laggard_states,
            default=laggard_states,
            key="laggard_state_selector"
        )

        if not selected_laggard_states:
            st.info("Select at least one state to view the trend.")
        else:
            laggard_trend = (
                df_month[df_month["State Name"].isin(selected_laggard_states)]
                .groupby(
                    ["State Name", "FYMonthOrder", "MonthLabel"],
                    as_index=False,
                    observed=False
                )[metric]
                .sum()
                .sort_values("FYMonthOrder")
            )

            # Remove zero months
            laggard_trend = laggard_trend[laggard_trend[metric] > 0]

            fig = px.line(
                laggard_trend,
                x="MonthLabel",
                y=metric,
                color="State Name",
                markers=True,
                text=laggard_trend[metric].apply(format_indian),
                category_orders={"MonthLabel": month_order}
            )

            fig.update_traces(
                textposition="top center",
                textfont=dict(size=11)
            )

            y_max = laggard_trend[metric].max()
            fig.update_yaxes(
                tickvals=np.linspace(0, y_max, 6),
                ticktext=[format_indian(v, 0) for v in np.linspace(0, y_max, 6)]
            )

            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 4 — METRO INDUSTRY VIEW (FY SAFE)
# ============================================================
with tab4:
    st.title("Metro Industry View")

    # --------------------------------------------------------
    # SIDEBAR FILTERS (TAB 4)
    # --------------------------------------------------------
    with st.sidebar:
        st.header("Metro Industry Filters")

        platform_tab4 = st.multiselect(
            "Platform (Tab 4)",
            sorted(tab4_industry_core["Platform"].dropna().unique()),
            default=[]
        )

        city_tab4 = st.multiselect(
            "City (Tab 4 – Core)",
            sorted(tab4_industry_core["City Name"].dropna().unique()),
            default=[]
        )

        category_tab4 = st.multiselect(
            "Category (Tab 4)",
            sorted(tab4_industry_core["Parent Category"].dropna().unique()),
            default=[]
        )

    # --------------------------------------------------------
    # SAFE DEFAULTS
    # --------------------------------------------------------
    platforms = platform_tab4 or tab4_industry_core["Platform"].dropna().unique().tolist()
    cities = city_tab4 or tab4_industry_core["City Name"].dropna().unique().tolist()
    categories = category_tab4 or tab4_industry_core["Parent Category"].dropna().unique().tolist()

    # --------------------------------------------------------
    # FY FILTER
    # --------------------------------------------------------
    ind_core = filter_by_fy(tab4_industry_core, selected_fy)
    ind_univ = filter_by_fy(tab4_industry_universe, selected_fy)
    godesi = filter_by_fy(tab4_godesi_gmv, selected_fy)

    # ========================================================
    # GRAPH 1 — GO DESi vs INDUSTRY SIZE
    # ========================================================
    st.subheader("GO DESi vs Industry Size")

    ind_g1 = ind_core[
        (ind_core["Platform"].isin(platforms))
        & (ind_core["City Name"].isin(cities))
        & (ind_core["Parent Category"].isin(categories))
    ]

    gmv_g1 = godesi[
        (godesi["Platform"].isin(platforms))
        & (godesi["City Name"].isin(cities))
        & (godesi["Parent Category"].isin(categories))
    ]

    ind_m = (
        ind_g1
        .groupby(["FYMonthOrder", "MonthLabel"], as_index=False, observed=False)["Industry_Size"]
        .sum()
        .sort_values("FYMonthOrder")
    )

    gmv_m = (
        gmv_g1
        .groupby(["FYMonthOrder", "MonthLabel"], as_index=False, observed=False)["GO_DESi_GMV"]
        .sum()
        .sort_values("FYMonthOrder")
    )

    comp = ind_m.merge(gmv_m, on=["FYMonthOrder", "MonthLabel"], how="left")
    comp["GO_DESi_GMV"] = comp["GO_DESi_GMV"].fillna(0)

    month_order = comp.sort_values("FYMonthOrder")["MonthLabel"].tolist()

    fig1 = px.bar(
        comp,
        x="MonthLabel",
        y=["Industry_Size", "GO_DESi_GMV"],
        barmode="group",
        category_orders={"MonthLabel": month_order}
    )

    for trace in fig1.data:
        trace.text = [format_indian(v) for v in trace.y]
        trace.textposition = "outside"
        trace.hoverinfo = "skip"

    y_max = max(comp["Industry_Size"].max(), comp["GO_DESi_GMV"].max())
    fig1.update_yaxes(
        tickvals=np.linspace(0, y_max, 6),
        ticktext=[format_indian(v, 0) for v in np.linspace(0, y_max, 6)]
    )

    st.plotly_chart(fig1, use_container_width=True)

    comp_disp = comp.copy()
    comp_disp["Industry_Size"] = comp_disp["Industry_Size"].apply(format_indian)
    comp_disp["GO_DESi_GMV"] = comp_disp["GO_DESi_GMV"].apply(format_indian)
    comp_disp["GO_DESi_Share_%"] = (
        comp["GO_DESi_GMV"] / comp["Industry_Size"].replace(0, np.nan) * 100
    ).apply(format_pct)

    st.dataframe(
        comp_disp[["MonthLabel", "Industry_Size", "GO_DESi_GMV", "GO_DESi_Share_%"]],
        use_container_width=True
    )

    # ========================================================
    # GRAPH 2 — INDUSTRY SIZE TREND (ALL CITIES)
    # ========================================================
    st.subheader("Industry Size Trend — All Cities")

    trend = (
        ind_univ[
            (ind_univ["Platform"].isin(platforms))
            & (ind_univ["Parent Category"].isin(categories))
        ]
        .groupby(["FYMonthOrder", "MonthLabel", "City Name"], as_index=False, observed=False)["Industry_Size"]
        .sum()
        .sort_values("FYMonthOrder")
    )

    month_order = trend.drop_duplicates("FYMonthOrder").sort_values("FYMonthOrder")["MonthLabel"].tolist()

    fig2 = px.line(
        trend,
        x="MonthLabel",
        y="Industry_Size",
        color="City Name",
        markers=True,
        text=trend["Industry_Size"].apply(format_indian),
        category_orders={"MonthLabel": month_order}
    )

    fig2.update_traces(textposition="top center")

    y_max = trend["Industry_Size"].max()
    fig2.update_yaxes(
        tickvals=np.linspace(0, y_max, 6),
        ticktext=[format_indian(v, 0) for v in np.linspace(0, y_max, 6)]
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ========================================================
    # GRAPH 3 — GO DESi GMV TREND (CORE CITIES)
    # ========================================================
    st.subheader("GO DESi GMV Trend — Core Cities")

    gmv_trend = (
        godesi[
            (godesi["Platform"].isin(platforms))
            & (godesi["City Name"].isin(cities))
            & (godesi["Parent Category"].isin(categories))
        ]
        .groupby(["FYMonthOrder", "MonthLabel", "City Name"], as_index=False, observed=False)["GO_DESi_GMV"]
        .sum()
        .sort_values("FYMonthOrder")
    )

    month_order = gmv_trend.drop_duplicates("FYMonthOrder").sort_values("FYMonthOrder")["MonthLabel"].tolist()

    fig3 = px.line(
        gmv_trend,
        x="MonthLabel",
        y="GO_DESi_GMV",
        color="City Name",
        markers=True,
        text=gmv_trend["GO_DESi_GMV"].apply(format_indian),
        category_orders={"MonthLabel": month_order}
    )

    fig3.update_traces(textposition="top center")

    y_max = gmv_trend["GO_DESi_GMV"].max()
    fig3.update_yaxes(
        tickvals=np.linspace(0, y_max, 6),
        ticktext=[format_indian(v, 0) for v in np.linspace(0, y_max, 6)]
    )

    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# TAB 5 — P-TYPE DEEP DIVE (FY SAFE)
# ------------------------------------------------------------
def render_ptype_section(
    pt_df,
    ptype,
    selected_platforms,
    selected_cities,
    key_suffix=""
):
    subset = pt_df.copy()

    # ---------------------------
    # FILTERS
    # ---------------------------
    if selected_platforms:
        subset = subset[subset["Platform"].isin(selected_platforms)]

    if selected_cities:
        subset = subset[subset["City"].isin(selected_cities)]

    subset = subset[subset["P Type"] == ptype]

    if subset.empty:
        st.info(f"No data for {ptype} with current filters.")
        return

    # ---------------------------
    # VARIANT FILTER
    # ---------------------------
    variants = sorted(subset["Variant"].dropna().unique().tolist())
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
    # MONTH ORDER
    # ---------------------------
    subset = subset.copy()
    subset["FYMonthOrder"] = subset["FYMonthOrder"].astype(int)

    # ---------------------------
    # AGGREGATION (INDUSTRY ONLY)
    # ---------------------------
    monthly = (
        subset
        .groupby(
            ["FYMonthOrder", "MonthLabel"],
            as_index=False,
            observed=False
        )["Absolute_Size"]
        .sum()
        .sort_values("FYMonthOrder")
    )

    if monthly.empty:
        st.info(f"No monthly data for {ptype}.")
        return

    monthly["Industry_Cr"] = monthly["Absolute_Size"] / 1e7

    # ---------------------------
    # PLOT
    # ---------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly["MonthLabel"],
            y=monthly["Industry_Cr"],
            mode="lines+markers+text",
            name="Industry Size (₹ Cr)",
            text=[f"{v:.1f} Cr" for v in monthly["Industry_Cr"]],
            textposition="top center"
        )
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Month",
        yaxis_title="Industry Size (₹ Cr)",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )

    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.title("P-Type Deep Dive — Industry vs GO DESi")

    with st.sidebar:
        st.header("Deep Dive Filters")

        platforms_tab5 = st.multiselect(
            "Platform (Tab 5)",
            sorted(pt_df["Platform"].dropna().unique().tolist()),
            default=[],
            key="platform_tab5"
        )

        cities_tab5 = st.multiselect(
            "City (Tab 5)",
            sorted(pt_df["City"].dropna().unique().tolist()),
            default=[],
            key="city_tab5"
        )

    sweets_tab, candy_tab = st.tabs(["Indian Sweets", "Candies & Gum"])

    with sweets_tab:
        st.subheader("Indian Sweets — P Type Trends")

        sweets_ptypes = [
            "Barfi", "Katli", "Laddu",
            "Peda", "Chikki", "Gajak", "Mysore Pak"
        ]

        for ptype in sweets_ptypes:
            st.markdown(f"### {ptype}")
            render_ptype_section(
                pt_df,
                ptype,
                platforms_tab5,
                cities_tab5,
                key_suffix="sweets"
            )
            st.markdown("---")

    with candy_tab:
        st.subheader("Candies & Gum — P Type Trends")

        candy_ptypes = ["Candy", "Gum", "Mint"]

        for ptype in candy_ptypes:
            st.markdown(f"### {ptype}")
            render_ptype_section(
                pt_df,
                ptype,
                platforms_tab5,
                cities_tab5,
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

    with st.expander("Metro Industry View", expanded=False):
        st.markdown("""
        ### Purpose
        Provides a city-level comparison of GO DESi performance against overall industry size across key metro markets.

        ---

        ### Filters

        **Platform**  
        Filters data by platform/platforms.

        **City (Core)**  
        Filters data to selected core metro cities.

        **Category**  
        Filters data by primary product category/categories.

        ---

        ### GO DESi vs Industry Size

        **Description**
        - Monthly bar chart comparing:
        - Total industry size
        - GO DESi GMV
        - Values shown at city aggregation level
        - Used to assess scale gap and penetration opportunity

        ---

        ### Industry Size Trend — All Cities

        **Description**
        - Month-wise industry size trend across all tracked cities
        - Separate line per city
        - Used to understand overall market expansion or contraction

        ---

        ### GO DESi GMV Trend — Core Cities

        **Description**
        - Month-wise GO DESi GMV trend for selected core cities
        - Separate line per city
        - Used to track brand performance within key metros

        ---

        ### Tabular View

        **Table Columns**
        - Month: Calendar month
        - Industry Size: Total market size for the city
        - GO DESi GMV: GO DESi sales value
        - GO DESi Share %: Percentage share of GO DESi within industry size

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
