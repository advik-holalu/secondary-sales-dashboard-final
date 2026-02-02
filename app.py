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
# TAB 1 — SALES OVERVIEW
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
    # APPLY FILTERS
    # ------------------------------------------------------------
    df_filt = apply_tab1_filters(df)

    if df_filt.empty:
        st.warning("No data for selected filters.")
        st.stop()

    month_order = (
        df_filt[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    # ============================================================
    # SECTION 1 — CATEGORY TREND (PARENT CATEGORY)
    # ============================================================
    st.title("Secondary Sales Overview")
    st.subheader("Parent Category-wise Trend")

    timeline = (
        df_filt
        .groupby(
            ["Year", "MonthNum", "MonthLabel", "Parent Category"],
            as_index=False
        )[metric]
        .sum()
        .sort_values(["Year", "MonthNum"])
    )

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
    # SECTION 2 — DONUTS
    # ============================================================
    st.subheader("Sales Distribution — Q1, Q2, Q3")

    donut_base = apply_tab1_filters(donut_df)

    def donut_pair(quarter, title):
        dfq = donut_base[donut_base["Quarter"] == quarter]

        c1, c2 = st.columns(2)

        with c1:
            reg = dfq.groupby("Region Name", as_index=False)[metric].sum()
            fig = px.pie(reg, names="Region Name", values=metric, hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            stt = dfq.groupby("State Name", as_index=False)[metric].sum()
            fig = px.pie(stt, names="State Name", values=metric, hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

    donut_pair("Q1", "Q1")
    donut_pair("Q2", "Q2")
    donut_pair("Q3", "Q3")

    # ============================================================
    # SECTION 3 — TOP SKUs
    # ============================================================
    st.subheader(f"Top 10 SKUs — Quarter-wise ({metric})")

    sku_base = apply_tab1_filters(sku_df)

    def render_top_skus_table(quarter):
        dfq = sku_base[sku_base["Quarter"] == quarter]
        if dfq.empty:
            return

        dfq = (
            dfq.sort_values(metric, ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        dfq.insert(0, "S.No", range(1, len(dfq) + 1))
        dfq[f"{metric} (₹)"] = dfq[metric].apply(format_indian)

        st.dataframe(
            dfq[
                [
                    "S.No",
                    "Item Name",
                    "Parent Category",
                    "L1 Category",
                    f"{metric} (₹)",
                ]
            ],
            use_container_width=True
        )

    render_top_skus_table("Q1")
    render_top_skus_table("Q2")
    render_top_skus_table("Q3")

    # ============================================================
    # SECTION 4 — STATE PERFORMANCE
    # ============================================================
    st.subheader("State Performance — Q1 vs Q2 vs Q3")

    df_state = apply_tab1_filters(state_q_df)

    q1 = (
        df_state[df_state["Quarter"] == "Q1"]
        .groupby("State Name", as_index=False)[metric]
        .sum()
        .rename(columns={metric: "Q1"})
    )

    q2 = (
        df_state[df_state["Quarter"] == "Q2"]
        .groupby("State Name", as_index=False)[metric]
        .sum()
        .rename(columns={metric: "Q2"})
    )

    q3 = (
        df_state[df_state["Quarter"] == "Q3"]
        .groupby("State Name", as_index=False)[metric]
        .sum()
        .rename(columns={metric: "Q3"})
    )

    merged = (
        q1.merge(q2, on="State Name", how="outer")
        .merge(q3, on="State Name", how="outer")
        .fillna(0)
    )


    merged["Total"] = merged["Q1"] + merged["Q2"] + merged["Q3"]
    merged = merged.sort_values("Total", ascending=False)

    merged["Q1 (₹)"] = merged["Q1"].apply(format_indian)
    merged["Q2 (₹)"] = merged["Q2"].apply(format_indian)
    merged["Q3 (₹)"] = merged["Q3"].apply(format_indian)

    merged.insert(0, "S.No", range(1, len(merged) + 1))

    st.dataframe(
        merged[
            ["S.No", "State Name", "Q1 (₹)", "Q2 (₹)", "Q3 (₹)"]
        ],
        use_container_width=True
    )
    
# ============================================================
# TAB 2 — TOP MARKETS (TOP 70% STATES)
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

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------
    df2 = tab2_df.copy()

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
    # HELPER — TOP 70% STATES
    # --------------------------------------------------------
    def get_top70_states(dfin, metric):
        state_tot = (
            dfin.groupby("State Name", as_index=False)[metric]
            .sum()
            .sort_values(metric, ascending=False)
        )
        total = state_tot[metric].sum()
        state_tot["CumShare"] = state_tot[metric].cumsum() / total * 100
        top_states = state_tot[state_tot["CumShare"] <= 70]["State Name"].tolist()
        return top_states or state_tot.head(1)["State Name"].tolist()

    # --------------------------------------------------------
    # BASELINE QUARTER
    # --------------------------------------------------------
    st.subheader("Top Markets Trend — Based on Selected Baseline Quarter")

    baseline_q = st.selectbox(
        "Baseline Quarter",
        sorted(df2["Quarter"].unique()),
        index=0
    )

    baseline_df = df2[df2["Quarter"] == baseline_q]
    top_states = get_top70_states(baseline_df, metric_tab2)

    plot_df = df2[df2["State Name"].isin(top_states)]

    # --------------------------------------------------------
    # MONTH ORDER
    # --------------------------------------------------------
    month_order = (
        plot_df[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    trend_df = (
        plot_df
        .groupby(
            ["State Name", "MonthNum", "MonthLabel"],
            as_index=False
        )[metric_tab2]
        .sum()
        .sort_values("MonthNum")
    )

    trend_df["MonthLabel"] = pd.Categorical(
        trend_df["MonthLabel"],
        categories=month_order,
        ordered=True
    )

    # --------------------------------------------------------
    # LEGEND ORDER (BY TOTAL CONTRIBUTION)
    # --------------------------------------------------------
    state_order = (
        trend_df
        .groupby("State Name", as_index=False)[metric_tab2]
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
        category_orders={"State Name": state_order}
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
# TAB 3 — GROWTH VS LAGGARDS
# ============================================================
with tab3:
    st.title("Growth vs Laggard Markets (Top 70% Contribution)")

    # ----------------------------
    # SIDEBAR FILTERS (TAB 3)
    # ----------------------------
    with st.sidebar:
        st.header("Growth vs Laggards Filters")

        metric = st.radio("Metric (Tab 3)", ["Revenue", "GMV"], index=0)

        compare_q = st.selectbox(
            "Compare Quarter",
            ["Q1", "Q2", "Q3"],
            index=2
        )

        baseline_q = st.selectbox(
            "Baseline Quarter",
            [q for q in ["Q1", "Q2", "Q3"] if q != compare_q],
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
    # APPLY FILTERS
    # ----------------------------
    df_month = tab3_month_df.copy()
    df_total = tab3_total_df.copy()

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
    # TOP 70% STATES (BASED ON BASELINE QUARTER)
    # ----------------------------
    baseline_df = df_month[df_month["Quarter"] == baseline_q]

    state_rank = (
        baseline_df
        .groupby("State Name", as_index=False)[metric]
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
        df_month[df_month["Quarter"] == baseline_q]
        .groupby("State Name", as_index=False)[metric]
        .mean()
        .rename(columns={metric: "Baseline"})
    )

    comp_avg = (
        df_month[df_month["Quarter"] == compare_q]
        .groupby("State Name", as_index=False)[metric]
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

# ============================================================
# TAB 4 — METRO INDUSTRY VIEW (FIXED DEFAULT FILTER BEHAVIOR)
# ============================================================
with tab4:
    st.title("Metro Industry View")

    # --------------------------------------------------------
    # SIDEBAR FILTERS (TAB 4)
    # --------------------------------------------------------
    with st.sidebar:
        st.header("Metro Industry Filters")

        year_tab4 = st.selectbox(
            "Year (Tab 4)",
            sorted(tab4_industry_core["Year"].unique()),
            index=0
        )

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
            sorted(
                tab4_industry_core["Primary Cat"].dropna().unique()
            ),
            default=[]
        )

    # --------------------------------------------------------
    # SAFE DEFAULTS (EMPTY = ALL)
    # --------------------------------------------------------
    platforms = (
        platform_tab4
        if platform_tab4
        else tab4_industry_core["Platform"].dropna().unique().tolist()
    )

    cities = (
        city_tab4
        if city_tab4
        else tab4_industry_core["City Name"].dropna().unique().tolist()
    )

    categories = (
        category_tab4
        if category_tab4
        else tab4_industry_core["Primary Cat"].dropna().unique().tolist()
    )

    # --------------------------------------------------------
    # MONTH ORDER HELPER
    # --------------------------------------------------------
    MONTH_MAP = {
        "Jan": 1, "Feb": 2, "Mar": 3,
        "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9,
        "Oct": 10, "Nov": 11, "Dec": 12
    }

    def apply_month_order(df):
        df = df.copy()
        df["Month"] = df["Month"].astype(str).str.strip().str[:3].str.title()
        df["MonthNum"] = df["Month"].map(MONTH_MAP)
        df = df.dropna(subset=["MonthNum"])
        df["MonthNum"] = df["MonthNum"].astype(int)
        df = df.sort_values("MonthNum")
        df["Month"] = pd.Categorical(
            df["Month"],
            categories=[m for m, _ in sorted(MONTH_MAP.items(), key=lambda x: x[1])],
            ordered=True
        )
        return df

    # --------------------------------------------------------
    # INDIAN Y-AXIS HELPER (NEW)
    # --------------------------------------------------------
    def apply_indian_yaxis(fig, max_val):
        ticks = np.linspace(0, max_val, 6)
        fig.update_yaxes(
            tickvals=ticks,
            ticktext=[format_indian(v, decimals=0) for v in ticks]
        )
        return fig

    # ========================================================
    # GRAPH 1 — GO DESi vs INDUSTRY SIZE
    # ========================================================
    st.subheader("GO DESi vs Industry Size")

    ind_g1 = tab4_industry_core[
        (tab4_industry_core["Year"] == year_tab4)
        & (tab4_industry_core["Platform"].isin(platforms))
        & (tab4_industry_core["City Name"].isin(cities))
        & (tab4_industry_core["Primary Cat"].isin(categories))
    ]

    gmv_g1 = tab4_godesi_gmv[
        (tab4_godesi_gmv["Year"] == year_tab4)
        & (tab4_godesi_gmv["Platform"].isin(platforms))
        & (tab4_godesi_gmv["City Name"].isin(cities))
        & (tab4_godesi_gmv["Primary Cat"].isin(categories))
    ]

    ind_m = apply_month_order(
        ind_g1.groupby("Month", as_index=False)["Industry Size"].sum()
    )

    gmv_m = apply_month_order(
        gmv_g1.groupby("Month", as_index=False)["GO DESi GMV"].sum()
    )

    comp = ind_m.merge(gmv_m, on="Month", how="left")

    for col in ["GO DESi GMV"]:
        if col in comp.columns:
            comp[col] = comp[col].fillna(0)

    comp["GO DESi Share %"] = (
        comp["GO DESi GMV"] / comp["Industry Size"].replace(0, np.nan) * 100
    ).round(2)

    fig1 = px.bar(
        comp,
        x="Month",
        y=["Industry Size", "GO DESi GMV"],
        barmode="group"
    )

    for trace in fig1.data:
        trace.text = [format_indian(v) for v in trace.y]
        trace.textposition = "outside"
        trace.hoverinfo = "skip"

    y_max = max(
        comp["Industry Size"].max(),
        comp["GO DESi GMV"].max()
    )
    fig1 = apply_indian_yaxis(fig1, y_max)

    st.plotly_chart(fig1, use_container_width=True)

    comp_disp = comp.copy()
    comp_disp["Industry Size"] = comp_disp["Industry Size"].apply(format_indian)
    comp_disp["GO DESi GMV"] = comp_disp["GO DESi GMV"].apply(format_indian)
    comp_disp["GO DESi Share %"] = comp_disp["GO DESi Share %"].apply(format_pct)

    st.dataframe(
        comp_disp[["Month", "Industry Size", "GO DESi GMV", "GO DESi Share %"]],
        use_container_width=True
    )

    # ========================================================
    # GRAPH 2 — INDUSTRY SIZE TREND (ALL CITIES)
    # ========================================================
    st.subheader("Industry Size Trend — All Cities")

    ind_g2 = tab4_industry_universe[
        (tab4_industry_universe["Year"] == year_tab4)
        & (tab4_industry_universe["Platform"].isin(platforms))
        & (tab4_industry_universe["Primary Cat"].isin(categories))
    ]

    trend = (
        ind_g2
        .groupby(["Month", "City Name"], as_index=False)["Industry Size"]
        .sum()
    )

    trend = apply_month_order(trend)

    fig2 = px.line(
        trend,
        x="Month",
        y="Industry Size",
        color="City Name",
        markers=True,
        text=trend["Industry Size"].apply(format_indian)
    )

    fig2.update_traces(textposition="top center")
    fig2 = apply_indian_yaxis(fig2, trend["Industry Size"].max())

    st.plotly_chart(fig2, use_container_width=True)

    # ========================================================
    # GRAPH 3 — GO DESi GMV TREND (CORE CITIES)
    # ========================================================
    st.subheader("GO DESi GMV Trend — Core Cities")

    gmv_g3 = tab4_godesi_gmv[
        (tab4_godesi_gmv["Year"] == year_tab4)
        & (tab4_godesi_gmv["Platform"].isin(platforms))
        & (tab4_godesi_gmv["City Name"].isin(cities))
        & (tab4_godesi_gmv["Primary Cat"].isin(categories))
    ]

    gmv_trend = (
        gmv_g3
        .groupby(["Month", "City Name"], as_index=False)["GO DESi GMV"]
        .sum()
    )

    gmv_trend = apply_month_order(gmv_trend)

    fig3 = px.line(
        gmv_trend,
        x="Month",
        y="GO DESi GMV",
        color="City Name",
        markers=True,
        text=gmv_trend["GO DESi GMV"].apply(format_indian)
    )

    fig3.update_traces(textposition="top center")
    fig3 = apply_indian_yaxis(fig3, gmv_trend["GO DESi GMV"].max())

    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# TAB 5 — P-TYPE SECTION RENDERER (CLOUD SAFE)
# ------------------------------------------------------------
def render_ptype_section(pt_df, ptype, selected_platforms, selected_cities, key_suffix=""):

    subset = pt_df.copy()

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
    # DATE DERIVATION (NO Date COLUMN ASSUMED)
    # ---------------------------
    subset["Month"] = subset["Month"].astype(str).str.strip().str[:3].str.title()

    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3,
        "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9,
        "Oct": 10, "Nov": 11, "Dec": 12,
    }

    subset["MonthNum"] = subset["Month"].map(month_map)
    subset["MonthLabel"] = subset["Month"]

    subset = subset.dropna(subset=["MonthNum"])
    subset["MonthNum"] = subset["MonthNum"].astype(int)

    # ---------------------------
    # AGGREGATIONS (CLEAN + SAFE)
    # ---------------------------
    industry = (
        subset
        .groupby("MonthNum", observed=False)["Absolute size"]
        .sum()
        .sort_index()
    )

    godesi = (
        subset[subset["Brand"].astype(str).str.upper().str.strip() == "GO DESI"]
        .groupby("MonthNum", observed=False)["Absolute size"]
        .sum()
        .reindex(industry.index, fill_value=0)
    )

    if industry.empty:
        st.info(f"No monthly data for {ptype}.")
        return

    month_labels = (
        subset.groupby("MonthNum", observed=False)["MonthLabel"]
        .first()
        .reindex(industry.index)
        .tolist()
    )

    industry_vals = industry.values.astype(float)
    godesi_vals = godesi.values.astype(float)

    industry_crore = industry_vals / 1e7

    share_pct = np.divide(
        godesi_vals,
        industry_vals,
        out=np.zeros_like(industry_vals),
        where=industry_vals > 0
    ) * 100

    # ---------------------------
    # PLOT (IDENTICAL TO OLD)
    # ---------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Industry size (₹ Cr)
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=industry_crore,
            name="Industry Size (₹ Cr)",
            mode="lines+markers+text",
            text=[f"{v:.1f} Cr" for v in industry_crore],
            textposition="top center"
        ),
        secondary_y=True
    )

    # GO DESi share (%)
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=share_pct,
            name="GO DESi Share (%)",
            mode="lines+markers+text",
            text=[format_pct(v) for v in share_pct],
            textposition="bottom center",
            line=dict(dash="dot")
        ),
        secondary_y=False
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )

    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="GO DESi Share (%)", secondary_y=False)
    fig.update_yaxes(title_text="Industry Size (₹ Cr)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

# ------------------------------------------------------------
# TAB 5 — P-TYPE DEEP DIVE
# ------------------------------------------------------------
with tab5:
    st.title("P-Type Deep Dive — Industry vs GO DESi")

    # --------------------------
    # GLOBAL FILTERS
    # --------------------------
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

    # --------------------------
    # INDIAN SWEETS
    # --------------------------
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

    # --------------------------
    # CANDIES & GUM
    # --------------------------
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
        - Primary Category: Product category  
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
