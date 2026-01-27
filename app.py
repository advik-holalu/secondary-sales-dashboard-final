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
# LOAD PRE-AGGREGATED DATA (TAB 1 – CHART 1)
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sales Overview",
    "Top Markets",
    "Growth vs Laggards",
    "Metro Industry View",
    "Product Type Deep Dive"
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

        cat_sel = st.multiselect(
            "Category",
            sorted(df["Primary Cat"].dropna().unique())
        )

        platform_sel = st.multiselect(
            "Platform",
            sorted(df["Platform"].dropna().unique())
        )

    # ------------------------------------------------------------
    # APPLY FILTERS
    # ------------------------------------------------------------
    df_filt = df.copy()

    if region_sel:
        df_filt = df_filt[df_filt["Region Name"].isin(region_sel)]
    if state_sel:
        df_filt = df_filt[df_filt["State Name"].isin(state_sel)]
    if cat_sel:
        df_filt = df_filt[df_filt["Primary Cat"].isin(cat_sel)]
    if platform_sel:
        df_filt = df_filt[df_filt["Platform"].isin(platform_sel)]

    if df_filt.empty:
        st.warning("No data for selected filters.")
        st.stop()

    # ------------------------------------------------------------
    # MONTH ORDER (DYNAMIC)
    # ------------------------------------------------------------
    month_order = (
        df_filt[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    # ============================================================
    # SECTION 1 — MOM SALES CHARtT
    # ============================================================

    # ------------------------------------------------------------
    # Q1 BENCHMARK
    # ------------------------------------------------------------
    Q1_KEYS = {"Apr", "May", "Jun"}

    q1 = df_filt[df_filt["MonthLabel"].isin(Q1_KEYS)]
    if q1.empty:
        q1_benchmark = np.nan
    else:
        monthly = (
            q1.groupby(["Year", "MonthLabel"], as_index=False)[metric]
            .sum()
        )
        q1_benchmark = monthly[metric].mean()

    # ------------------------------------------------------------
    # AGG FOR CHART
    # ------------------------------------------------------------
    timeline = (
        df_filt
        .groupby(["Year", "MonthNum", "MonthLabel", "Primary Cat"], as_index=False)[metric]
        .sum()
        .sort_values(["Year", "MonthNum"])
    )

    # ------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------
    st.title("Secondary Sales Overview")
    st.subheader("Category-wise Trend (Dynamic Months) with Q1 Benchmark")

    fig = px.line(
        timeline,
        x="MonthLabel",
        y=metric,
        color="Primary Cat",
        markers=True,
        text=timeline[metric].apply(format_indian),
        category_orders={"MonthLabel": month_order}
    )

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------
    fig.update_yaxes(tickformat=",")

    # ------------------------------------------------------------
    # Q1 Benchmark line
    # ------------------------------------------------------------
    if not np.isnan(q1_benchmark):
        fig.add_hline(
            y=q1_benchmark,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Q1 Avg: {format_indian(q1_benchmark)}",
            annotation_position="top left"
        )

    st.plotly_chart(fig, use_container_width=True)



    # ============================================================
    # SECTION 2 — SALES DISTRIBUTION (Q1 / Q2 / Q3) — DONUT CHARTS
    # ============================================================

    def filter_donut_df(df, quarter, categories, platforms):
        out = df[df["Quarter"] == quarter]

        if categories:
            out = out[out["Primary Cat"].isin(categories)]

        if platforms:
            out = out[out["Platform"].isin(platforms)]

        return out


    st.subheader("Sales Distribution — Q1, Q2, Q3")


    def donut_pair(quarter, title_suffix):
        dfq = filter_donut_df(donut_df, quarter, cat_sel, platform_sel)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"**Region-wise Share — {title_suffix}**")
            reg = dfq.groupby("Region Name", as_index=False)[metric].sum()
            if reg.empty:
                st.info("No data.")
            else:
                fig = px.pie(
                    reg,
                    names="Region Name",
                    values=metric,
                    hole=0.45
                )

                fig.update_traces(
                    text=reg[metric].apply(format_indian),
                    textinfo="text",
                    textposition="inside"
                )

                st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown(f"**State-wise Share — {title_suffix}**")
            stt = dfq.groupby("State Name", as_index=False)[metric].sum()
            if stt.empty:
                st.info("No data.")
            else:
                fig = px.pie(
                    stt,
                    names="State Name",
                    values=metric,
                    hole=0.45
                )

                fig.update_traces(
                    text=stt[metric].apply(format_indian),
                    textinfo="text",
                    textposition="inside"
                )

                st.plotly_chart(fig, use_container_width=True)


    donut_pair("Q1", "Q1 (Apr–Jun)")
    donut_pair("Q2", "Q2 (Jul–Sep)")
    donut_pair("Q3", "Q3 (Oct–Dec)")

    # ============================================================
    # SECTION 3 — TOP 10 SKUs (Q1 / Q2 / Q3)
    # ============================================================

    st.subheader(f"Top 10 SKUs — Quarter-wise ({metric})")


    def render_top_skus_table(quarter, title_suffix):
        st.markdown(f"**{title_suffix}**")

        dfq = sku_df[sku_df["Quarter"] == quarter]

        # Apply SAME filters as Tab 1 (Category + Platform)
        if cat_sel:
            dfq = dfq[dfq["Primary Cat"].isin(cat_sel)]

        if platform_sel:
            dfq = dfq[dfq["Platform"].isin(platform_sel)]

        if dfq.empty:
            st.info("No SKUs match the selected filters.")
            return

        # Sort + pick Top 10
        dfq = (
            dfq
            .sort_values(metric, ascending=False)
            .head(10)
            .copy()
        )

        total_val = dfq[metric].sum()

        dfq["% of Total"] = (
            dfq[metric] / total_val * 100
        ).apply(format_pct)

        # Format metric for display
        dfq[f"{metric} (₹)"] = dfq[metric].apply(format_indian)

        dfq[f"{metric} (₹)"] = dfq[metric].apply(format_indian)

        st.dataframe(
            dfq[
                [
                    "Item Name",
                    "Primary Cat",
                    "Platform",
                    f"{metric} (₹)",
                    "% of Total",
                ]
            ],
            use_container_width=True
        )


    render_top_skus_table("Q1", "Q1 (Apr–Jun)")
    render_top_skus_table("Q2", "Q2 (Jul–Sep)")
    render_top_skus_table("Q3", "Q3 (Oct–Dec)")

    # ============================================================
    # SECTION 4 — STATE PERFORMANCE (Q1 vs Q2 + Q3)
    # ============================================================

    st.subheader("State Performance — Q1 vs Q2 + Q3")

    # ----------------------------
    # Apply filters (Category + Platform)
    # ----------------------------
    df_state = state_q_df.copy()

    if cat_sel:
        df_state = df_state[df_state["Primary Cat"].isin(cat_sel)]

    if platform_sel:
        df_state = df_state[df_state["Platform"].isin(platform_sel)]

    if df_state.empty:
        st.info("No data available for selected filters.")
        st.stop()

    # ----------------------------
    # Quarter-wise state totals
    # ----------------------------
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

    # ----------------------------
    # Merge quarters
    # ----------------------------
    merged = (
        q1.merge(q2, on="State Name", how="outer")
        .merge(q3, on="State Name", how="outer")
        .fillna(0)
    )

    # ----------------------------
    # Derived metrics
    # ----------------------------
    merged["Q2 Δ% vs Q1"] = (
        (merged["Q2"] - merged["Q1"]) /
        merged["Q1"].replace(0, np.nan) * 100
    ).apply(format_pct)

    total_q2 = merged["Q2"].sum()
    merged["Share % (Q2)"] = (
        merged["Q2"] / total_q2 * 100
    ).apply(format_pct)

    # ----------------------------
    # Formatting helpers
    # ----------------------------
    merged["Q1 (₹)"] = merged["Q1"].apply(format_indian)
    merged["Q2 (₹)"] = merged["Q2"].apply(format_indian)
    merged["Q3 (₹)"] = merged["Q3"].apply(format_indian)

    # ----------------------------
    # Display table
    # ----------------------------

    display_df = (
        merged
        .assign(_share_sort=merged["Q2"] / total_q2)
        .sort_values("_share_sort", ascending=False)
        .drop(columns="_share_sort")
    )

    st.dataframe(
        display_df[
            [
                "State Name",
                "Q1 (₹)",
                "Q2 (₹)",
                "Q3 (₹)",
                "Q2 Δ% vs Q1",
                "Share % (Q2)",
            ]
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

        cat_tab2 = st.multiselect(
            "Category (Tab 2)",
            sorted(tab2_df["Primary Cat"].dropna().unique()),
            default=[],
            key="cat_tab2"
        )

        platform_tab2 = st.multiselect(
            "Platform (Tab 2)",
            sorted(tab2_df["Platform"].dropna().unique()),
            default=[],
            key="platform_tab2"
        )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------
    df2 = tab2_df.copy()

    if cat_tab2:
        df2 = df2[df2["Primary Cat"].isin(cat_tab2)]

    if platform_tab2:
        df2 = df2[df2["Platform"].isin(platform_tab2)]

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

    # ========================================================
    # VISUAL 1 — OVERALL TREND (Q1-BASED TOP 70%)
    # ========================================================
    st.subheader("Overall State Trend — Based on Q1 Top 70% States")

    q1_df = df2[df2["Quarter"] == "Q1"]
    top_q1_states = get_top70_states(q1_df, metric_tab2)

    overall_df = df2[df2["State Name"].isin(top_q1_states)]

    month_order = (
        overall_df[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    overall_trend = (
        overall_df
        .groupby(
            ["State Name", "MonthNum", "MonthLabel"],
            as_index=False
        )[metric_tab2]
        .sum()
        .sort_values("MonthNum")
    )

    overall_trend["MonthLabel"] = pd.Categorical(
        overall_trend["MonthLabel"],
        month_order,
        ordered=True
    )

    fig1 = px.line(
        overall_trend,
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True,
        text=overall_trend[metric_tab2].apply(format_indian)
    )

    fig1.update_traces(
        textposition="top center"
    )

    fig1.update_yaxes(tickformat=",")

    st.plotly_chart(fig1, use_container_width=True)

    # ========================================================
    # VISUAL 2 — QUARTER-WISE TREND (INTERNAL SELECTOR)
    # ========================================================
    st.subheader("Quarter-wise State Trend")

    quarter_sel = st.selectbox(
        "Select Quarter",
        sorted(df2["Quarter"].unique()),
        index=0
    )

    q_df = df2[df2["Quarter"] == quarter_sel]

    top_q_states = get_top70_states(q_df, metric_tab2)

    q_plot_df = q_df[q_df["State Name"].isin(top_q_states)]

    month_order_q = (
        q_plot_df[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    q_trend = (
        q_plot_df
        .groupby(
            ["State Name", "MonthNum", "MonthLabel"],
            as_index=False
        )[metric_tab2]
        .sum()
        .sort_values("MonthNum")
    )

    q_trend["MonthLabel"] = pd.Categorical(
        q_trend["MonthLabel"],
        month_order_q,
        ordered=True
    )

    fig2 = px.line(
        q_trend,
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True,
        text=q_trend[metric_tab2].apply(format_indian)
    )

    fig2.update_traces(
        textposition="top center"
    )

    fig2.update_yaxes(tickformat=",")

    st.plotly_chart(fig2, use_container_width=True)

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

        cat_sel = st.multiselect(
            "Category (Tab 3)",
            sorted(tab3_month_df["Primary Cat"].dropna().unique()),
            default=[]
        )

        platform_sel = st.multiselect(
            "Platform (Tab 3)",
            sorted(tab3_month_df["Platform"].dropna().unique()),
            default=[]
        )

    # ----------------------------
    # APPLY FILTERS
    # ----------------------------
    df_month = tab3_month_df.copy()
    df_total = tab3_total_df.copy()

    if cat_sel:
        df_month = df_month[df_month["Primary Cat"].isin(cat_sel)]
        df_total = df_total[df_total["Primary Cat"].isin(cat_sel)]

    if platform_sel:
        df_month = df_month[df_month["Platform"].isin(platform_sel)]
        df_total = df_total[df_total["Platform"].isin(platform_sel)]

    if df_month.empty or df_total.empty:
        st.warning("No data for selected filters.")
        st.stop()

    # ----------------------------
    # TOP 70% STATES (FULL PERIOD)
    # ----------------------------
    state_rank = (
        df_total
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
    # BASELINE vs COMPARE (AVG OF MONTHS)
    # ----------------------------
    base_avg = (
        df_month[df_month["Quarter"] == baseline_q]
        .groupby("State Name", as_index=False)[metric]
        .mean()
        .rename(columns={metric: "Baseline_avg"})
    )

    comp_avg = (
        df_month[df_month["Quarter"] == compare_q]
        .groupby("State Name", as_index=False)[metric]
        .mean()
        .rename(columns={metric: "Compare_avg"})
    )

    growth_df = (
        base_avg.merge(comp_avg, on="State Name", how="outer")
        .fillna(0)
    )

    growth_df["Growth %"] = (
        (growth_df["Compare_avg"] - growth_df["Baseline_avg"]) /
        growth_df["Baseline_avg"].replace(0, np.nan) * 100
    )

    # ----------------------------
    # SPLIT GROWTH / LAGGARDS
    # ----------------------------
    growth_pos = growth_df[growth_df["Growth %"] > 0].sort_values("Growth %", ascending=False).head(5)
    growth_neg = growth_df[growth_df["Growth %"] < 0].sort_values("Growth %").head(5)

    # ----------------------------
    # A. GROWTH BARS
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

            fig.update_traces(
                textposition="outside"
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader(f"Top Laggards — {compare_q} vs {baseline_q}")
        if growth_neg.empty:
            st.info("No laggard states.")
        else:
            ffig = px.bar(
                growth_neg,
                x="Growth %",
                y="State Name",
                orientation="h",
                text=growth_neg["Growth %"].apply(format_pct)
            )

            fig.update_traces(
                textposition="outside"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # B. BENCHMARK SHIFT (DUMBBELL)
    # ----------------------------
    st.subheader(f"Benchmark Shift — {baseline_q} ● vs {compare_q} ○")

    shown = pd.concat([growth_pos, growth_neg]).drop_duplicates("State Name")

    if shown.empty:
        st.info("No states to display.")
    else:
        fig = px.scatter(
            shown,
            x="Baseline_avg",
            y="State Name",
            text=shown["Baseline_avg"].apply(format_indian),
            labels={"Baseline_avg": metric},
        )

        fig.update_traces(
            textposition="middle right"
        )

        fig.add_scatter(
            x=shown["Compare_avg"],
            y=shown["State Name"],
            mode="markers+text",
            text=shown["Compare_avg"].apply(format_indian),
            textposition="middle left",
            marker_symbol="circle-open",
            name=compare_q
        )

        for _, r in shown.iterrows():
            fig.add_shape(
                type="line",
                x0=r["Baseline_avg"],
                x1=r["Compare_avg"],
                y0=r["State Name"],
                y1=r["State Name"],
                line=dict(color="gray")
            )

        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # C. DRILL-DOWN (MONTHLY)
    # ----------------------------
    st.subheader("Drill-down — Monthly Trends")

    month_order = (
        df_month[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    def plot_states(states, title):
        if not states:
            return
        temp = df_month[df_month["State Name"].isin(states)]
        timeline = (
            temp.groupby(["State Name", "MonthNum", "MonthLabel"], as_index=False)[metric]
            .sum()
            .sort_values("MonthNum")
        )
        timeline["MonthLabel"] = pd.Categorical(
            timeline["MonthLabel"], month_order, ordered=True
        )
        fig = px.line(
            timeline,
            x="MonthLabel",
            y=metric,
            color="State Name",
            markers=True,
            text=timeline[metric].apply(format_indian)
        )

        fig.update_traces(
            textposition="top center"
        )

        fig.update_yaxes(tickformat=",")

        st.markdown(f"**{title}**")
        st.plotly_chart(fig, use_container_width=True)

    if not growth_pos.empty:
        sel_g = st.multiselect(
            "Growth States",
            sorted(growth_pos["State Name"]),
            default=sorted(growth_pos["State Name"]),
            key="tab3_growth_states"
        )
        plot_states(sel_g, "Growth States — Monthly")

    if not growth_neg.empty:
        sel_l = st.multiselect(
            "Laggard States",
            sorted(growth_neg["State Name"]),
            default=sorted(growth_neg["State Name"]),
            key="tab3_laggard_states"
        )
        plot_states(sel_l, "Laggard States — Monthly")

# ============================================================
# TAB 4 — METRO INDUSTRY VIEW
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
            ["Blinkit", "Instamart", "Zepto"],
            default=["Blinkit", "Instamart", "Zepto"]
        )

        city_tab4 = st.multiselect(
            "City (Tab 4 – Core)",
            ["Bengaluru", "Mumbai", "Delhi", "Kolkata"],
            default=["Bengaluru", "Mumbai", "Delhi", "Kolkata"]
        )

        category_tab4 = st.multiselect(
            "Category (Tab 4)",
            ["Candies and Confectionary", "Indian Sweets"],
            default=["Candies and Confectionary", "Indian Sweets"]
        )

    # ========================================================
    # GRAPH 1 — GO DESi vs INDUSTRY SIZE
    # ========================================================
    st.subheader("GO DESi vs Industry Size")

    ind_g1 = tab4_industry_core[
        (tab4_industry_core["Year"] == year_tab4)
        & (tab4_industry_core["Platform"].isin(platform_tab4))
        & (tab4_industry_core["City Name"].isin(city_tab4))
    ]

    gmv_g1 = tab4_godesi_gmv[
        (tab4_godesi_gmv["Year"] == year_tab4)
        & (tab4_godesi_gmv["Platform"].isin(platform_tab4))
        & (tab4_godesi_gmv["City Name"].isin(city_tab4))
    ]

    # Optional category filter (only if column exists)
    if "Primary Cat" in ind_g1.columns:
        ind_g1 = ind_g1[ind_g1["Primary Cat"].isin(category_tab4)]

    if "Primary Cat" in gmv_g1.columns:
        gmv_g1 = gmv_g1[gmv_g1["Primary Cat"].isin(category_tab4)]

    ind_m = ind_g1.groupby("Month", as_index=False)["Industry Size"].sum()
    gmv_m = gmv_g1.groupby("Month", as_index=False)["GO DESi GMV"].sum()

    comp = ind_m.merge(gmv_m, on="Month", how="left")
    comp["GO DESi Share %"] = (comp["GO DESi GMV"] / comp["Industry Size"] * 100).round(2)

    fig1 = px.bar(
        comp,
        x="Month",
        y=["Industry Size", "GO DESi GMV"],
        barmode="group",
        title="GO DESi GMV vs Industry Size",
        text_auto=False
    )

    # Add labels manually for Indian format
    for trace in fig1.data:
        trace.text = [format_indian(v) for v in trace.y]
        trace.textposition = "outside"
        trace.hoverinfo = "skip"

    fig1.update_yaxes(tickformat=",")

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
        & (tab4_industry_universe["Platform"].isin(platform_tab4))
    ]

    if "Primary Cat" in ind_g2.columns:
        ind_g2 = ind_g2[ind_g2["Primary Cat"].isin(category_tab4)]

    trend = (
        ind_g2
        .groupby(["Month", "City Name"], as_index=False)["Industry Size"]
        .sum()
    )

    fig2 = px.line(
        trend,
        x="Month",
        y="Industry Size",
        color="City Name",
        markers=True,
        text=trend["Industry Size"].apply(format_indian)
    )

    fig2.update_traces(
        textposition="top center"
    )

    fig2.update_yaxes(tickformat=",")

    st.plotly_chart(fig2, use_container_width=True)

    # ========================================================
    # GRAPH 3 — GO DESi GMV TREND
    # ========================================================
    st.subheader("GO DESi GMV Trend — Core Cities")

    gmv_g3 = tab4_godesi_gmv[
        (tab4_godesi_gmv["Year"] == year_tab4)
        & (tab4_godesi_gmv["Platform"].isin(platform_tab4))
        & (tab4_godesi_gmv["City Name"].isin(city_tab4))
    ]

    if "Primary Cat" in gmv_g3.columns:
        gmv_g3 = gmv_g3[gmv_g3["Primary Cat"].isin(category_tab4)]

    gmv_trend = (
        gmv_g3
        .groupby(["Month", "City Name"], as_index=False)["GO DESi GMV"]
        .sum()
    )

    fig3 = px.line(
        gmv_trend,
        x="Month",
        y="GO DESi GMV",
        color="City Name",
        markers=True,
        text=gmv_trend["GO DESi GMV"].apply(format_indian)
    )

    fig3.update_traces(
        textposition="top center"
    )

    fig3.update_yaxes(tickformat=",")

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