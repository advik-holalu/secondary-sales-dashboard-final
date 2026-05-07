import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Secondary Sales Dashboard",
    layout="wide"
)

# ------------------------------------------------------------
# TAB STYLE OVERRIDE (USER GUIDE)
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
# DASHBOARD TITLE
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

# ============================================================
# LOAD PRE-AGGREGATED DATA
# ============================================================

# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------
@st.cache_data
def load_tab1_data():
    return pd.read_parquet(
        "data_agg/tab1_category_trend.parquet"
    )

df = load_tab1_data()

@st.cache_data
def load_tab1_donut_data():
    return pd.read_parquet(
        "data_agg/tab1_quarter_donut.parquet"
    )

donut_df = load_tab1_donut_data()

@st.cache_data
def load_tab1_top_sku_data():
    return pd.read_parquet(
        "data_agg/tab1_top_skus_quarter.parquet"
    )

sku_df = load_tab1_top_sku_data()

@st.cache_data
def load_tab1_state_quarter_data():
    return pd.read_parquet(
        "data_agg/tab1_state_quarter.parquet"
    )

state_q_df = load_tab1_state_quarter_data()

# ------------------------------------------------------------
# TAB 2
# ------------------------------------------------------------
@st.cache_data
def load_tab2_state_month_data():
    return pd.read_parquet(
        "data_agg/tab2_state_month_trend.parquet"
    )

tab2_df = load_tab2_state_month_data()

# ------------------------------------------------------------
# TAB 3
# ------------------------------------------------------------
@st.cache_data
def load_tab3_monthly():
    return pd.read_parquet(
        "data_agg/tab3_state_quarter_month_avg.parquet"
    )

tab3_month_df = load_tab3_monthly()

@st.cache_data
def load_tab3_totals():
    return pd.read_parquet(
        "data_agg/tab3_state_total_alltime.parquet"
    )

tab3_total_df = load_tab3_totals()

# ------------------------------------------------------------
# TAB 4
# ------------------------------------------------------------
@st.cache_data
def load_tab4_data():
    return pd.read_parquet(
        "data_agg/tab4_trend.parquet"
    )

tab4_df = load_tab4_data()

# ------------------------------------------------------------
# TAB 5
# ------------------------------------------------------------
@st.cache_data
def load_tab5_ptype_month():
    return pd.read_parquet(
        "data_agg/tab5_ptype_month.parquet"
    )

@st.cache_data
def load_tab5_ptype_variant():
    return pd.read_parquet(
        "data_agg/tab5_ptype_variant_month.parquet"
    )

@st.cache_data
def load_tab5_ptype_base():
    return pd.read_parquet(
        "data_agg/tab5_ptype_variant_month.parquet"
    )

pt_df = load_tab5_ptype_base()

tab5_month_df = load_tab5_ptype_month()

tab5_variant_df = load_tab5_ptype_variant()

tab5_df = pd.read_parquet(
    "data_agg/tab5_ptype_final.parquet"
)

# ============================================================
# GLOBAL FY LIST
# ============================================================
def _get_all_financial_years(*dfs):

    fys = set()

    for _df in dfs:

        if _df is None or len(_df) == 0:
            continue

        if "FinancialYear" in _df.columns:

            fys.update(
                _df["FinancialYear"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

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
    st.error(
        "No FinancialYear values found in loaded parquet files."
    )
    st.stop()

# ============================================================
# GLOBAL FY FILTER + FILTER NAVIGATION
# ============================================================
with st.sidebar:

    # --------------------------------------------------------
    # DEFAULT = LATEST 2 FYs
    # --------------------------------------------------------
    default_fys = (
        ALL_FY[-2:]
        if len(ALL_FY) >= 2
        else ALL_FY
    )

    selected_fys = st.multiselect(
        "Financial Year",
        options=ALL_FY,
        default=default_fys,
        key="global_financial_year",
    )

    if not selected_fys:
        st.toast(
            "Select at least 1 Financial Year",
            icon="⚠️"
        )
        st.stop()

    st.markdown("---")

    # --------------------------------------------------------
    # FILTER NAVIGATION
    # --------------------------------------------------------
    active_filter_section = st.selectbox(
        "Filter Section",
        [
            "Sales Overview",
            "Top Markets",
            "Growth vs Laggards",
            "Industry View",
            "Product Type Deep Dive"
        ],
        index=0,
        key="active_filter_section"
    )

# ============================================================
# HELPERS
# ============================================================
def filter_by_fy(
    dfin: pd.DataFrame,
    fys: list
) -> pd.DataFrame:

    if dfin is None or dfin.empty:
        return dfin

    if "FinancialYear" not in dfin.columns:
        return dfin

    return (
        dfin[dfin["FinancialYear"].isin(fys)]
        .copy()
    )

# ------------------------------------------------------------
# MONTH ORDER
# ------------------------------------------------------------
def get_month_order(
    dfin: pd.DataFrame
):

    if dfin is None or dfin.empty:
        return []

    if (
        "FYMonthOrder" not in dfin.columns
        or
        "MonthLabel" not in dfin.columns
    ):
        return []

    return (
        dfin[
            ["FYMonthOrder", "MonthLabel"]
        ]
        .drop_duplicates()
        .sort_values("FYMonthOrder")["MonthLabel"]
        .astype(str)
        .tolist()
    )

# ------------------------------------------------------------
# MONTH CATEGORY ORDER
# ------------------------------------------------------------
def enforce_month_order(
    dfin: pd.DataFrame,
    month_order: list
):

    if (
        dfin is None
        or
        dfin.empty
        or
        not month_order
    ):
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
# FY MONTH AXIS (ONLY FOR NON-YOY CHARTS)
# ------------------------------------------------------------
def add_fy_month_axis(
    dfin: pd.DataFrame
):

    if dfin is None or dfin.empty:
        return dfin

    out = dfin.copy()

    out["FYMonthKey"] = (
        out["FinancialYear"].astype(str)
        + "_"
        + out["FYMonthOrder"]
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )

    out["FYMonthLabel"] = (
        out["MonthLabel"].astype(str)
        + " "
        + out["FinancialYear"].astype(str)
    )

    return out

# ============================================================
# NUMBER FORMATTERS
# ============================================================
def format_indian(
    v,
    decimals=1
):

    if (
        v is None
        or
        (
            isinstance(v, float)
            and np.isnan(v)
        )
    ):
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

# ------------------------------------------------------------
# PERCENT FORMATTER
# ------------------------------------------------------------
def format_pct(
    v,
    decimals=1
):

    if (
        v is None
        or
        (
            isinstance(v, float)
            and np.isnan(v)
        )
    ):
        return "0.0%"

    try:
        v = float(v)

    except Exception:
        return "0.0%"

    return f"{v:.{decimals}f}%"

# ============================================================
# OPTIONAL LINE LABEL STYLE
# ============================================================
def apply_line_label_style(
    fig,
    text_size=13
):

    for trace in fig.data:

        trace.textfont = dict(
            color=trace.line.color,
            size=text_size
        )

        trace.textposition = "top center"

    return fig

# ============================================================
# FY DIVIDER (OLD LOGIC)
# ONLY FOR CONTINUOUS FY CHARTS
# ============================================================
def add_fy_divider(
    fig,
    df,
    selected_fys
):

    if len(selected_fys) != 2:
        return fig

    if (
        "FYMonthKey" not in df.columns
        or
        "FinancialYear" not in df.columns
    ):
        return fig

    second_fy = sorted(selected_fys)[1]

    x_order = (
        df["FYMonthKey"]
        .drop_duplicates()
        .tolist()
    )

    second_fy_points = (
        df[
            df["FinancialYear"] == second_fy
        ]["FYMonthKey"]
    )

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
# DEFINE TABS
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
# TAB 1 — SALES OVERVIEW (FINAL YOY + OVERLAID FY LOGIC)
# ============================================================
with tab1:

    # ------------------------------------------------------------
    # SIDEBAR FILTERS
    # ------------------------------------------------------------
    if active_filter_section == "Sales Overview":

        with st.sidebar:

            st.header("Sales Overview Filters")

            metric = st.radio(
                "Metric",
                ["Revenue", "GMV"],
                index=0
            )

            region_sel = st.multiselect(
                "Region",
                sorted(df["Region Name"].dropna().unique())
            )

            if region_sel:
                state_opts = sorted(
                    df[
                        df["Region Name"].isin(region_sel)
                    ]["State Name"].dropna().unique()
                )
            else:
                state_opts = sorted(
                    df["State Name"].dropna().unique()
                )

            state_sel = st.multiselect(
                "State",
                state_opts
            )

            # --------------------------------------------------------
            # DEFAULT CATEGORY = INDIAN SWEETS
            # --------------------------------------------------------
            all_parent_cats = sorted(
                df["Parent Category"].dropna().unique()
            )

            default_parent = (
                ["Indian Sweets"]
                if "Indian Sweets" in all_parent_cats
                else []
            )

            parent_cat_sel = st.multiselect(
                "Parent Category",
                all_parent_cats,
                default=default_parent
            )

            if parent_cat_sel:

                l1_opts = sorted(
                    df[
                        df["Parent Category"].isin(parent_cat_sel)
                    ]["L1 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l1_opts = sorted(
                    df["L1 Category"]
                    .dropna()
                    .unique()
                )

            l1_sel = st.multiselect(
                "L1 Category",
                l1_opts
            )

            if l1_sel:

                l2_opts = sorted(
                    df[
                        df["L1 Category"].isin(l1_sel)
                    ]["L2 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l2_opts = sorted(
                    df["L2 Category"]
                    .dropna()
                    .unique()
                )

            l2_sel = st.multiselect(
                "L2 Category",
                l2_opts
            )

            if l2_sel:

                l3_opts = sorted(
                    df[
                        df["L2 Category"].isin(l2_sel)
                    ]["L3 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l3_opts = sorted(
                    df["L3 Category"]
                    .dropna()
                    .unique()
                )

            l3_sel = st.multiselect(
                "L3 Category",
                l3_opts
            )

            platform_sel = st.multiselect(
                "Platform",
                sorted(
                    df["Platform"]
                    .dropna()
                    .unique()
                )
            )

        # ============================================================
        # FILTER FUNCTION
        # ============================================================
        def apply_tab1_filters(dfin):

            out = dfin.copy()

            if region_sel:
                out = out[
                    out["Region Name"].isin(region_sel)
                ]

            if state_sel:
                out = out[
                    out["State Name"].isin(state_sel)
                ]

            if parent_cat_sel:
                out = out[
                    out["Parent Category"].isin(parent_cat_sel)
                ]

            if l1_sel:
                out = out[
                    out["L1 Category"].isin(l1_sel)
                ]

            if l2_sel:
                out = out[
                    out["L2 Category"].isin(l2_sel)
                ]

            if l3_sel:
                out = out[
                    out["L3 Category"].isin(l3_sel)
                ]

            if platform_sel:
                out = out[
                    out["Platform"].isin(platform_sel)
                ]

            return out

        # ============================================================
        # APPLY FILTERS
        # ============================================================
        df_filt = apply_tab1_filters(df)

        df_filt = df_filt[
            df_filt["FinancialYear"].isin(selected_fys)
        ]

        if df_filt.empty:
            st.warning("No data for selected filters.")
            st.stop()

        month_order = get_month_order(df_filt)

        # ============================================================
        # CATEGORY TREND (OVERLAID YOY STYLE)
        # ============================================================
        st.title("Secondary Sales Overview")

        st.subheader(
            "Parent Category-wise Trend"
        )

        timeline = (
            df_filt
            .groupby(
                [
                    "FinancialYear",
                    "FYMonthOrder",
                    "MonthLabel",
                    "Parent Category"
                ],
                as_index=False
            )[metric]
            .sum()
        )

        timeline = enforce_month_order(
            timeline,
            month_order
        )

        timeline = timeline.sort_values(
            ["FYMonthOrder", "FinancialYear"]
        )

        # ------------------------------------------------------------
        # CREATE LEGEND LABEL
        # EXAMPLE:
        # Indian Sweets | FY2025-26
        # ------------------------------------------------------------
        timeline["LegendLabel"] = (
            timeline["Parent Category"]
            + " | "
            + timeline["FinancialYear"]
        )

        # ------------------------------------------------------------
        # PLOT
        # ------------------------------------------------------------
        fig = px.line(
            timeline,
            x="MonthLabel",
            y=metric,
            color="LegendLabel",
            markers=True,
            text=timeline[metric].apply(format_indian),
            category_orders={
                "MonthLabel": month_order
            }
        )

        fig.update_traces(
            textposition="top center"
        )

        # ------------------------------------------------------------
        # Y AXIS FORMAT
        # ------------------------------------------------------------
        y_max = timeline[metric].max()

        y_ticks = np.linspace(
            0,
            y_max,
            6
        )

        fig.update_yaxes(
            tickvals=y_ticks,
            ticktext=[
                format_indian(v, 0)
                for v in y_ticks
            ]
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title=metric,
            legend_title="Category | FY"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

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
# TAB 2 — TOP MARKETS (FINAL CLEAN VERSION)
# ============================================================
with tab2:

    st.title("Top Markets — State Trends (Top 70% Contribution)")

    # ========================================================
    # APPLY FY FILTER FIRST
    # ========================================================
    df2_master = tab2_df.copy()

    df2_master = df2_master[
        df2_master["FinancialYear"].isin(selected_fys)
    ]

    # ========================================================
    # SIDEBAR FILTERS
    # ========================================================
    if active_filter_section == "Top Markets":

        with st.sidebar:

            st.header("Top Markets Filters")

            # ----------------------------------------------------
            # METRIC
            # ----------------------------------------------------
            metric_tab2 = st.radio(
                "Metric (Tab 2)",
                ["Revenue", "GMV"],
                index=0,
                key="metric_tab2"
            )

            # ----------------------------------------------------
            # DEFAULT CATEGORY
            # ----------------------------------------------------
            parent_options = sorted(
                df2_master["Parent Category"]
                .dropna()
                .unique()
            )

            default_parent = (
                ["Indian Sweets"]
                if "Indian Sweets" in parent_options
                else []
            )

            parent_cat_sel = st.multiselect(
                "Parent Category (Tab 2)",
                parent_options,
                default=default_parent,
                key="parent_tab2"
            )

            # ----------------------------------------------------
            # L1
            # ----------------------------------------------------
            if parent_cat_sel:

                l1_opts = sorted(
                    df2_master[
                        df2_master["Parent Category"].isin(parent_cat_sel)
                    ]["L1 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l1_opts = sorted(
                    df2_master["L1 Category"]
                    .dropna()
                    .unique()
                )

            l1_sel = st.multiselect(
                "L1 Category (Tab 2)",
                l1_opts,
                key="l1_tab2"
            )

            # ----------------------------------------------------
            # L2
            # ----------------------------------------------------
            if l1_sel:

                l2_opts = sorted(
                    df2_master[
                        df2_master["L1 Category"].isin(l1_sel)
                    ]["L2 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l2_opts = sorted(
                    df2_master["L2 Category"]
                    .dropna()
                    .unique()
                )

            l2_sel = st.multiselect(
                "L2 Category (Tab 2)",
                l2_opts,
                key="l2_tab2"
            )

            # ----------------------------------------------------
            # L3
            # ----------------------------------------------------
            if l2_sel:

                l3_opts = sorted(
                    df2_master[
                        df2_master["L2 Category"].isin(l2_sel)
                    ]["L3 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l3_opts = sorted(
                    df2_master["L3 Category"]
                    .dropna()
                    .unique()
                )

            l3_sel = st.multiselect(
                "L3 Category (Tab 2)",
                l3_opts,
                key="l3_tab2"
            )

            # ----------------------------------------------------
            # PLATFORM
            # ----------------------------------------------------
            platform_sel = st.multiselect(
                "Platform (Tab 2)",
                sorted(
                    df2_master["Platform"]
                    .dropna()
                    .unique()
                ),
                key="platform_tab2"
            )

            # ----------------------------------------------------
            # BASELINE QUARTER (FY AWARE)
            # ----------------------------------------------------
            quarter_df = (
                df2_master[
                    ["FinancialYear", "FYQuarter"]
                ]
                .dropna()
                .drop_duplicates()
            )

            quarter_order_map = {
                "Q1": 1,
                "Q2": 2,
                "Q3": 3,
                "Q4": 4
            }

            quarter_df["QuarterOrder"] = (
                quarter_df["FYQuarter"]
                .map(quarter_order_map)
            )

            quarter_df = quarter_df.sort_values(
                ["FinancialYear", "QuarterOrder"]
            )

            quarter_df["QuarterLabel"] = (
                quarter_df["FinancialYear"]
                + " | "
                + quarter_df["FYQuarter"]
            )

            quarter_options = (
                quarter_df["QuarterLabel"]
                .tolist()
            )

            selected_quarter_label = st.selectbox(
                "Baseline Quarter",
                quarter_options,
                index=len(quarter_options) - 1,
                key="baseline_q_tab2"
            )

            baseline_fy = (
                selected_quarter_label.split(" | ")[0]
            )

            baseline_q = (
                selected_quarter_label.split(" | ")[1]
            )

        # ========================================================
        # APPLY FILTERS
        # ========================================================
        df2 = df2_master.copy()

        if parent_cat_sel:
            df2 = df2[
                df2["Parent Category"].isin(parent_cat_sel)
            ]

        if l1_sel:
            df2 = df2[
                df2["L1 Category"].isin(l1_sel)
            ]

        if l2_sel:
            df2 = df2[
                df2["L2 Category"].isin(l2_sel)
            ]

        if l3_sel:
            df2 = df2[
                df2["L3 Category"].isin(l3_sel)
            ]

        if platform_sel:
            df2 = df2[
                df2["Platform"].isin(platform_sel)
            ]

        if df2.empty:
            st.warning("No data available for selected filters.")
            st.stop()

        # ========================================================
        # TOP 70% STATES
        # ========================================================
        def get_top70_states(dfin, metric):

            state_tot = (
                dfin
                .groupby(
                    "State Name",
                    as_index=False,
                    observed=False
                )[metric]
                .sum()
                .sort_values(
                    metric,
                    ascending=False
                )
            )

            total = state_tot[metric].sum()

            state_tot["CumShare"] = (
                state_tot[metric]
                .cumsum()
                / total
                * 100
            )

            top_states = (
                state_tot[
                    state_tot["CumShare"] <= 70
                ]["State Name"]
                .tolist()
            )

            return (
                top_states
                or state_tot.head(1)["State Name"].tolist()
            )

        baseline_df = df2[
            df2["FYQuarter"] == baseline_q
        ]

        top_states = get_top70_states(
            baseline_df,
            metric_tab2
        )

        # ========================================================
        # SHOW TOP STATES
        # ========================================================
        st.markdown("### Top Contributing States")

        st.markdown(
            ", ".join(top_states)
        )

        # ========================================================
        # INTERNAL STATE FILTER
        # ========================================================
        selected_states = st.multiselect(
            "Select States to Display",
            options=top_states,
            default=top_states[:2],
            key="top_market_state_filter"
        )

        if not selected_states:
            st.warning("Select at least one state.")
            st.stop()

        plot_df = df2[
            df2["State Name"].isin(selected_states)
        ]

        # ========================================================
        # MONTH ORDER
        # ========================================================
        month_order = get_month_order(plot_df)

        trend_df = (
            plot_df
            .groupby(
                [
                    "FinancialYear",
                    "FYMonthOrder",
                    "MonthLabel",
                    "State Name"
                ],
                as_index=False,
                observed=False
            )[metric_tab2]
            .sum()
        )

        trend_df = enforce_month_order(
            trend_df,
            month_order
        )

        # ========================================================
        # SORT
        # ========================================================
        trend_df = trend_df.sort_values(
            ["FYMonthOrder", "FinancialYear"]
        )

        # ========================================================
        # LEGEND LABEL
        # ========================================================
        trend_df["LegendLabel"] = (
            trend_df["State Name"]
            + " | "
            + trend_df["FinancialYear"]
        )

        # ========================================================
        # PLOT
        # ========================================================
        fig = px.line(
            trend_df,
            x="MonthLabel",
            y=metric_tab2,
            color="LegendLabel",
            markers=True,
            text=trend_df[metric_tab2].apply(format_indian),
            category_orders={
                "MonthLabel": month_order
            }
        )

        fig.update_traces(
            textposition="top center"
        )

        # ========================================================
        # Y AXIS
        # ========================================================
        y_max = trend_df[metric_tab2].max()

        y_ticks = np.linspace(
            0,
            y_max,
            6
        )

        fig.update_yaxes(
            tickvals=y_ticks,
            ticktext=[
                format_indian(v, decimals=0)
                for v in y_ticks
            ]
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title=metric_tab2,
            legend_title="State | FY",
            height=700
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "States are selected based on the chosen baseline quarter contributing the first 70% of sales and are tracked consistently across all months."
        )

# ============================================================
# TAB 3 — GROWTH VS LAGGARDS (FINAL OVERLAID FY VERSION)
# ============================================================
with tab3:

    st.title("Growth vs Laggard Markets (Top 70% Contribution)")

    # ========================================================
    # APPLY FY FILTER FIRST
    # ========================================================
    df_month_master = (
        tab3_month_df[
            tab3_month_df["FinancialYear"].isin(selected_fys)
        ]
        .copy()
    )

    df_total_master = (
        tab3_total_df[
            tab3_total_df["FinancialYear"].isin(selected_fys)
        ]
        .copy()
    )

    # ========================================================
    # SIDEBAR FILTERS
    # ========================================================
    if active_filter_section == "Growth vs Laggards":

        with st.sidebar:

            st.header("Growth vs Laggards Filters")

            # ----------------------------------------------------
            # METRIC
            # ----------------------------------------------------
            metric = st.radio(
                "Metric (Tab 3)",
                ["Revenue", "GMV"],
                index=0
            )

            # ----------------------------------------------------
            # QUARTER OPTIONS (FY AWARE)
            # ----------------------------------------------------
            quarter_df = (
                df_month_master[
                    ["FinancialYear", "FYQuarter"]
                ]
                .dropna()
                .drop_duplicates()
            )

            quarter_order_map = {
                "Q1": 1,
                "Q2": 2,
                "Q3": 3,
                "Q4": 4
            }

            quarter_df["QuarterOrder"] = (
                quarter_df["FYQuarter"]
                .map(quarter_order_map)
            )

            quarter_df = quarter_df.sort_values(
                ["FinancialYear", "QuarterOrder"]
            )

            quarter_df["QuarterLabel"] = (
                quarter_df["FinancialYear"]
                + " | "
                + quarter_df["FYQuarter"]
            )

            quarter_options = (
                quarter_df["QuarterLabel"]
                .tolist()
            )

            # ----------------------------------------------------
            # COMPARE QUARTER
            # ----------------------------------------------------
            compare_label = st.selectbox(
                "Compare Quarter",
                quarter_options,
                index=len(quarter_options) - 1,
                key="compare_q_tab3"
            )

            # ----------------------------------------------------
            # BASELINE QUARTER
            # ----------------------------------------------------
            baseline_options = [
                q for q in quarter_options
                if q != compare_label
            ]

            if not baseline_options:
                st.warning("No baseline quarter available.")
                st.stop()

            baseline_label = st.selectbox(
                "Baseline Quarter",
                baseline_options,
                index=0,
                key="baseline_q_tab3"
            )

            # ----------------------------------------------------
            # SAFE SPLIT
            # ----------------------------------------------------
            compare_fy, compare_q = compare_label.split(" | ")
            baseline_fy, baseline_q = baseline_label.split(" | ")

            # ----------------------------------------------------
            # DEFAULT CATEGORY
            # ----------------------------------------------------
            parent_options = sorted(
                df_month_master["Parent Category"]
                .dropna()
                .unique()
            )

            default_parent = (
                ["Indian Sweets"]
                if "Indian Sweets" in parent_options
                else []
            )

            parent_sel = st.multiselect(
                "Parent Category (Tab 3)",
                parent_options,
                default=default_parent
            )

            # ----------------------------------------------------
            # L1
            # ----------------------------------------------------
            if parent_sel:

                l1_opts = sorted(
                    df_month_master[
                        df_month_master["Parent Category"].isin(parent_sel)
                    ]["L1 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l1_opts = sorted(
                    df_month_master["L1 Category"]
                    .dropna()
                    .unique()
                )

            l1_sel = st.multiselect(
                "L1 Category (Tab 3)",
                l1_opts
            )

            # ----------------------------------------------------
            # L2
            # ----------------------------------------------------
            if l1_sel:

                l2_opts = sorted(
                    df_month_master[
                        df_month_master["L1 Category"].isin(l1_sel)
                    ]["L2 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l2_opts = sorted(
                    df_month_master["L2 Category"]
                    .dropna()
                    .unique()
                )

            l2_sel = st.multiselect(
                "L2 Category (Tab 3)",
                l2_opts
            )

            # ----------------------------------------------------
            # L3
            # ----------------------------------------------------
            if l2_sel:

                l3_opts = sorted(
                    df_month_master[
                        df_month_master["L2 Category"].isin(l2_sel)
                    ]["L3 Category"]
                    .dropna()
                    .unique()
                )

            else:

                l3_opts = sorted(
                    df_month_master["L3 Category"]
                    .dropna()
                    .unique()
                )

            l3_sel = st.multiselect(
                "L3 Category (Tab 3)",
                l3_opts
            )

            # ----------------------------------------------------
            # PLATFORM
            # ----------------------------------------------------
            platform_sel = st.multiselect(
                "Platform (Tab 3)",
                sorted(
                    df_month_master["Platform"]
                    .dropna()
                    .unique()
                )
            )

        # ========================================================
        # APPLY FILTERS
        # ========================================================
        df_month = df_month_master.copy()
        df_total = df_total_master.copy()

        if parent_sel:
            df_month = df_month[
                df_month["Parent Category"].isin(parent_sel)
            ]
            df_total = df_total[
                df_total["Parent Category"].isin(parent_sel)
            ]

        if l1_sel:
            df_month = df_month[
                df_month["L1 Category"].isin(l1_sel)
            ]
            df_total = df_total[
                df_total["L1 Category"].isin(l1_sel)
            ]

        if l2_sel:
            df_month = df_month[
                df_month["L2 Category"].isin(l2_sel)
            ]
            df_total = df_total[
                df_total["L2 Category"].isin(l2_sel)
            ]

        if l3_sel:
            df_month = df_month[
                df_month["L3 Category"].isin(l3_sel)
            ]
            df_total = df_total[
                df_total["L3 Category"].isin(l3_sel)
            ]

        if platform_sel:
            df_month = df_month[
                df_month["Platform"].isin(platform_sel)
            ]
            df_total = df_total[
                df_total["Platform"].isin(platform_sel)
            ]

        if df_month.empty or df_total.empty:
            st.warning("No data for selected filters.")
            st.stop()

        # ========================================================
        # TOP 70% STATES
        # ========================================================
        baseline_df = df_month[
            (df_month["FinancialYear"] == baseline_fy)
            &
            (df_month["FYQuarter"] == baseline_q)
        ]

        state_rank = (
            baseline_df
            .groupby(
                "State Name",
                as_index=False,
                observed=False
            )[metric]
            .sum()
            .sort_values(
                metric,
                ascending=False
            )
        )

        total_val = state_rank[metric].sum()

        state_rank["CumPct"] = (
            state_rank[metric]
            .cumsum()
            / total_val
        )

        top_states = (
            state_rank[
                state_rank["CumPct"] <= 0.70
            ]["State Name"]
            .tolist()
        )

        if not top_states:
            top_states = (
                state_rank
                .head(1)["State Name"]
                .tolist()
            )

        df_month = df_month[
            df_month["State Name"].isin(top_states)
        ]

        # ========================================================
        # BASELINE vs COMPARE
        # ========================================================
        base_avg = (
            df_month[
                (df_month["FinancialYear"] == baseline_fy)
                &
                (df_month["FYQuarter"] == baseline_q)
            ]
            .groupby(
                "State Name",
                as_index=False,
                observed=False
            )[metric]
            .mean()
            .rename(columns={metric: "Baseline"})
        )

        comp_avg = (
            df_month[
                (df_month["FinancialYear"] == compare_fy)
                &
                (df_month["FYQuarter"] == compare_q)
            ]
            .groupby(
                "State Name",
                as_index=False,
                observed=False
            )[metric]
            .mean()
            .rename(columns={metric: "Compare"})
        )

        growth_df = (
            base_avg
            .merge(
                comp_avg,
                on="State Name",
                how="outer"
            )
            .fillna(0)
        )

        growth_df["Growth %"] = (
            (
                growth_df["Compare"]
                - growth_df["Baseline"]
            )
            /
            growth_df["Baseline"].replace(0, np.nan)
            * 100
        )

        # ========================================================
        # SPLIT
        # ========================================================
        growth_pos = (
            growth_df[
                growth_df["Growth %"] > 0
            ]
            .sort_values(
                "Growth %",
                ascending=False
            )
            .head(5)
        )

        growth_neg = (
            growth_df[
                growth_df["Growth %"] < 0
            ]
            .sort_values("Growth %")
            .head(5)
        )

        # ========================================================
        # VISUALS
        # ========================================================
        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                f"Top Growth — {compare_label} vs {baseline_label}"
            )

            if not growth_pos.empty:

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

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with c2:

            st.subheader(
                f"Top Laggards — {compare_label} vs {baseline_label}"
            )

            if not growth_neg.empty:

                fig = px.bar(
                    growth_neg,
                    x="Growth %",
                    y="State Name",
                    orientation="h",
                    text=growth_neg["Growth %"].apply(format_pct)
                )

                fig.update_traces(
                    textposition="outside"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ========================================================
        # DRILL DOWN
        # ========================================================
        st.markdown("---")

        st.subheader("Monthly Trends — Drill-down")

        month_order = get_month_order(df_month)

        # ========================================================
        # GROWTH STATES
        # ========================================================
        st.markdown("### Growth States — Monthly Trend")

        growth_states = (
            growth_pos["State Name"]
            .tolist()
        )

        if growth_states:

            st.caption(
                "Top Growth States: "
                + ", ".join(growth_states)
            )

            selected_growth_states = st.multiselect(
                "Select Growth States",
                options=growth_states,
                default=growth_states[:2],
                key="growth_state_selector"
            )

            growth_trend = (
                df_month[
                    df_month["State Name"].isin(selected_growth_states)
                ]
                .groupby(
                    [
                        "FinancialYear",
                        "FYMonthOrder",
                        "MonthLabel",
                        "State Name"
                    ],
                    as_index=False,
                    observed=False
                )[metric]
                .sum()
            )

            growth_trend = enforce_month_order(
                growth_trend,
                month_order
            )

            growth_trend = growth_trend.sort_values(
                ["FYMonthOrder", "FinancialYear"]
            )

            growth_trend["LegendLabel"] = (
                growth_trend["State Name"]
                + " | "
                + growth_trend["FinancialYear"]
            )

            fig = px.line(
                growth_trend,
                x="MonthLabel",
                y=metric,
                color="LegendLabel",
                markers=True,
                text=growth_trend[metric].apply(format_indian),
                category_orders={
                    "MonthLabel": month_order
                }
            )

            fig.update_traces(
                textposition="top center"
            )

            y_max = growth_trend[metric].max()

            y_ticks = np.linspace(
                0,
                y_max,
                6
            )

            fig.update_yaxes(
                tickvals=y_ticks,
                ticktext=[
                    format_indian(v, decimals=0)
                    for v in y_ticks
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ========================================================
        # LAGGARD STATES
        # ========================================================
        st.markdown("### Laggard States — Monthly Trend")

        laggard_states = (
            growth_neg["State Name"]
            .tolist()
        )

        if laggard_states:

            st.caption(
                "Top Laggard States: "
                + ", ".join(laggard_states)
            )

            selected_laggard_states = st.multiselect(
                "Select Laggard States",
                options=laggard_states,
                default=laggard_states[:2],
                key="laggard_state_selector"
            )

            laggard_trend = (
                df_month[
                    df_month["State Name"].isin(selected_laggard_states)
                ]
                .groupby(
                    [
                        "FinancialYear",
                        "FYMonthOrder",
                        "MonthLabel",
                        "State Name"
                    ],
                    as_index=False,
                    observed=False
                )[metric]
                .sum()
            )

            laggard_trend = enforce_month_order(
                laggard_trend,
                month_order
            )

            laggard_trend = laggard_trend.sort_values(
                ["FYMonthOrder", "FinancialYear"]
            )

            laggard_trend["LegendLabel"] = (
                laggard_trend["State Name"]
                + " | "
                + laggard_trend["FinancialYear"]
            )

            fig = px.line(
                laggard_trend,
                x="MonthLabel",
                y=metric,
                color="LegendLabel",
                markers=True,
                text=laggard_trend[metric].apply(format_indian),
                category_orders={
                    "MonthLabel": month_order
                }
            )

            fig.update_traces(
                textposition="top center"
            )

            y_max = laggard_trend[metric].max()

            y_ticks = np.linspace(
                0,
                y_max,
                6
            )

            fig.update_yaxes(
                tickvals=y_ticks,
                ticktext=[
                    format_indian(v, decimals=0)
                    for v in y_ticks
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# ============================================================
# TAB 4 — INDUSTRY VIEW (FINAL OVERLAID FY VERSION)
# ============================================================
with tab4:

    st.title("Industry View")

    # ========================================================
    # APPLY FY FILTER FIRST
    # ========================================================
    df4_master = tab4_df.copy()

    df4_master = df4_master[
        df4_master["FinancialYear"].isin(selected_fys)
    ]

    # ========================================================
    # SIDEBAR FILTERS
    # ========================================================
    if active_filter_section == "Industry View":

        with st.sidebar:

            st.header("Industry Filters")

            # ----------------------------------------------------
            # METRIC
            # ----------------------------------------------------
            metric_tab4 = st.radio(
                "Metric",
                ["GMV", "SP"],
                index=0,
                key="tab4_metric"
            )

            # ----------------------------------------------------
            # PLATFORM
            # ----------------------------------------------------
            platform_options = sorted(
                df4_master["Platform"]
                .dropna()
                .unique()
            )

            default_platform = (
                ["Blinkit"]
                if "Blinkit" in platform_options
                else []
            )

            platform_sel = st.multiselect(
                "Platform",
                platform_options,
                default=default_platform
            )

            # ----------------------------------------------------
            # REGION
            # ----------------------------------------------------
            city_options = sorted(
                df4_master["City Name"]
                .dropna()
                .unique()
            )

            default_city = (
                ["PAN India"]
                if "PAN India" in city_options
                else []
            )

            city_sel = st.multiselect(
                "Region",
                city_options,
                default=default_city
            )

            # ----------------------------------------------------
            # CATEGORY
            # ----------------------------------------------------
            category_options = sorted(
                df4_master["Parent Category"]
                .dropna()
                .unique()
            )

            default_category = (
                ["Indian Sweets"]
                if "Indian Sweets" in category_options
                else []
            )

            category_sel = st.multiselect(
                "Parent Category",
                category_options,
                default=default_category
            )

        # ========================================================
        # APPLY FILTERS
        # ========================================================
        df4 = df4_master.copy()

        if platform_sel:
            df4 = df4[
                df4["Platform"].isin(platform_sel)
            ]

        if city_sel:
            df4 = df4[
                df4["City Name"].isin(city_sel)
            ]

        if category_sel:
            df4 = df4[
                df4["Parent Category"].isin(category_sel)
            ]

        if df4.empty:
            st.warning("No data for selected filters.")
            st.stop()

        month_order = get_month_order(df4)

        # ========================================================
        # METRIC SWITCH
        # ========================================================
        if metric_tab4 == "GMV":

            industry_col = "Industry_Size_GMV"
            godesi_col = "GO_DESI_GMV"
            share_col = "Market_Share_GMV"

        else:

            industry_col = "Industry_Size_SP"
            godesi_col = "GO_DESI_SP"
            share_col = "Market_Share_SP"

        # ========================================================
        # GRAPH 1 — MARKET SHARE
        # ========================================================
        st.subheader("GO DESi Market Share Trend (%)")

        share_df = (
            df4
            .groupby(
                [
                    "FinancialYear",
                    "FYMonthOrder",
                    "MonthLabel",
                    "City Name"
                ],
                as_index=False
            )[
                [godesi_col, industry_col]
            ]
            .sum()
        )

        share_df[share_col] = np.where(
            share_df[industry_col] > 0,
            (
                share_df[godesi_col]
                /
                share_df[industry_col]
            ) * 100,
            0
        )

        share_df = enforce_month_order(
            share_df,
            month_order
        )

        share_df = share_df.sort_values(
            ["FYMonthOrder", "FinancialYear"]
        )

        share_df["LegendLabel"] = (
            share_df["City Name"]
            + " | "
            + share_df["FinancialYear"]
        )

        fig = px.line(
            share_df,
            x="MonthLabel",
            y=share_col,
            color="LegendLabel",
            markers=True,
            text=share_df[share_col].apply(format_pct),
            category_orders={
                "MonthLabel": month_order
            }
        )

        fig.update_traces(
            textposition="top center"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ========================================================
        # GRAPH 2 — INDUSTRY SIZE vs GO DESI
        # ========================================================
        st.subheader("Industry Size vs GO DESi")

        bar_df = (
            df4
            .groupby(
                [
                    "FinancialYear",
                    "FYMonthOrder",
                    "MonthLabel"
                ],
                as_index=False
            )[
                [industry_col, godesi_col]
            ]
            .sum()
        )

        bar_df = enforce_month_order(
            bar_df,
            month_order
        )

        bar_df = bar_df.sort_values(
            ["FYMonthOrder", "FinancialYear"]
        )

        bar_df["MonthFY"] = (
            bar_df["MonthLabel"].astype(str)
            + " | "
            + bar_df["FinancialYear"].astype(str)
        )

        bar_df_melt = bar_df.melt(
            id_vars=["MonthFY"],
            value_vars=[industry_col, godesi_col],
            var_name="Type",
            value_name="Value"
        )

        bar_df_melt["Type"] = (
            bar_df_melt["Type"]
            .map({
                industry_col: "Industry Size",
                godesi_col: "GO DESi"
            })
        )

        fig = px.bar(
            bar_df_melt,
            x="MonthFY",
            y="Value",
            color="Type",
            barmode="group",
            text=bar_df_melt["Value"].apply(format_indian)
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ========================================================
        # GRAPH 3 — INDUSTRY TREND
        # ========================================================
        st.subheader("Industry Size Trend")

        ind_df = (
            df4
            .groupby(
                [
                    "FinancialYear",
                    "FYMonthOrder",
                    "MonthLabel",
                    "City Name"
                ],
                as_index=False
            )[industry_col]
            .sum()
        )

        ind_df = enforce_month_order(
            ind_df,
            month_order
        )

        ind_df = ind_df.sort_values(
            ["FYMonthOrder", "FinancialYear"]
        )

        ind_df["LegendLabel"] = (
            ind_df["City Name"]
            + " | "
            + ind_df["FinancialYear"]
        )

        fig = px.line(
            ind_df,
            x="MonthLabel",
            y=industry_col,
            color="LegendLabel",
            markers=True,
            text=ind_df[industry_col].apply(format_indian),
            category_orders={
                "MonthLabel": month_order
            }
        )

        fig.update_traces(
            textposition="top center"
        )

        y_max = ind_df[industry_col].max()

        y_ticks = np.linspace(
            0,
            y_max,
            6
        )

        fig.update_yaxes(
            tickvals=y_ticks,
            ticktext=[
                format_indian(v, decimals=0)
                for v in y_ticks
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ========================================================
        # GRAPH 4 — GO DESI TREND
        # ========================================================
        st.subheader("GO DESi Sales Trend")

        gd_df = (
            df4
            .groupby(
                [
                    "FinancialYear",
                    "FYMonthOrder",
                    "MonthLabel",
                    "City Name"
                ],
                as_index=False
            )[godesi_col]
            .sum()
        )

        gd_df = enforce_month_order(
            gd_df,
            month_order
        )

        gd_df = gd_df.sort_values(
            ["FYMonthOrder", "FinancialYear"]
        )

        gd_df["LegendLabel"] = (
            gd_df["City Name"]
            + " | "
            + gd_df["FinancialYear"]
        )

        fig = px.line(
            gd_df,
            x="MonthLabel",
            y=godesi_col,
            color="LegendLabel",
            markers=True,
            text=gd_df[godesi_col].apply(format_indian),
            category_orders={
                "MonthLabel": month_order
            }
        )

        fig.update_traces(
            textposition="top center"
        )

        y_max = gd_df[godesi_col].max()

        y_ticks = np.linspace(
            0,
            y_max,
            6
        )

        fig.update_yaxes(
            tickvals=y_ticks,
            ticktext=[
                format_indian(v, decimals=0)
                for v in y_ticks
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

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
    if active_filter_section == "Product Type Deep Dive":

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
