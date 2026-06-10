"""
app.py - NSS Challenge 5.2
===========================
Welfare Scheme Participation & Gap Analysis
Interactive Streamlit Policy Dashboard

Panels:
  1. 🏠 Executive Summary   - KPIs + National Choropleth Maps
  2. 🗺️  Gap Maps            - DVI / PGS / OPS choropleth explorer
  3. 🔍 District Scorecard  - Per-district analysis + XAI card
  4. 🧩 Cluster Analysis    - Segmentation explorer + cluster playbooks
  5. ⚙️  Impact Simulator   - Budget-aware intervention modelling
  6. 🏆 Priority Ranking    - Top-N districts table + export
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import json
import re

from src.data_loader   import load_master
from src.index_builder import build_all_indexes
from src.models        import district_clustering, train_gap_model, explain_district, CLUSTER_COLORS, CLUSTER_LABELS
from src.recommender   import (
    get_cluster_recommendations, simulate_intervention,
    generate_district_report, rank_districts, INTERVENTIONS
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WelfareScope India | NSS Challenge 5.2",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

ST_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700;800&display=swap');

/* ── Global ───────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

[data-testid="stHeader"] {
    background-color: var(--background-color) !important;
    border-bottom: 1px solid rgba(128, 128, 128, 0.15) !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--secondary-background-color) !important;
    border-right: 1px solid rgba(128, 128, 128, 0.15) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label {
    color: var(--text-color) !important;
    opacity: 0.7;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── KPI Cards ────────────────────────────────────────────────────────────── */
.kpi-card {
    background: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(128, 128, 128, 0.1);
}
.kpi-value { font-size: 2.2rem; font-weight: 800; line-height: 1.1; margin-bottom: 4px; }
.kpi-label {
    font-size: 0.75rem;
    color: var(--text-color);
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.kpi-delta { font-size: 0.8rem; margin-top: 6px; }
.kpi-delta.bad  { color: #f85149; }
.kpi-delta.good { color: #3fb950; }

/* ── Section headers ─────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.5rem; font-weight: 700;
    color: var(--text-color);
    border-left: 4px solid var(--primary-color);
    padding-left: 14px; margin-bottom: 18px; margin-top: 10px;
}

/* ── AI Card ─────────────────────────────────────────────────────────────── */
.ai-card {
    background: linear-gradient(135deg, rgba(31, 111, 235, 0.12) 0%, rgba(31, 111, 235, 0.03) 100%);
    border: 1px solid rgba(31, 111, 235, 0.35);
    border-radius: 14px;
    padding: 20px 24px;
}
.ai-card h4 { color: var(--primary-color); margin-bottom: 10px; }
.ai-card p  { color: var(--text-color); line-height: 1.6; font-size: 0.92rem; }

/* ── Intervention card ───────────────────────────────────────────────────── */
.intv-card {
    background: rgba(128, 128, 128, 0.04);
    border: 1px solid rgba(128, 128, 128, 0.12);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.intv-card h5 { color: var(--primary-color); margin-bottom: 6px; }
.intv-card p  { color: var(--text-color); opacity: 0.8; font-size: 0.82rem; margin: 0; }

/* ── Driver badge ────────────────────────────────────────────────────────── */
.driver-badge {
    background: rgba(128, 128, 128, 0.07);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.driver-badge .feat  { color: var(--primary-color); font-weight: 600; font-size: 0.85rem; }
.driver-badge .pct   { color: #f0883e; font-size: 0.85rem; }

/* ── Metric overrides ────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.12);
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-color) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text-color) !important;
    opacity: 0.7;
}

/* ── Plotly chart overrides via SVG selectors ────────────────────────────── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
.js-plotly-plot .plotly .bglayer rect { fill: transparent !important; }
.js-plotly-plot .plotly .gridlayer path {
    stroke: var(--text-color) !important;
    opacity: 0.1 !important;
}
.js-plotly-plot .plotly .zerolinepath {
    stroke: var(--text-color) !important;
    opacity: 0.25 !important;
}
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .g-xtitle text,
.js-plotly-plot .plotly .g-ytitle text,
.js-plotly-plot .plotly .g-gtitle text,
.js-plotly-plot .plotly .legendtext,
.js-plotly-plot .plotly .cbaxis text,
.js-plotly-plot .plotly .polar text {
    fill: var(--text-color) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    color: var(--text-color) !important;
    opacity: 0.6;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary-color) !important;
    border-bottom-color: var(--primary-color) !important;
    opacity: 1;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(128, 128, 128, 0.2); border-radius: 3px; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⚙️ Loading & processing data …", ttl=3600)
def get_data():
    master    = load_master()
    print("DEBUG: master shape:", master.shape)
    print("DEBUG: master district_names head:\n", master["district_name"].head())
    indexed   = build_all_indexes(master)
    clustered = district_clustering(indexed)
    model, imp_df = train_gap_model(clustered)
    return clustered, model, imp_df


@st.cache_data(show_spinner=False)
def get_geojson():
    geo_path = pathlib.Path(__file__).parent / "data" / "geojson" / "india_districts.geojson"
    if geo_path.exists() and geo_path.stat().st_size > 100_000:
        with open(geo_path, "r") as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🇮🇳 WelfareScope India")
        st.caption("NSS IIT Roorkee · Challenge 5.2 · 2026")
        st.divider()

        page = st.radio(
            "Navigation",
            ["🏠 Executive Summary", "🗺️ Gap Maps", "🔍 District Scorecard",
             "🧩 Cluster Analysis", "⚙️ Impact Simulator", "🏆 Priority Ranking"],
            label_visibility="collapsed",
        )
        st.divider()

        st.markdown("**Filter by State**")
        states = ["All States"] + sorted(df["state"].unique().tolist())
        sel_state = st.selectbox("State", states, label_visibility="collapsed")

        st.markdown("**Scheme**")
        scheme = st.selectbox("Scheme", ["Both", "PMUY", "PM-KISAN"], label_visibility="collapsed")

        st.divider()
        st.caption("Data: Census 2011 · NFHS-5 · MoPNG · MoAFW")
        st.caption("v1.0 - 2026")

    return page, sel_state, scheme


# ══════════════════════════════════════════════════════════════════════════════
# HELPER - KPI card HTML
# ══════════════════════════════════════════════════════════════════════════════
def kpi_card(label: str, value: str, delta: str = "", delta_type: str = "bad",
             color: str = "#58a6ff") -> str:
    delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""


def human_format(n: float) -> str:
    if n >= 1e7:  return f"{n/1e7:.1f} Cr"
    if n >= 1e5:  return f"{n/1e5:.1f} L"
    if n >= 1e3:  return f"{n/1e3:.1f} K"
    return str(int(n))


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 - EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def panel_executive_summary(df: pd.DataFrame, geojson, scheme: str = "Both"):
    st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)

    total_pop           = df["population"].sum()
    pmuy_unserved_tot   = df["pmuy_unserved"].sum()
    pkisan_unserved_tot = df["pmkisan_unserved"].sum()
    avg_dvi_nat         = df["dvi"].mean()
    high_priority_cnt   = (df["ops_rank"] <= 50).sum()

    # Dynamic KPI Cards based on scheme filter
    cards = [
        ("Total Districts",       str(len(df)),                         "",                          "good", "#58a6ff"),
    ]
    if scheme == "Both" or scheme == "PMUY":
        cards.append(("PMUY Unserved HH",      human_format(pmuy_unserved_tot),      "PMUY gap",                  "bad",  "#f85149"))
    if scheme == "Both" or scheme == "PM-KISAN":
        cards.append(("PM-KISAN Unserved",     human_format(pkisan_unserved_tot),    "DBT not received",          "bad",  "#f85149"))
    
    avg_pgs_val = df["pmuy_pgs"].mean() if scheme == "PMUY" else (df["pmkisan_pgs"].mean() if scheme == "PM-KISAN" else df["avg_pgs"].mean())
    pgs_label = f"Avg {scheme} Gap" if scheme != "Both" else "Avg Participation Gap"
    cards.append((pgs_label, f"{avg_pgs_val:.0%}", "national average", "bad", "#f0883e"))
    
    cards.append(("Avg Vulnerability (DVI)", f"{avg_dvi_nat:.2f}",               "0=safe · 1=extreme",        "bad",  "#e3b341"))
    cards.append(("High-Priority Districts", str(high_priority_cnt),              "top-50 by OPS rank",        "bad",  "#bc8cff"))

    # Render KPI cards in a dynamic column layout
    cols = st.columns(len(cards))
    for col, (lbl, val, dlt, dlt_type, clr) in zip(cols, cards):
        with col:
            st.markdown(kpi_card(lbl, val, dlt, dlt_type, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two charts side-by-side
    left, right = st.columns(2)

    with left:
        chart_title = f"National Gap by State - {scheme}" if scheme != "Both" else "National Gap by State - PMUY"
        scheme_col = "pmkisan_pgs" if scheme == "PM-KISAN" else "pmuy_pgs"
        scheme_unserved = "pmkisan_unserved" if scheme == "PM-KISAN" else "pmuy_unserved"
        
        st.markdown(f"**{chart_title}**")
        state_data = (
            df.groupby("state")
            .agg(pgs=(scheme_col, "mean"), unserved=(scheme_unserved, "sum"))
            .reset_index()
            .sort_values("pgs", ascending=False)
            .head(20)
        )
        fig = px.bar(
            state_data, x="pgs", y="state", orientation="h",
            color="pgs",
            color_continuous_scale=["#3fb950", "#e3b341", "#f85149"],
            labels={"pgs": "Avg PGS", "state": ""},
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, height=380, margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="Inter", color="#8b949e"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Cluster Distribution - Districts**")
        cluster_counts = df.groupby(["cluster_label", "cluster_color"])["district_id"].count().reset_index()
        cluster_counts.columns = ["label", "color", "count"]
        fig2 = px.pie(
            cluster_counts, names="label", values="count",
            color="label",
            color_discrete_map={r["label"]: r["color"] for _, r in cluster_counts.iterrows()},
            hole=0.55, template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(font=dict(color="#8b949e", size=11)),
            font=dict(family="Inter"),
        )
        fig2.update_traces(textfont_color="#e6edf3")
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: DVI vs PGS
    y_col = "pmuy_pgs" if scheme == "PMUY" else ("pmkisan_pgs" if scheme == "PM-KISAN" else "avg_pgs")
    st.markdown(f"**District Scatter - Vulnerability (DVI) vs Participation Gap ({scheme if scheme != 'Both' else 'PGS'})**")
    fig3 = px.scatter(
        df, x="dvi", y=y_col,
        color="cluster_label",
        color_discrete_map={v[0]: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
        size="population", size_max=30,
        hover_data=["district_name", "state", "ops_rank"],
        labels={"dvi": "District Vulnerability Index", y_col: f"{scheme if scheme != 'Both' else 'Avg'} Gap Score"},
        template="plotly_dark",
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
        height=400, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(title="", font=dict(color="#8b949e")),
        font=dict(family="Inter", color="#8b949e"),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 - GAP MAPS
# ══════════════════════════════════════════════════════════════════════════════
def panel_gap_maps(df: pd.DataFrame, geojson, scheme: str = "Both"):
    st.markdown('<div class="section-header">🗺️ Interactive Gap Maps</div>', unsafe_allow_html=True)

    default_idx = 3 if scheme == "PMUY" else (4 if scheme == "PM-KISAN" else 0)
    metric = st.selectbox(
        "Map Metric",
        ["avg_pgs", "dvi", "ops", "pmuy_pgs", "pmkisan_pgs"],
        index=default_idx,
        format_func=lambda x: {
            "avg_pgs": "Average Participation Gap (PGS)",
            "dvi"    : "District Vulnerability Index (DVI)",
            "ops"    : "Outreach Prioritization Score (OPS)",
            "pmuy_pgs"   : "PMUY Gap Score",
            "pmkisan_pgs": "PM-KISAN Gap Score",
        }[x],
    )

    color_scales = {
        "avg_pgs"    : "Reds", "dvi": "OrRd", "ops": "YlOrRd",
        "pmuy_pgs"   : "Reds", "pmkisan_pgs": "Blues",
    }

    left, right = st.columns([3, 2])

    with left:
        if geojson and len(geojson.get("features", [])) > 0:
            # Folium choropleth map
            m = folium.Map(location=[22.5, 82.5], zoom_start=5,
                           tiles="CartoDB dark_matter", control_scale=True)
            folium.Choropleth(
                geo_data=geojson,
                name="choropleth",
                data=df,
                columns=["district_name", metric],
                key_on="feature.properties.NAME_2",
                fill_color=color_scales.get(metric, "YlOrRd"),
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=metric.upper(),
                nan_fill_color="#0d1117",
            ).add_to(m)
            st_folium(m, width="100%", height=500, returned_objects=[])
        else:
            # Plotly scatter-geo fallback (uses state-level aggregation)
            st.info("ℹ️ GeoJSON not available - showing state-level bubble map as fallback.")
            state_agg = df.groupby("state").agg(**{metric: (metric, "mean")}).reset_index()
            # rough state centroids (sample)
            centroids = {
                "Uttar Pradesh": (26.8, 80.9), "Bihar": (25.1, 85.3),
                "Rajasthan": (27.0, 74.2), "Madhya Pradesh": (22.7, 77.7),
                "Maharashtra": (19.7, 75.7), "West Bengal": (22.9, 87.8),
                "Gujarat": (22.3, 71.2), "Karnataka": (15.3, 75.7),
                "Tamil Nadu": (11.1, 78.7), "Andhra Pradesh": (15.9, 79.7),
                "Telangana": (17.4, 78.5), "Odisha": (20.9, 84.2),
                "Jharkhand": (23.6, 85.3), "Chhattisgarh": (21.3, 81.9),
                "Assam": (26.2, 92.9), "Punjab": (31.1, 75.3),
                "Haryana": (29.1, 76.1), "Kerala": (10.8, 76.3),
                "Uttarakhand": (30.1, 79.0), "Himachal Pradesh": (31.9, 77.1),
            }
            state_agg["lat"] = state_agg["state"].map(lambda s: centroids.get(s, (20.6, 79.0))[0])
            state_agg["lon"] = state_agg["state"].map(lambda s: centroids.get(s, (20.6, 79.0))[1])
            fig = px.scatter_geo(
                state_agg, lat="lat", lon="lon", color=metric,
                size=metric, hover_name="state",
                color_continuous_scale="YlOrRd",
                scope="asia", template="plotly_dark",
            )
            fig.update_geos(
                center=dict(lat=22.5, lon=82.5), projection_scale=4.5,
                bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#1c2128",
                showocean=True, oceancolor="#0d1117", showlakes=False,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=500,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Top 15 districts by this metric**")
        top15 = df.nlargest(15, metric)[["district_name", "state", metric, "cluster_label"]].reset_index(drop=True)
        top15.index += 1
        top15.columns = ["District", "State", metric.upper(), "Cluster"]
        st.dataframe(top15, use_container_width=True, height=420)

        st.markdown("**Distribution**")
        fig_hist = px.histogram(
            df, x=metric, nbins=40, color_discrete_sequence=["#58a6ff"],
            template="plotly_dark",
            labels={metric: metric.upper()},
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=200, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, font=dict(family="Inter", color="#8b949e"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 - DISTRICT SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
def panel_district_scorecard(df: pd.DataFrame, model, imp_df: pd.DataFrame):
    st.markdown('<div class="section-header">🔍 District Scorecard</div>', unsafe_allow_html=True)

    states   = sorted(df["state"].unique().tolist())
    sel_state = st.selectbox("Select State", states, key="sc_state")
    districts = sorted(df[df["state"] == sel_state]["district_name"].unique().tolist())
    sel_dist  = st.selectbox("Select District", districts, key="sc_dist")

    row = df[df["district_name"] == sel_dist].iloc[0]
    xai = explain_district(row, model, imp_df, top_n=4)

    # Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### {sel_dist} · {sel_state}")
        st.markdown(f"*OPS Rank: **#{int(row['ops_rank'])}** of {len(df)} districts · {row['cluster_label']}*")

    with col2:
        st.markdown(f"""
        <div style="text-align:right">
            <span style="background:{row['cluster_color']};color:#000;padding:4px 12px;
            border-radius:20px;font-size:0.78rem;font-weight:700">
            {row['cluster_label']}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Index gauges
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("DVI",        f"{row['dvi']:.3f}",        help="District Vulnerability Index [0=safe, 1=critical]")
    with c2:
        st.metric("Avg PGS",    f"{row['avg_pgs']:.1%}",    help="Average Participation Gap Score")
    with c3:
        st.metric("OPS",        f"{row['ops']:.3f}",        help="Outreach Prioritization Score")
    with c4:
        st.metric("Population", human_format(row["population"]))

    st.divider()

    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Radar chart - district vs national average
        cats    = ["DVI", "Literacy", "Female\nLiteracy", "Bank\nAccess", "Solid\nFuel", "SC/ST\nShare"]
        dist_vals = [
            row["dvi"],
            row["literacy_rate"],
            row["female_literacy"],
            row["bank_account_pct"],
            row["solid_fuel_pct"],
            (row["sc_pct"] + row["st_pct"]) / 2,
        ]
        nat_vals = [
            df["dvi"].mean(),
            df["literacy_rate"].mean(),
            df["female_literacy"].mean(),
            df["bank_account_pct"].mean(),
            df["solid_fuel_pct"].mean(),
            ((df["sc_pct"] + df["st_pct"]) / 2).mean(),
        ]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=dist_vals + [dist_vals[0]], theta=cats + [cats[0]],
            fill="toself", name=sel_dist,
            line_color="#58a6ff", fillcolor="rgba(88,166,255,0.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=nat_vals + [nat_vals[0]], theta=cats + [cats[0]],
            fill="toself", name="National Avg",
            line_color="#8b949e", fillcolor="rgba(139,148,158,0.08)",
            line_dash="dash",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262d", color="#8b949e"),
                angularaxis=dict(gridcolor="#21262d", color="#8b949e"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=360, margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(font=dict(color="#8b949e")),
            template="plotly_dark",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Coverage bar
        fig_cov = go.Figure(go.Bar(
            x=["PMUY Coverage", "PM-KISAN Coverage"],
            y=[float(row["pmuy_coverage"]) * 100, float(row["pmkisan_coverage"]) * 100],
            marker_color=["#f0883e", "#3fb950"],
            text=[f"{float(row['pmuy_coverage'])*100:.1f}%", f"{float(row['pmkisan_coverage'])*100:.1f}%"],
            textposition="inside", textfont=dict(color="#e6edf3"),
        ))
        fig_cov.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=220, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 100], title="Coverage %", color="#8b949e", gridcolor="#21262d"),
            xaxis=dict(color="#8b949e"),
            font=dict(family="Inter", color="#8b949e"),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    with right_col:
        # XAI Card
        # Convert markdown bold ** to HTML <strong> so it renders correctly inside HTML container
        formatted_summary = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", xai["ai_summary"])
        st.markdown(f"""
        <div class="ai-card">
            <h4>🤖 AI Gap Diagnosis</h4>
            <p>{formatted_summary}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br><strong>Top Gap Drivers</strong>", unsafe_allow_html=True)
        for drv in xai["top_drivers"]:
            feat_label = drv["feature"].replace("_", " ").title()
            val_str    = f"{drv['district_value']:.2f}" if drv["district_value"] is not None else "N/A"
            st.markdown(f"""
            <div class="driver-badge">
                <span class="feat">📌 {feat_label} = {val_str}</span>
                <span class="pct">{drv['importance_pct']:.1f}% weight</span>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="intv-card">
                <h5>💡 {drv['intervention_name']}</h5>
                <p>{drv['intervention_detail']}</p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 - CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def panel_cluster_analysis(df: pd.DataFrame):
    st.markdown('<div class="section-header">🧩 Cluster Segmentation</div>', unsafe_allow_html=True)

    sil = df.attrs.get("silhouette_score", "N/A")
    st.caption(f"K-Means (k=4) · Silhouette Score: {sil} · Features: DVI, PGS, Rurality, Literacy, Banking, ST/SC")

    top, bot = st.columns([3, 2])

    with top:
        fig = px.scatter(
            df, x="dvi", y="avg_pgs",
            color="cluster_label",
            color_discrete_map={v[0]: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
            size="population", size_max=28,
            hover_data=["district_name", "state", "ops_rank"],
            labels={"dvi": "DVI", "avg_pgs": "Avg PGS"},
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(title="", font=dict(color="#8b949e", size=11)),
            font=dict(family="Inter", color="#8b949e"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with bot:
        # Box plot DVI per cluster
        fig2 = px.box(
            df, x="cluster_label", y="dvi",
            color="cluster_label",
            color_discrete_map={v[0]: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
            template="plotly_dark",
            labels={"cluster_label": "", "dvi": "DVI"},
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
            height=380, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(tickangle=-20, tickfont=dict(size=10)),
            font=dict(family="Inter", color="#8b949e"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Cluster playbooks
    st.divider()
    st.markdown("**Policy Playbooks by Cluster**")
    sel_cluster = st.selectbox("Select Cluster", list(range(4)),
                               format_func=lambda i: CLUSTER_LABELS[i][0])
    playbook = get_cluster_recommendations(sel_cluster)
    st.markdown(f"### {playbook['title']}")
    st.info(playbook["strategy"])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommended Actions**")
        for a in playbook["actions"]:
            st.markdown(f"- {a}")
    with col2:
        st.markdown("**KPI Targets**")
        for k, v in playbook["kpi_targets"].items():
            st.markdown(f"- **{k}**: {v}")

    # Cluster-wise table
    st.markdown("**District Summary by Cluster**")
    cluster_summary = (
        df.groupby("cluster_label")
        .agg(
            districts=("district_id", "count"),
            avg_dvi=("dvi", "mean"),
            avg_pgs=("avg_pgs", "mean"),
            total_unserved=("pmuy_unserved", "sum"),
        )
        .reset_index()
        .round(3)
    )
    cluster_summary.columns = ["Cluster", "Districts", "Avg DVI", "Avg PGS", "Total Unserved (PMUY)"]
    st.dataframe(cluster_summary, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 - IMPACT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
def panel_impact_simulator(df: pd.DataFrame, global_scheme: str = "Both"):
    st.markdown('<div class="section-header">⚙️ Intervention Impact Simulator</div>', unsafe_allow_html=True)
    st.caption("Simulate the effect of policy interventions on a district's participation gap and estimate costs.")

    # District selector
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_state = st.selectbox("State", sorted(df["state"].unique()), key="sim_state")
    with c2:
        districts = sorted(df[df["state"] == sel_state]["district_name"].unique())
        sel_dist  = st.selectbox("District", districts, key="sim_dist")
    with c3:
        default_idx = 1 if global_scheme == "PMUY" else (2 if global_scheme == "PM-KISAN" else 0)
        scheme = st.selectbox("Scheme", ["both", "pmuy", "pmkisan"], index=default_idx, key="sim_scheme",
                              format_func=lambda x: {"both":"Both Schemes","pmuy":"PMUY","pmkisan":"PM-KISAN"}[x])

    row = df[df["district_name"] == sel_dist].iloc[0]

    # Intervention selector + budget
    intv_options = {k: v["label"] for k, v in INTERVENTIONS.items()}
    sel_intvs = st.multiselect(
        "Select Interventions",
        list(intv_options.keys()),
        default=["mobile_enrolment_camps"],
        format_func=lambda k: intv_options[k],
    )
    budget = st.slider("Budget (₹ Lakhs)", 5, 500, 100, 5) * 1e5

    if not sel_intvs:
        st.warning("Please select at least one intervention.")
        return

    result = simulate_intervention(row, sel_intvs, budget, scheme=scheme)

    # Results
    st.divider()
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.metric("Current PGS",       f"{result['base_pgs']:.1%}")
    with r2:
        st.metric("Projected PGS",     f"{result['simulated_pgs']:.1%}",
                  delta=f"-{result['gap_reduction_pct']:.1f}%", delta_color="inverse")
    with r3:
        st.metric("New Beneficiaries", human_format(result["new_beneficiaries"]))
    with r4:
        cost_cr = result["total_cost_inr"] / 1e7
        st.metric("Total Cost",        f"₹{cost_cr:.2f} Cr")
    with r5:
        st.metric("Cost / Enrolment",  f"₹{result['cost_per_enrolment']:,}")

    feasibility_color = "#3fb950" if result["within_budget"] else "#f85149"
    feasibility_label = "✅ Within Budget" if result["within_budget"] else "❌ Over Budget"
    st.markdown(f'<p style="color:{feasibility_color};font-weight:700">{feasibility_label}</p>',
                unsafe_allow_html=True)

    # Waterfall chart: gap before → gap after
    fig = go.Figure(go.Waterfall(
        x=["Current Gap"] + [INTERVENTIONS[k]["label"] for k in sel_intvs] + ["Projected Gap"],
        y=[result["base_pgs"]] +
          [-INTERVENTIONS[k]["gap_reduction_pct"] * result["base_pgs"] for k in sel_intvs] +
          [result["simulated_pgs"]],
        measure=["absolute"] + ["relative"] * len(sel_intvs) + ["total"],
        connector={"line": {"color": "#30363d"}},
        decreasing={"marker": {"color": "#3fb950"}},
        increasing={"marker": {"color": "#f85149"}},
        totals={"marker": {"color": "#58a6ff"}},
        text=[f"{v:.1%}" for v in [result["base_pgs"]] +
              [-INTERVENTIONS[k]["gap_reduction_pct"] * result["base_pgs"] for k in sel_intvs] +
              [result["simulated_pgs"]]],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Inter", color="#8b949e"),
        yaxis=dict(title="Participation Gap Score", gridcolor="#21262d"),
        xaxis=dict(tickangle=-15),
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-intervention breakdown table
    st.markdown("**Intervention Breakdown**")
    intv_df = pd.DataFrame(result["interventions_used"])
    if not intv_df.empty:
        intv_df["estimated_cost_inr"] = intv_df["estimated_cost_inr"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(intv_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 - PRIORITY RANKING
# ══════════════════════════════════════════════════════════════════════════════
def panel_priority_ranking(df: pd.DataFrame, scheme: str = "Both"):
    st.markdown('<div class="section-header">🏆 Priority District Ranking</div>', unsafe_allow_html=True)

    top_n = st.slider("Show top N districts", 10, 100, 25, 5)
    top_df = rank_districts(df, top_n=top_n)

    y_metric = "pmuy_pgs" if scheme == "PMUY" else ("pmkisan_pgs" if scheme == "PM-KISAN" else "avg_pgs")
    y_label = f"{scheme} PGS" if scheme != "Both" else "PGS"

    # Colour-coded rank display
    fig = px.bar(
        top_df.head(25), x="ops", y="district_name", orientation="h",
        color=y_metric,
        color_continuous_scale=["#3fb950", "#e3b341", "#f85149"],
        hover_data=["state", "dvi", "pmuy_unserved", "pmkisan_unserved"],
        labels={"ops": "OPS", "district_name": "District", y_metric: y_label},
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
        height=600, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(autorange="reversed", color="#8b949e"),
        xaxis=dict(title="Outreach Prioritization Score", color="#8b949e", gridcolor="#21262d"),
        font=dict(family="Inter", color="#8b949e"),
        coloraxis_colorbar=dict(title=y_label, tickfont=dict(color="#8b949e")),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Top {top_n} Priority Districts - Full Table**")
    display_cols = ["ops_rank", "district_name", "state", "cluster_label", "dvi"]
    rename_dict = {
        "ops_rank": "Rank", "district_name": "District", "state": "State",
        "cluster_label": "Cluster", "dvi": "DVI", "ops": "OPS",
    }
    if scheme == "PMUY":
        display_cols += ["pmuy_pgs", "ops", "pmuy_unserved"]
        rename_dict["pmuy_pgs"] = "PMUY PGS"
        rename_dict["pmuy_unserved"] = "PMUY Unserved"
    elif scheme == "PM-KISAN":
        display_cols += ["pmkisan_pgs", "ops", "pmkisan_unserved"]
        rename_dict["pmkisan_pgs"] = "PMKISAN PGS"
        rename_dict["pmkisan_unserved"] = "PMKISAN Unserved"
    else:
        display_cols += ["avg_pgs", "ops", "pmuy_unserved", "pmkisan_unserved"]
        rename_dict["avg_pgs"] = "Avg PGS"
        rename_dict["pmuy_unserved"] = "PMUY Unserved"
        rename_dict["pmkisan_unserved"] = "PMKISAN Unserved"

    display_cols = [c for c in display_cols if c in top_df.columns]
    st.dataframe(
        top_df[display_cols].rename(columns=rename_dict),
        use_container_width=True, height=500,
    )

    # Download button
    csv_bytes = top_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Priority List (CSV)",
        data=csv_bytes,
        file_name=f"priority_districts_top{top_n}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df, model, imp_df = get_data()
    geojson           = get_geojson()

    page, sel_state, scheme = render_sidebar(df)

    # Theme CSS injection
    st.markdown(ST_THEME_CSS, unsafe_allow_html=True)

    # Apply state filter if selected
    view_df = df if sel_state == "All States" else df[df["state"] == sel_state]

    if page == "🏠 Executive Summary":
        panel_executive_summary(view_df, geojson, scheme)
    elif page == "🗺️ Gap Maps":
        panel_gap_maps(view_df, geojson, scheme)
    elif page == "🔍 District Scorecard":
        panel_district_scorecard(df, model, imp_df)   # always full df for district search
    elif page == "🧩 Cluster Analysis":
        panel_cluster_analysis(view_df)
    elif page == "⚙️ Impact Simulator":
        panel_impact_simulator(df, scheme)            # full df for district selector
    elif page == "🏆 Priority Ranking":
        panel_priority_ranking(view_df, scheme)


if __name__ == "__main__":
    main()
