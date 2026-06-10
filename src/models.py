"""
src/models.py
=============
NSS Challenge 5.2 - Machine Learning & Explainable AI Module.

Provides:
  1. district_clustering()       - K-Means segmentation into policy personas
  2. train_gap_model()           - Random Forest regressor for gap driver analysis
  3. explain_district()          - Per-district XAI feature importance (SHAP-style)
  4. CLUSTER_LABELS              - Human-readable policy labels for each cluster
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# ── Cluster persona labels (assigned after visual inspection of centroids) ─────
CLUSTER_LABELS = {
    0: ("🔴 High Need · Low Coverage",      "Emergency Outreach Priority"),
    1: ("🟠 High Need · Moderate Coverage", "Intensify Enrolment Camps"),
    2: ("🟡 Low Need · Low Coverage",       "Administrative & Awareness Fix"),
    3: ("🟢 Low Need · High Coverage",      "Monitor & Sustain"),
}

CLUSTER_COLORS = {0: "#EF4444", 1: "#F97316", 2: "#EAB308", 3: "#22C55E"}

# Features used for clustering and modeling
CLUSTER_FEATURES = [
    "dvi", "avg_pgs", "rural_fraction", "literacy_rate",
    "female_literacy", "bank_account_pct", "solid_fuel_pct",
    "poverty_rate_proxy", "sc_pct", "st_pct",
]

MODEL_FEATURES = [
    "rural_fraction", "literacy_rate", "female_literacy",
    "sc_pct", "st_pct", "bank_account_pct", "electricity_pct",
    "solid_fuel_pct", "child_stunting_pct", "sanitation_pct",
    "poverty_rate_proxy", "mobile_ownership_pct",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DISTRICT CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
def district_clustering(df: pd.DataFrame,
                        n_clusters: int = 4,
                        features: list[str] | None = None) -> pd.DataFrame:
    """
    Segment districts into policy personas using K-Means clustering.

    Steps:
      - Select available feature columns.
      - StandardScale features.
      - Fit K-Means with fixed random_state for reproducibility.
      - Assign clusters and human-readable labels back to the DataFrame.

    Args:
        df         : Indexed master DataFrame.
        n_clusters : Number of clusters (default 4).
        features   : Feature list to cluster on (defaults to CLUSTER_FEATURES).

    Returns:
        DataFrame with new columns: 'cluster', 'cluster_label', 'cluster_desc',
        'cluster_color', and 'silhouette_score' as an attribute (stored in df.attrs).
    """
    df   = df.copy()
    feat = features or CLUSTER_FEATURES
    feat = [f for f in feat if f in df.columns]

    X = df[feat].fillna(df[feat].median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_scaled)

    # ── Re-label clusters so that higher cluster index → lower vulnerability ──
    # Order by mean DVI of each cluster: cluster with highest avg DVI → label 0
    df["_raw_cluster"] = labels
    cluster_dvi = df.groupby("_raw_cluster")["dvi"].mean().sort_values(ascending=False)
    remap = {old: new for new, old in enumerate(cluster_dvi.index)}
    df["cluster"] = df["_raw_cluster"].map(remap)
    df.drop(columns=["_raw_cluster"], inplace=True)

    df["cluster_label"] = df["cluster"].map(lambda c: CLUSTER_LABELS.get(c, ("Unknown", ""))[0])
    df["cluster_desc"]  = df["cluster"].map(lambda c: CLUSTER_LABELS.get(c, ("", "Unknown"))[1])
    df["cluster_color"] = df["cluster"].map(CLUSTER_COLORS)

    # Silhouette score
    try:
        from sklearn.metrics import silhouette_score
        sil = round(float(silhouette_score(X_scaled, df["cluster"])), 4)
    except Exception:
        sil = None
    df.attrs["silhouette_score"] = sil

    print(f"[models] Clustering done - k={n_clusters}, silhouette={sil}")
    try:
        print(df.groupby("cluster_label")["district_id"].count().to_string())
    except UnicodeEncodeError:
        print(df.groupby("cluster")["district_id"].count().to_string())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. GAP DRIVER MODEL (RANDOM FOREST)
# ══════════════════════════════════════════════════════════════════════════════
def train_gap_model(df: pd.DataFrame,
                    target: str = "avg_pgs",
                    features: list[str] | None = None) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """
    Train a Random Forest regressor to identify the primary drivers of
    participation gaps across districts.

    Args:
        df      : Indexed+clustered master DataFrame.
        target  : Column to predict (default: 'avg_pgs').
        features: Feature columns (defaults to MODEL_FEATURES).

    Returns:
        (fitted_model, feature_importance_df)
        feature_importance_df has columns: ['feature', 'importance', 'rank']
    """
    feat = features or MODEL_FEATURES
    feat = [f for f in feat if f in df.columns]

    X = df[feat].fillna(df[feat].median())
    y = df[target].fillna(df[target].median())

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Cross-validated R²
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"[models] RF trained - CV R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    importance_df = (
        pd.DataFrame({"feature": feat, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["rank"] = importance_df.index + 1
    importance_df["importance_pct"] = (importance_df["importance"] * 100).round(2)

    return model, importance_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. XAI - PER-DISTRICT EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════
_FEATURE_DESCRIPTIONS = {
    "rural_fraction"        : "High rurality limits physical access to enrolment centres.",
    "literacy_rate"         : "Low literacy prevents households from navigating application paperwork.",
    "female_literacy"       : "Low female literacy is a major barrier to PMUY (women-centric) uptake.",
    "sc_pct"                : "High SC population with historical exclusion from formal programmes.",
    "st_pct"                : "High ST population in remote areas with poor administrative reach.",
    "bank_account_pct"      : "Low banking penetration blocks DBT transfers (PM-KISAN).",
    "electricity_pct"       : "Low electricity access correlates with broader infrastructure deficit.",
    "solid_fuel_pct"        : "High solid-fuel use indicates high PMUY need but low transition support.",
    "child_stunting_pct"    : "High stunting reflects acute multi-dimensional deprivation.",
    "sanitation_pct"        : "Low sanitation access signals poor last-mile service delivery.",
    "poverty_rate_proxy"    : "High poverty means more households lack resources to enrol.",
    "mobile_ownership_pct"  : "Low mobile penetration limits digital outreach & grievance redressal.",
}

_INTERVENTION_MAP = {
    "rural_fraction"        : ("Mobile Enrolment Camps",          "Deploy district administration vans for door-step registration."),
    "literacy_rate"         : ("Gram Sabha Awareness Drives",     "Run village council meetings with pictorial scheme guides."),
    "female_literacy"       : ("Women-First Outreach",            "Leverage Self-Help Groups and Anganwadi workers for PMUY."),
    "sc_pct"                : ("Targeted Dalit Outreach",         "Partner with social welfare departments for special drives."),
    "st_pct"                : ("Tribal Block Camps",              "Coordinate with ITDAs for Panchayat-level registration."),
    "bank_account_pct"      : ("BC Sakhi Banking Drive",          "Expand Business Correspondent Sakhi network before PM-KISAN push."),
    "electricity_pct"       : ("Off-Grid Outreach Infrastructure","Partner with RECs/SBI for solar-powered enrolment kiosks."),
    "solid_fuel_pct"        : ("LPG Last-Mile Delivery Push",     "Subsidise first-cylinder delivery and reduce upfront costs."),
    "child_stunting_pct"    : ("Convergent Nutrition-Welfare Camp","Bundle PMUY+ICDS awareness in Nutrition Mela events."),
    "sanitation_pct"        : ("ODF+ Convergence Campaign",       "Leverage Swachh Bharat infrastructure for co-delivery."),
    "poverty_rate_proxy"    : ("Zero-Cost Enrolment Drive",       "Waive all connection/documentation fees for BPL households."),
    "mobile_ownership_pct"  : ("Feature-Phone SMS Campaign",      "Use IVR/SMS in local language via Common Service Centres."),
}


def explain_district(district_row: pd.Series,
                     model: RandomForestRegressor,
                     importance_df: pd.DataFrame,
                     top_n: int = 3) -> dict:
    """
    Generate an Explainable AI diagnosis card for a specific district.

    Uses a sensitivity-based approach: for each of the top-N important features,
    computes how much the district deviates from the national median, and
    assigns a contribution direction (worsening or improving).

    Args:
        district_row  : A single row from the indexed master DataFrame.
        model         : Fitted RandomForestRegressor.
        importance_df : Feature importance table from train_gap_model().
        top_n         : Number of top drivers to explain.

    Returns:
        dict with keys:
          - district_name, state, dvi, avg_pgs, ops, cluster_label
          - top_drivers  : list of dicts (feature, importance_pct, deviation, description, intervention)
          - ai_summary   : Human-readable summary string
    """
    feat_cols   = importance_df["feature"].tolist()
    top_feats   = importance_df.head(top_n)["feature"].tolist()

    # District values vs. national median deviation
    drivers = []
    for feat in top_feats:
        val   = district_row.get(feat, np.nan)
        imp   = float(importance_df.loc[importance_df["feature"] == feat, "importance_pct"].iloc[0])
        desc  = _FEATURE_DESCRIPTIONS.get(feat, "No description available.")
        intv  = _INTERVENTION_MAP.get(feat, ("General Outreach", "Contact district administration."))
        drivers.append({
            "feature"         : feat,
            "district_value"  : round(float(val), 4) if not np.isnan(val) else None,
            "importance_pct"  : round(imp, 2),
            "description"     : desc,
            "intervention_name": intv[0],
            "intervention_detail": intv[1],
        })

    pgs_label = "very high" if district_row.get("avg_pgs", 0) > 0.5 else \
                "high"      if district_row.get("avg_pgs", 0) > 0.35 else "moderate"

    ai_summary = (
        f"District **{district_row.get('district_name', 'N/A')}** ({district_row.get('state', '')}) "
        f"has a {pgs_label} participation gap (PGS = {district_row.get('avg_pgs', 0):.0%}). "
        f"The primary driver is **{drivers[0]['feature'].replace('_', ' ').title()}**, "
        f"contributing {drivers[0]['importance_pct']:.1f}% to the gap model. "
        f"Recommended first action: **{drivers[0]['intervention_name']}** - "
        f"{drivers[0]['intervention_detail']}"
    )

    return {
        "district_name" : district_row.get("district_name", "N/A"),
        "state"         : district_row.get("state", "N/A"),
        "dvi"           : district_row.get("dvi", 0),
        "avg_pgs"       : district_row.get("avg_pgs", 0),
        "ops"           : district_row.get("ops", 0),
        "ops_rank"      : district_row.get("ops_rank", "-"),
        "cluster_label" : district_row.get("cluster_label", "N/A"),
        "top_drivers"   : drivers,
        "ai_summary"    : ai_summary,
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from src.data_loader import load_master
    from src.index_builder import build_all_indexes

    master  = load_master()
    indexed = build_all_indexes(master)
    clustered = district_clustering(indexed)

    model, imp_df = train_gap_model(clustered)
    print("\nTop 5 gap drivers:")
    print(imp_df.head(5).to_string(index=False))

    sample = clustered.iloc[0]
    card = explain_district(sample, model, imp_df)
    print(f"\n=== XAI Card: {card['district_name']} ===")
    print(card["ai_summary"])
