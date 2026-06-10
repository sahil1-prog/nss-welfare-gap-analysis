"""
src/index_builder.py
====================
NSS Challenge 5.2 - District Composite Index Calculator.

Computes:
  1. District Vulnerability Index (DVI)    - multi-dimensional marginalization score
  2. Participation Gap Score  (PGS)        - per-scheme unserved fraction [0, 1]
  3. Outreach Prioritization Score (OPS)   - weighted composite for field targeting

All scores are clipped to [0, 1] and added as new columns to the input DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Configuration ──────────────────────────────────────────────────────────────
DVI_FEATURES = [
    # (column_name, direction)  direction=1 → higher value → higher vulnerability
    ("rural_fraction",        1),
    ("poverty_rate_proxy",    1),
    ("sc_pct",                1),
    ("st_pct",                1),
    ("solid_fuel_pct",        1),   # from NFHS-5
    ("child_stunting_pct",    1),   # from NFHS-5
    # Inverted features (higher value → lower vulnerability)
    ("literacy_rate",        -1),
    ("female_literacy",      -1),
    ("bank_account_pct",     -1),
    ("electricity_pct",      -1),
    ("sanitation_pct",       -1),
]

# Weights for OPS (must sum to 1.0)
OPS_WEIGHTS = {"pgs": 0.40, "dvi": 0.30, "volume": 0.30}


def _safe_min_max(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series; returns zeros if constant."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DISTRICT VULNERABILITY INDEX (DVI)
# ══════════════════════════════════════════════════════════════════════════════
def compute_dvi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the District Vulnerability Index.

    Each feature is min-max normalised to [0, 1].
    Features with direction == -1 are flipped (1 - normalised) so that
    higher component always means higher vulnerability.
    The DVI is the unweighted average of all components.

    Args:
        df: Master DataFrame containing DVI_FEATURES columns.

    Returns:
        DataFrame with added columns:
          - dvi_component_{col} for each feature
          - dvi  (the composite index)
    """
    df = df.copy()
    component_cols = []

    for col, direction in DVI_FEATURES:
        if col not in df.columns:
            print(f"  [DVI] WARNING: column '{col}' not found - skipped.")
            continue
        norm = _safe_min_max(df[col].fillna(df[col].median()))
        if direction == -1:
            norm = 1 - norm
        comp_col = f"dvi_comp_{col}"
        df[comp_col] = norm.round(4)
        component_cols.append(comp_col)

    if component_cols:
        df["dvi"] = df[component_cols].mean(axis=1).clip(0, 1).round(4)
    else:
        df["dvi"] = 0.0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARTICIPATION GAP SCORE (PGS)
# ══════════════════════════════════════════════════════════════════════════════
def compute_pgs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Participation Gap Score for PMUY and PM-KISAN.

    PGS_i = max(0, (Eligible_i - Beneficiaries_i) / Eligible_i)

    A PGS of 0 → fully covered; PGS of 1 → completely uncovered.

    Args:
        df: Master DataFrame with pmuy_eligible, pmuy_connections,
            pmkisan_eligible, pmkisan_beneficiaries columns.

    Returns:
        DataFrame with added columns:
          - pmuy_pgs       (PMUY gap score)
          - pmkisan_pgs    (PM-KISAN gap score)
          - pmuy_unserved  (absolute count of uncovered PMUY beneficiaries)
          - pmkisan_unserved
          - avg_pgs        (average of the two PGS values)
    """
    df = df.copy()

    # PMUY
    pmuy_elig = df["pmuy_eligible"].clip(lower=1)
    df["pmuy_pgs"]      = ((pmuy_elig - df["pmuy_connections"]) / pmuy_elig).clip(0, 1).round(4)
    df["pmuy_unserved"] = (pmuy_elig - df["pmuy_connections"]).clip(lower=0).astype(int)

    # PM-KISAN
    pkisan_elig = df["pmkisan_eligible"].clip(lower=1)
    df["pmkisan_pgs"]      = ((pkisan_elig - df["pmkisan_beneficiaries"]) / pkisan_elig).clip(0, 1).round(4)
    df["pmkisan_unserved"] = (pkisan_elig - df["pmkisan_beneficiaries"]).clip(lower=0).astype(int)

    df["avg_pgs"] = ((df["pmuy_pgs"] + df["pmkisan_pgs"]) / 2).round(4)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTREACH PRIORITIZATION SCORE (OPS)
# ══════════════════════════════════════════════════════════════════════════════
def compute_ops(df: pd.DataFrame,
                pgs_col: str = "avg_pgs",
                weights: dict | None = None) -> pd.DataFrame:
    """
    Compute the Outreach Prioritization Score.

    OPS_i = w_pgs × PGS_i  +  w_dvi × DVI_i  +  w_vol × NormLog(UnservedPop_i)

    Unserved population (volume) is log-transformed to prevent a small number of
    mega-districts from dominating the priority list.

    Args:
        df       : DataFrame with 'dvi', pgs_col, 'pmuy_unserved', 'pmkisan_unserved'.
        pgs_col  : Column to use as the gap score (default: 'avg_pgs').
        weights  : Dict with keys 'pgs', 'dvi', 'volume'. Defaults to OPS_WEIGHTS.

    Returns:
        DataFrame with added columns:
          - ops_volume_norm  (log-normalised unserved population)
          - ops              (the composite prioritization score)
          - ops_rank         (rank 1 = highest priority)
    """
    df = df.copy()
    w  = weights or OPS_WEIGHTS

    # Total unserved population across both schemes
    total_unserved = df["pmuy_unserved"] + df["pmkisan_unserved"]
    log_unserved   = np.log10(total_unserved.clip(lower=1))
    df["ops_volume_norm"] = _safe_min_max(log_unserved).round(4)

    df["ops"] = (
        w["pgs"]    * df[pgs_col]
        + w["dvi"]  * df["dvi"]
        + w["volume"] * df["ops_volume_norm"]
    ).clip(0, 1).round(4)

    df["ops_rank"] = df["ops"].rank(ascending=False, method="min").astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONVENIENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
def build_all_indexes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full index pipeline: DVI → PGS → OPS.

    Args:
        df: Master dataset from data_loader.load_master().

    Returns:
        DataFrame enriched with all composite index columns.
    """
    print("[index_builder] Computing DVI …")
    df = compute_dvi(df)
    print("[index_builder] Computing PGS …")
    df = compute_pgs(df)
    print("[index_builder] Computing OPS …")
    df = compute_ops(df)
    print(f"[index_builder] Done. Shape: {df.shape}")
    return df


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from src.data_loader import load_master

    master = load_master()
    indexed = build_all_indexes(master)
    print("\nIndex summary:")
    print(indexed[["district_name", "state", "dvi", "pmuy_pgs", "pmkisan_pgs", "avg_pgs", "ops", "ops_rank"]]
          .sort_values("ops_rank")
          .head(10)
          .to_string(index=False))
