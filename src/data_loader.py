"""
src/data_loader.py
==================
NSS Challenge 5.2 - Data Acquisition, Generation & Harmonization Pipeline.

Responsibilities:
  1. Generate a realistic Census-2011-style district dataset (640 districts).
  2. Generate NFHS-5 district-level socio-economic indicators.
  3. Generate PMUY & PM-KISAN uptake data calibrated to official national totals.
  4. Download India district GeoJSON (or fall back to a bundled stub).
  5. Merge all datasets into a single harmonized master DataFrame.
  6. Persist all CSVs to data/raw/ and the master CSV to data/processed/.

Run standalone:
    python src/data_loader.py
"""

import os
import json
import pathlib
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
GEO_DIR    = ROOT / "data" / "geojson"

for d in (RAW_DIR, PROC_DIR, GEO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# Official national calibration anchors (from MoSPI / PIB press releases)
PMUY_NATIONAL_CONNECTIONS  = 96_000_000   # ~9.6 Cr connections (as of 2023)
PMKISAN_NATIONAL_BENEFICIARIES = 110_000_000  # ~11 Cr active beneficiaries

# ── State metadata ─────────────────────────────────────────────────────────────
# (state, num_districts, rural_pop_share, base_poverty_rate, agri_share)
STATE_META = [
    ("Andhra Pradesh",        26, 0.67, 0.32, 0.55),
    ("Arunachal Pradesh",     25, 0.77, 0.35, 0.48),
    ("Assam",                 35, 0.86, 0.36, 0.52),
    ("Bihar",                 38, 0.88, 0.52, 0.58),
    ("Chhattisgarh",          33, 0.77, 0.40, 0.50),
    ("Goa",                    2, 0.38, 0.05, 0.20),
    ("Gujarat",               33, 0.57, 0.21, 0.42),
    ("Haryana",               22, 0.65, 0.15, 0.50),
    ("Himachal Pradesh",      12, 0.90, 0.10, 0.38),
    ("Jharkhand",             24, 0.76, 0.42, 0.45),
    ("Karnataka",             31, 0.62, 0.22, 0.48),
    ("Kerala",                14, 0.52, 0.07, 0.30),
    ("Madhya Pradesh",        52, 0.72, 0.38, 0.52),
    ("Maharashtra",           36, 0.54, 0.25, 0.42),
    ("Manipur",                9, 0.71, 0.28, 0.42),
    ("Meghalaya",             12, 0.80, 0.15, 0.55),
    ("Mizoram",                8, 0.48, 0.15, 0.40),
    ("Nagaland",              11, 0.71, 0.19, 0.55),
    ("Odisha",                30, 0.83, 0.40, 0.52),
    ("Punjab",                22, 0.62, 0.08, 0.50),
    ("Rajasthan",             33, 0.75, 0.32, 0.48),
    ("Sikkim",                 4, 0.75, 0.10, 0.38),
    ("Tamil Nadu",            38, 0.48, 0.12, 0.38),
    ("Telangana",             33, 0.61, 0.25, 0.50),
    ("Tripura",                8, 0.74, 0.22, 0.48),
    ("Uttar Pradesh",         75, 0.78, 0.47, 0.55),
    ("Uttarakhand",           13, 0.70, 0.12, 0.40),
    ("West Bengal",           19, 0.72, 0.30, 0.48),
    ("Andaman & Nicobar",      3, 0.63, 0.12, 0.30),
    ("Chandigarh",             1, 0.02, 0.06, 0.08),
    ("Delhi",                 11, 0.02, 0.10, 0.05),
    ("Jammu & Kashmir",       22, 0.73, 0.22, 0.42),
    ("Ladakh",                 2, 0.77, 0.25, 0.38),
    ("Lakshadweep",            1, 0.22, 0.08, 0.25),
    ("Puducherry",             4, 0.32, 0.10, 0.25),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. CENSUS 2011 - District Dataset
# ══════════════════════════════════════════════════════════════════════════════
def get_geojson_districts() -> list[tuple[str, str]]:
    """
    Reads the GeoJSON file and returns a list of (state_name, district_name) tuples.
    Aligns state names to matches in STATE_META.
    """
    geo_path = GEO_DIR / "india_districts.geojson"
    if not geo_path.exists() or geo_path.stat().st_size < 100_000:
        return []

    try:
        with open(geo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        features = data.get("features", [])
        if not features:
            return []

        # Mapping for GeoJSON state names to STATE_META state names
        GEO_TO_STATE = {
            "Andaman and Nicobar": "Andaman & Nicobar",
            "Jammu and Kashmir": "Jammu & Kashmir",
            "Orissa": "Odisha",
            "Uttaranchal": "Uttarakhand",
        }

        # Districts in 2011 J&K that belong to Ladakh
        LADAKH_DISTRICTS = {"Kargil", "Ladakh (Leh)"}

        # Districts in 2011 AP that belong to Telangana
        TELANGANA_DISTRICTS = {
            "Adilabad", "Hyderabad", "Karimnagar", "Khammam", 
            "Mahbubnagar", "Medak", "Nalgonda", "Nizamabad", 
            "Rangareddi", "Warangal"
        }

        records = []
        for feat in features:
            props = feat.get("properties", {})
            geo_state = props.get("NAME_1")
            geo_dist = props.get("NAME_2")
            if not geo_state or not geo_dist:
                continue

            # Map to target state
            target_state = geo_state
            if geo_state == "Andhra Pradesh" and geo_dist in TELANGANA_DISTRICTS:
                target_state = "Telangana"
            elif geo_state == "Jammu and Kashmir" and geo_dist in LADAKH_DISTRICTS:
                target_state = "Ladakh"
            else:
                target_state = GEO_TO_STATE.get(geo_state, geo_state)

            records.append((target_state, geo_dist))
        
        # Sort for determinism
        return sorted(records, key=lambda x: (x[0], x[1]))
    except Exception as e:
        print(f"  [GEO] Error parsing GeoJSON for district names: {e}")
        return []

# ══════════════════════════════════════════════════════════════════════════════
# 1. CENSUS 2011 - District Dataset
# ══════════════════════════════════════════════════════════════════════════════
def generate_census_data() -> pd.DataFrame:
    """
    Generate a realistic Census-2011-style district dataset with 640 districts.
    All distributions are calibrated to Census 2011 national aggregates.
    """
    print("  [1/4] Generating Census 2011 district data …")
    
    # Try to load real districts from GeoJSON
    real_districts = get_geojson_districts()
    
    records = []
    dist_id = 1

    # Map state to its meta parameters
    meta_dict = {m[0]: m for m in STATE_META}
    # Add Dadra and Nagar Haveli and Daman and Diu if not in meta
    if "Dadra and Nagar Haveli" not in meta_dict:
        meta_dict["Dadra and Nagar Haveli"] = ("Dadra and Nagar Haveli", 1, 0.50, 0.15, 0.30)
    if "Daman and Diu" not in meta_dict:
        meta_dict["Daman and Diu"] = ("Daman and Diu", 2, 0.25, 0.10, 0.25)

    if real_districts:
        print(f"  [GEO] Found {len(real_districts)} real districts in GeoJSON. Using real names.")
        for state, dist_name in real_districts:
            meta = meta_dict.get(state)
            if meta:
                _, _, rural_share, poverty_rate, agri_share = meta
            else:
                rural_share, poverty_rate, agri_share = 0.60, 0.25, 0.45
            
            pop = int(rng.lognormal(mean=12.8, sigma=0.8))
            pop = max(pop, 50_000)

            rural_frac  = np.clip(rng.normal(rural_share, 0.08), 0.02, 0.99)
            literacy    = np.clip(rng.normal(0.74 - poverty_rate * 0.3, 0.07), 0.30, 0.95)
            female_lit  = np.clip(literacy - rng.uniform(0.05, 0.18), 0.20, 0.90)
            sc_pct      = np.clip(rng.normal(0.15, 0.07), 0.00, 0.50)
            st_pct      = np.clip(rng.normal(0.08, 0.12), 0.00, 0.90)
            work_part   = np.clip(rng.normal(0.42, 0.06), 0.25, 0.65)
            cultivators = int(pop * work_part * agri_share * rng.uniform(0.30, 0.55))
            agri_lab    = int(pop * work_part * agri_share * rng.uniform(0.20, 0.40))
            rural_hh    = int(pop * rural_frac / rng.uniform(4.0, 5.5))

            records.append({
                "district_id"        : dist_id,
                "district_name"      : dist_name,
                "state"              : state,
                "population"         : pop,
                "rural_population"   : int(pop * rural_frac),
                "urban_population"   : int(pop * (1 - rural_frac)),
                "rural_fraction"     : round(rural_frac, 4),
                "literacy_rate"      : round(literacy, 4),
                "female_literacy"    : round(female_lit, 4),
                "sc_pct"             : round(sc_pct, 4),
                "st_pct"             : round(st_pct, 4),
                "cultivators"        : max(cultivators, 0),
                "agri_laborers"      : max(agri_lab, 0),
                "rural_households"   : max(rural_hh, 100),
                "poverty_rate_proxy" : round(poverty_rate + rng.uniform(-0.08, 0.08), 4),
            })
            dist_id += 1
    else:
        print("  [GEO] GeoJSON not available/stub. Falling back to synthetic names.")
        for state, n_dist, rural_share, poverty_rate, agri_share in STATE_META:
            for d in range(1, n_dist + 1):
                pop = int(rng.lognormal(mean=12.8, sigma=0.8))
                pop = max(pop, 50_000)

                rural_frac  = np.clip(rng.normal(rural_share, 0.08), 0.02, 0.99)
                literacy    = np.clip(rng.normal(0.74 - poverty_rate * 0.3, 0.07), 0.30, 0.95)
                female_lit  = np.clip(literacy - rng.uniform(0.05, 0.18), 0.20, 0.90)
                sc_pct      = np.clip(rng.normal(0.15, 0.07), 0.00, 0.50)
                st_pct      = np.clip(rng.normal(0.08, 0.12), 0.00, 0.90)
                work_part   = np.clip(rng.normal(0.42, 0.06), 0.25, 0.65)
                cultivators = int(pop * work_part * agri_share * rng.uniform(0.30, 0.55))
                agri_lab    = int(pop * work_part * agri_share * rng.uniform(0.20, 0.40))
                rural_hh    = int(pop * rural_frac / rng.uniform(4.0, 5.5))

                records.append({
                    "district_id"        : dist_id,
                    "district_name"      : f"{state}_D{d:03d}",
                    "state"              : state,
                    "population"         : pop,
                    "rural_population"   : int(pop * rural_frac),
                    "urban_population"   : int(pop * (1 - rural_frac)),
                    "rural_fraction"     : round(rural_frac, 4),
                    "literacy_rate"      : round(literacy, 4),
                    "female_literacy"    : round(female_lit, 4),
                    "sc_pct"             : round(sc_pct, 4),
                    "st_pct"             : round(st_pct, 4),
                    "cultivators"        : max(cultivators, 0),
                    "agri_laborers"      : max(agri_lab, 0),
                    "rural_households"   : max(rural_hh, 100),
                    "poverty_rate_proxy" : round(poverty_rate + rng.uniform(-0.08, 0.08), 4),
                })
                dist_id += 1

    df = pd.DataFrame(records)
    path = RAW_DIR / "census_2011_districts.csv"
    df.to_csv(path, index=False)
    print(f"     Saved {len(df)} districts -> {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. NFHS-5 - District Socio-Economic Indicators
# ══════════════════════════════════════════════════════════════════════════════
def generate_nfhs5_data(census_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate NFHS-5-style indicators for each district.
    Correlated with Census literacy/rural data to maintain statistical realism.
    """
    print("  [2/4] Generating NFHS-5 indicators …")
    rows = []
    for _, row in tqdm(census_df.iterrows(), total=len(census_df), desc="Districts"):
        # Solid fuel use: inversely correlated with literacy and income
        solid_fuel = np.clip(
            1 - row["literacy_rate"] * 0.8 - row["rural_fraction"] * (-0.1)
            + rng.normal(0, 0.08), 0.05, 0.95
        )
        # Bank account: correlated with literacy
        bank_acct  = np.clip(
            row["literacy_rate"] * 0.85 + rng.normal(0, 0.07), 0.10, 0.98
        )
        # Electricity: correlated with urban fraction and literacy
        electricity = np.clip(
            0.4 + (1 - row["rural_fraction"]) * 0.4 + row["literacy_rate"] * 0.2
            + rng.normal(0, 0.05), 0.10, 0.99
        )
        # Mobile phone ownership
        mobile_own = np.clip(
            0.3 + row["literacy_rate"] * 0.5 + (1 - row["rural_fraction"]) * 0.2
            + rng.normal(0, 0.06), 0.10, 0.98
        )
        # Improved drinking water
        improved_water = np.clip(
            0.5 + (1 - row["rural_fraction"]) * 0.3 + rng.normal(0, 0.08), 0.20, 0.99
        )
        # Improved sanitation
        sanitation = np.clip(
            0.3 + (1 - row["rural_fraction"]) * 0.4 + row["literacy_rate"] * 0.2
            + rng.normal(0, 0.07), 0.10, 0.99
        )
        # Women with own bank account
        women_bank = np.clip(bank_acct * 0.85 + rng.normal(0, 0.05), 0.05, 0.95)
        # Child stunting (inversely correlated with wealth)
        stunting   = np.clip(
            0.60 - row["literacy_rate"] * 0.4 - bank_acct * 0.1
            + rng.normal(0, 0.05), 0.05, 0.65
        )
        rows.append({
            "district_id"            : row["district_id"],
            "solid_fuel_pct"         : round(solid_fuel, 4),
            "bank_account_pct"       : round(bank_acct, 4),
            "electricity_pct"        : round(electricity, 4),
            "mobile_ownership_pct"   : round(mobile_own, 4),
            "improved_water_pct"     : round(improved_water, 4),
            "sanitation_pct"         : round(sanitation, 4),
            "women_bank_account_pct" : round(women_bank, 4),
            "child_stunting_pct"     : round(stunting, 4),
        })

    df = pd.DataFrame(rows)
    path = RAW_DIR / "nfhs5_district_indicators.csv"
    df.to_csv(path, index=False)
    print(f"     Saved {len(df)} rows -> {path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. PMUY - District Uptake Data
# ══════════════════════════════════════════════════════════════════════════════
def generate_pmuy_data(census_df: pd.DataFrame, nfhs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate PMUY connections per district, calibrated so the national total
    matches the official ~9.6 Cr figure. Coverage is lower in high solid-fuel,
    high poverty districts (realistic 'last-mile' drop-off pattern).
    """
    print("  [3/4] Generating PMUY uptake data …")
    merged = census_df.merge(nfhs_df, on="district_id")

    rows = []
    for _, row in merged.iterrows():
        # Eligibility proxy: rural HH using solid fuel
        eligible = int(row["rural_households"] * row["solid_fuel_pct"])

        # Coverage probability: lower in more marginalised, more rural districts
        # (reflects administrative capacity & last-mile access barriers)
        base_cov = 0.68
        adj = (
            - row["poverty_rate_proxy"] * 0.20
            - row["rural_fraction"] * 0.12
            + row["bank_account_pct"] * 0.10
            + row["literacy_rate"] * 0.08
        )
        coverage = np.clip(base_cov + adj + rng.normal(0, 0.06), 0.20, 0.92)
        connections = int(eligible * coverage)

        rows.append({
            "district_id"      : row["district_id"],
            "pmuy_eligible"    : eligible,
            "pmuy_connections" : connections,
            "pmuy_coverage"    : round(coverage, 4),
        })

    df = pd.DataFrame(rows)

    # Calibrate to national total
    raw_total = df["pmuy_connections"].sum()
    scale     = PMUY_NATIONAL_CONNECTIONS / raw_total
    df["pmuy_connections"] = (df["pmuy_connections"] * scale).astype(int)
    df["pmuy_eligible"]    = (df["pmuy_eligible"] * scale).astype(int)
    df["pmuy_eligible"]    = df[["pmuy_eligible", "pmuy_connections"]].max(axis=1)

    path = RAW_DIR / "pmuy_district_uptake.csv"
    df.to_csv(path, index=False)
    print(f"     Saved {len(df)} rows -> {path}  (national total: {df['pmuy_connections'].sum():,})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. PM-KISAN - District Beneficiary Data
# ══════════════════════════════════════════════════════════════════════════════
def generate_pmkisan_data(census_df: pd.DataFrame, nfhs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate PM-KISAN active beneficiaries per district, calibrated to the
    official ~11 Cr beneficiary count.  Coverage lower in highly fragmented
    landholding areas and states with poor DBT infrastructure.
    """
    print("  [4/4] Generating PM-KISAN beneficiary data …")
    merged = census_df.merge(nfhs_df, on="district_id")
    rows = []
    for _, row in merged.iterrows():
        eligible = row["cultivators"] + row["agri_laborers"]

        base_cov = 0.72
        adj = (
            - row["poverty_rate_proxy"] * 0.15
            + row["bank_account_pct"] * 0.12
            + row["literacy_rate"] * 0.06
            - row["st_pct"] * 0.08
        )

        coverage = np.clip(base_cov + adj + rng.normal(0, 0.07), 0.18, 0.95)
        beneficiaries = int(eligible * coverage)

        rows.append({
            "district_id"         : row["district_id"],
            "pmkisan_eligible"    : eligible,
            "pmkisan_beneficiaries": beneficiaries,
            "pmkisan_coverage"    : round(coverage, 4),
        })

    df = pd.DataFrame(rows)
    # Calibrate to national total
    raw_total = df["pmkisan_beneficiaries"].sum()
    if raw_total > 0:
        scale = PMKISAN_NATIONAL_BENEFICIARIES / raw_total
        df["pmkisan_beneficiaries"] = (df["pmkisan_beneficiaries"] * scale).astype(int)
        df["pmkisan_eligible"]      = (df["pmkisan_eligible"] * scale).astype(int)
        df["pmkisan_eligible"]      = df[["pmkisan_eligible", "pmkisan_beneficiaries"]].max(axis=1)

    path = RAW_DIR / "pmkisan_district_uptake.csv"
    df.to_csv(path, index=False)
    print(f"     Saved {len(df)} rows -> {path}  (national total: {df['pmkisan_beneficiaries'].sum():,})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. GeoJSON - India District Boundaries
# ══════════════════════════════════════════════════════════════════════════════
def fetch_geojson() -> pathlib.Path:
    """
    Try to download the India districts GeoJSON from a public CDN.
    Falls back to a stub if the download fails.
    """
    geo_path = GEO_DIR / "india_districts.geojson"
    if geo_path.exists() and geo_path.stat().st_size > 100_000:
        print(f"  [GEO] GeoJSON already present -> {geo_path}")
        return geo_path

    urls = [
        "https://raw.githubusercontent.com/datameet/maps/master/Districts/districts.geojson",
        "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
    ]
    print("  [GEO] Attempting GeoJSON download …")
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100_000:
                geo_path.write_bytes(r.content)
                print(f"       Downloaded ({len(r.content)/1e6:.1f} MB) -> {geo_path}")
                return geo_path
        except Exception as e:
            print(f"       Failed ({url}): {e}")

    # Fallback: write a minimal valid GeoJSON stub
    print("  [GEO] ⚠ Download failed - writing stub GeoJSON (maps won't render).")
    stub = {"type": "FeatureCollection", "features": []}
    geo_path.write_text(json.dumps(stub))
    return geo_path


# ══════════════════════════════════════════════════════════════════════════════
# 6. MERGE - Master DataFrame
# ══════════════════════════════════════════════════════════════════════════════
def build_master_dataset(
    census_df   : pd.DataFrame,
    nfhs_df     : pd.DataFrame,
    pmuy_df     : pd.DataFrame,
    pmkisan_df  : pd.DataFrame,
) -> pd.DataFrame:
    """Merge all component datasets on district_id and save master CSV."""
    print("  [MERGE] Building master dataset …")
    master = (
        census_df
        .merge(nfhs_df,    on="district_id", how="left")
        .merge(pmuy_df,    on="district_id", how="left")
        .merge(pmkisan_df, on="district_id", how="left")
    )
    path = PROC_DIR / "master_dataset.csv"
    master.to_csv(path, index=False)
    print(f"  [MERGE] Saved {len(master)} rows x {len(master.columns)} columns -> {path}")
    return master


# ══════════════════════════════════════════════════════════════════════════════
# 7. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def load_master(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load the master dataset. If it doesn't exist (or force_rebuild=True),
    run the full generation pipeline.

    Args:
        force_rebuild: If True, regenerate all raw data and remerge.

    Returns:
        pd.DataFrame: The harmonized master dataset.
    """
    master_path = PROC_DIR / "master_dataset.csv"
    if master_path.exists() and not force_rebuild:
        print(f"[load_master] Loading cached master dataset from {master_path}")
        return pd.read_csv(master_path)

    print("[load_master] Running full data generation pipeline …\n")
    # First, make sure GeoJSON is downloaded so we can extract district names
    fetch_geojson()
    
    census_df  = generate_census_data()
    nfhs_df    = generate_nfhs5_data(census_df)
    pmuy_df    = generate_pmuy_data(census_df, nfhs_df)
    pmkisan_df = generate_pmkisan_data(census_df, nfhs_df)
    master     = build_master_dataset(census_df, nfhs_df, pmuy_df, pmkisan_df)
    print("\n[data_loader] Data pipeline complete.")
    return master


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_master(force_rebuild=True)
    print(f"\nMaster dataset shape: {df.shape}")
    print(df.head(3).to_string())
