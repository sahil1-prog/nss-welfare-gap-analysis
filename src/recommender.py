"""
src/recommender.py
==================
NSS Challenge 5.2 - Policy Recommendation & Intervention Impact Simulator.

Provides:
  1. get_cluster_recommendations()  - Cluster-level policy playbook
  2. simulate_intervention()        - Budget-aware impact simulation
  3. generate_district_report()     - Full policy scorecard for a district
  4. rank_districts()               - Top-N priority districts across India
"""

import numpy as np
import pandas as pd
from typing import Literal

# ── Intervention parameters (evidence-calibrated) ─────────────────────────────
# Each intervention has:
#   gap_reduction_pct  : fraction of PGS it can close (conservative estimate)
#   cost_per_person    : INR cost to enrol one additional beneficiary
#   effort_months      : typical deployment time in months
#   description        : readable description for the dashboard

INTERVENTIONS = {
    "awareness_campaign": {
        "label"             : "📢 Awareness Campaign",
        "gap_reduction_pct" : 0.10,
        "cost_per_person"   : 120,
        "effort_months"     : 2,
        "description"       : "Targeted radio/IVR/wall-painting campaigns in local language explaining scheme eligibility and benefits.",
        "best_for_clusters" : [2],   # Admin/awareness gap
    },
    "mobile_enrolment_camps": {
        "label"             : "🏕️ Mobile Enrolment Camps",
        "gap_reduction_pct" : 0.22,
        "cost_per_person"   : 350,
        "effort_months"     : 3,
        "description"       : "District administration vans with pre-loaded Aadhaar/eKYC kits deployed to remote panchayats.",
        "best_for_clusters" : [0, 1],   # High-need clusters
    },
    "bc_sakhi_banking_drive": {
        "label"             : "🏦 BC Sakhi Banking Drive",
        "gap_reduction_pct" : 0.15,
        "cost_per_person"   : 200,
        "effort_months"     : 4,
        "description"       : "Expand Business Correspondent Sakhi network for last-mile DBT account opening before PM-KISAN push.",
        "best_for_clusters" : [0, 1, 2],
    },
    "shg_women_outreach": {
        "label"             : "👩 SHG Women-First Outreach",
        "gap_reduction_pct" : 0.18,
        "cost_per_person"   : 180,
        "effort_months"     : 3,
        "description"       : "Leverage Self-Help Group networks and Anganwadi workers for door-to-door PMUY registration drives.",
        "best_for_clusters" : [0, 1],
    },
    "grievance_redressal_camps": {
        "label"             : "⚖️ Grievance Redressal Camps",
        "gap_reduction_pct" : 0.08,
        "cost_per_person"   : 90,
        "effort_months"     : 1,
        "description"       : "Block-level camps to fix eKYC failures, duplicate entries, and DBT seeding errors.",
        "best_for_clusters" : [2, 3],
    },
}


# ── Cluster-level policy playbook ──────────────────────────────────────────────
CLUSTER_PLAYBOOKS = {
    0: {
        "title"     : "🔴 Emergency Outreach Cluster",
        "strategy"  : "Maximum ground mobilisation required. Deploy mobile camps + SHG outreach simultaneously.",
        "actions"   : [
            "Declare district as 'High-Priority Welfare District' at state level.",
            "Deploy mobile enrolment camps to every Gram Panchayat within 90 days.",
            "Assign dedicated district nodal officers for PM-KISAN & PMUY tracking.",
            "Convergent drive: PMUY + PM-KISAN + Jan Dhan in a single camp.",
            "Weekly monitoring via NIC MIS dashboard with state-level escalation.",
        ],
        "kpi_targets": {
            "PMUY Coverage in 6 months"     : ">75%",
            "PM-KISAN Seeding Accuracy"     : ">90%",
            "Camp Coverage (Panchayats)"    : "100%",
        },
    },
    1: {
        "title"     : "🟠 Intensify Enrolment Cluster",
        "strategy"  : "Good administrative base exists; intensify last-mile reach with targeted campaigns.",
        "actions"   : [
            "Scale mobile camps to bottom 30% of blocks by coverage.",
            "Run IVR awareness drives in local dialect explaining LPG refill process.",
            "Coordinate with gas agencies for doorstep cylinder delivery for new PMUY connections.",
            "BC Sakhi network expansion for PM-KISAN DBT seeding.",
        ],
        "kpi_targets": {
            "PMUY Coverage in 6 months"     : ">80%",
            "PM-KISAN Coverage in 3 months" : ">85%",
        },
    },
    2: {
        "title"     : "🟡 Administrative Fix Cluster",
        "strategy"  : "Moderate need but significant administrative and data-quality gaps. Focus on de-duplication and grievance redressal.",
        "actions"   : [
            "Conduct full eKYC audit and deduplicate beneficiary lists.",
            "Grievance camps to resolve rejected applications.",
            "Awareness campaigns targeting newly eligible households (SECC exclusions).",
            "Monitor refill rates for PMUY to detect active-but-unused connections.",
        ],
        "kpi_targets": {
            "Data Accuracy (eKYC pass rate)": ">95%",
            "Pending Grievances Resolved"   : "100% within 60 days",
        },
    },
    3: {
        "title"     : "🟢 Monitor & Sustain Cluster",
        "strategy"  : "High performance districts. Focus on quality monitoring and learning dissemination.",
        "actions"   : [
            "Document and publish best practices from high-performing blocks.",
            "Shift resources to neighbouring districts in Clusters 0-2.",
            "Quarterly monitoring audits to prevent regression.",
            "Pilot new digital delivery channels (WhatsApp chatbot, CSC kiosks).",
        ],
        "kpi_targets": {
            "Coverage Maintenance"          : ">90%",
            "Beneficiary Satisfaction"      : ">80% (survey)",
        },
    },
}


def get_cluster_recommendations(cluster_id: int) -> dict:
    """Return the policy playbook for a given cluster ID."""
    return CLUSTER_PLAYBOOKS.get(cluster_id, CLUSTER_PLAYBOOKS[2])


# ══════════════════════════════════════════════════════════════════════════════
# INTERVENTION IMPACT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
def simulate_intervention(
    district_row      : pd.Series,
    intervention_keys : list[str],
    budget_inr        : float,
    scheme            : Literal["pmuy", "pmkisan", "both"] = "both",
) -> dict:
    """
    Simulate the impact of one or more interventions on a district's gap.

    Args:
        district_row      : A row from the indexed master DataFrame.
        intervention_keys : List of keys from INTERVENTIONS dict.
        budget_inr        : Total budget in INR.
        scheme            : Which scheme to simulate ('pmuy', 'pmkisan', 'both').

    Returns:
        dict with:
          - base_pgs          : Current PGS before intervention
          - simulated_pgs     : Projected PGS after intervention
          - gap_reduction     : Absolute gap reduction
          - new_beneficiaries : Estimated additional beneficiaries reached
          - total_cost        : Total program cost (INR)
          - cost_per_enrolment: INR per additional beneficiary
          - interventions_used: List of intervention details
          - feasibility       : 'Within Budget' or 'Over Budget'
    """
    if scheme == "pmuy":
        base_pgs   = float(district_row.get("pmuy_pgs",    0))
        unserved   = int(district_row.get("pmuy_unserved", 0))
    elif scheme == "pmkisan":
        base_pgs   = float(district_row.get("pmkisan_pgs",      0))
        unserved   = int(district_row.get("pmkisan_unserved",   0))
    else:
        base_pgs   = float(district_row.get("avg_pgs",          0))
        unserved   = int(district_row.get("pmuy_unserved",  0)) + \
                     int(district_row.get("pmkisan_unserved", 0))

    # Combine gap reductions (diminishing returns: each additional intervention
    # closes a fraction of the *remaining* gap rather than the full gap)
    combined_reduction = 0.0
    total_cost         = 0.0
    intv_details       = []

    for key in intervention_keys:
        if key not in INTERVENTIONS:
            continue
        intv = INTERVENTIONS[key]
        marginal_pgs_reduction = intv["gap_reduction_pct"] * (1 - combined_reduction)
        
        # Calculate marginal beneficiaries and cost under diminishing returns
        new_beneficiaries_intv = int(unserved * marginal_pgs_reduction)
        intv_cost = new_beneficiaries_intv * intv["cost_per_person"]
        
        combined_reduction    += marginal_pgs_reduction
        total_cost            += intv_cost

        intv_details.append({
            "intervention"      : intv["label"],
            "gap_reduction_pct" : round(intv["gap_reduction_pct"] * 100, 1),
            "new_beneficiaries" : new_beneficiaries_intv,
            "estimated_cost_inr": round(intv_cost),
            "deployment_months" : intv["effort_months"],
        })

    # Clamp reduction to max possible (can't exceed current gap)
    combined_reduction = min(combined_reduction, 1.0)
    new_beneficiaries  = int(unserved * combined_reduction)
    simulated_pgs      = max(0.0, base_pgs * (1 - combined_reduction))
    cost_per_enrolment = round(total_cost / max(new_beneficiaries, 1))

    return {
        "district_name"      : district_row.get("district_name", "N/A"),
        "state"              : district_row.get("state", "N/A"),
        "scheme"             : scheme,
        "base_pgs"           : round(base_pgs, 4),
        "simulated_pgs"      : round(simulated_pgs, 4),
        "gap_reduction_pct"  : round(combined_reduction * 100, 1),
        "new_beneficiaries"  : new_beneficiaries,
        "total_cost_inr"     : round(total_cost),
        "cost_per_enrolment" : cost_per_enrolment,
        "within_budget"      : total_cost <= budget_inr,
        "budget_inr"         : budget_inr,
        "interventions_used" : intv_details,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISTRICT SCORECARD GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_district_report(district_row: pd.Series, xai_card: dict) -> dict:
    """
    Compile a complete policy scorecard for a district.

    Args:
        district_row : Row from the indexed+clustered master DataFrame.
        xai_card     : Output of models.explain_district() for this district.

    Returns:
        dict representing the full district scorecard.
    """
    cluster_id    = int(district_row.get("cluster", 0))
    cluster_recs  = get_cluster_recommendations(cluster_id)

    pmuy_coverage  = 1 - float(district_row.get("pmuy_pgs",    0))
    pkisan_coverage = 1 - float(district_row.get("pmkisan_pgs", 0))

    return {
        "district_name"      : district_row.get("district_name", "N/A"),
        "state"              : district_row.get("state", "N/A"),
        "population"         : int(district_row.get("population", 0)),
        "dvi"                : float(district_row.get("dvi",        0)),
        "avg_pgs"            : float(district_row.get("avg_pgs",    0)),
        "ops"                : float(district_row.get("ops",        0)),
        "ops_rank"           : int(district_row.get("ops_rank",     0)),
        "cluster"            : cluster_id,
        "cluster_label"      : district_row.get("cluster_label",   "N/A"),
        "pmuy_coverage_pct"  : round(pmuy_coverage * 100, 1),
        "pmkisan_coverage_pct": round(pkisan_coverage * 100, 1),
        "pmuy_unserved"      : int(district_row.get("pmuy_unserved",    0)),
        "pmkisan_unserved"   : int(district_row.get("pmkisan_unserved", 0)),
        "ai_summary"         : xai_card.get("ai_summary", ""),
        "top_drivers"        : xai_card.get("top_drivers", []),
        "cluster_playbook"   : cluster_recs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISTRICT RANKING
# ══════════════════════════════════════════════════════════════════════════════
def rank_districts(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Return the top-N priority districts sorted by OPS rank.

    Args:
        df    : Indexed + clustered master DataFrame.
        top_n : Number of districts to return.

    Returns:
        Filtered DataFrame of top-N priority districts.
    """
    cols = [
        "ops_rank", "district_name", "state", "cluster_label",
        "dvi", "avg_pgs", "ops",
        "pmuy_unserved", "pmkisan_unserved",
        "literacy_rate", "bank_account_pct",
    ]
    cols = [c for c in cols if c in df.columns]
    return (
        df[cols]
        .sort_values("ops_rank")
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from src.data_loader    import load_master
    from src.index_builder  import build_all_indexes
    from src.models         import district_clustering, train_gap_model, explain_district

    master    = load_master()
    indexed   = build_all_indexes(master)
    clustered = district_clustering(indexed)
    model, imp_df = train_gap_model(clustered)

    print("\n=== Top 10 Priority Districts ===")
    top10 = rank_districts(clustered, top_n=10)
    print(top10[["ops_rank", "district_name", "state", "avg_pgs", "ops"]].to_string(index=False))

    # Simulate intervention on top district
    top_dist = clustered[clustered["ops_rank"] == 1].iloc[0]
    xai = explain_district(top_dist, model, imp_df)
    sim = simulate_intervention(
        top_dist,
        intervention_keys=["mobile_enrolment_camps", "bc_sakhi_banking_drive"],
        budget_inr=10_000_000,   # ₹1 Crore budget
        scheme="both",
    )
    print(f"\n=== Intervention Simulation: {sim['district_name']} ===")
    print(f"  Base PGS:         {sim['base_pgs']:.2%}")
    print(f"  Simulated PGS:    {sim['simulated_pgs']:.2%}")
    print(f"  New Beneficiaries:{sim['new_beneficiaries']:,}")
    print(f"  Total Cost:       ₹{sim['total_cost_inr']:,.0f}")
    print(f"  Cost/Enrolment:   ₹{sim['cost_per_enrolment']:,}")
    print(f"  Within Budget:    {sim['within_budget']}")
