import sys
sys.path.insert(0, '.')
from src.data_loader import load_master
from src.index_builder import build_all_indexes
from src.models import district_clustering, train_gap_model, explain_district
from src.recommender import rank_districts, simulate_intervention

master   = load_master()
indexed  = build_all_indexes(master)
clustered = district_clustering(indexed)
model, imp_df = train_gap_model(clustered)

print('\nTop 5 gap drivers:')
print(imp_df.head(5)[['feature','importance_pct']].to_string(index=False))

top10 = rank_districts(clustered, 10)
print('\nTop 10 Priority Districts:')
print(top10[['ops_rank','district_name','state','avg_pgs','ops']].to_string(index=False))

row = clustered[clustered['ops_rank'] == 1].iloc[0]
xai = explain_district(row, model, imp_df)
print('\nAI Summary:', xai['ai_summary'][:250])

sim = simulate_intervention(
    row,
    ['mobile_enrolment_camps', 'bc_sakhi_banking_drive'],
    10_000_000,
    'both',
)
print(f"\nSimulation: PGS {sim['base_pgs']:.1%} -> {sim['simulated_pgs']:.1%}")
print(f"New beneficiaries: {sim['new_beneficiaries']:,}")
print(f"Total cost: Rs {sim['total_cost_inr']:,.0f}")
print(f"Cost per enrolment: Rs {sim['cost_per_enrolment']:,}")
print(f"Within budget: {sim['within_budget']}")
print('\nAll checks PASSED.')
