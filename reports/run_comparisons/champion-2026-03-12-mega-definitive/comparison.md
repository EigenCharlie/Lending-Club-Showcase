# Run Comparison: champion-2026-03-12-mega-definitive

- Generated: 2026-03-13T12:07:33.813286+00:00
- Overall gates pass: `True`
- Conformal promotion pass: `True`
- Conformal statistical warning: `True`
- Artifact coherence pass: `True`
- Fairness absolute (business) pass: `True`
- A/B gate mode: `no_regression`
- A/B no-regression pass: `True`
- A/B significance (diagnostic): `False`

## Gates
- `artifact_coherence`: **PASS**
- `pd_quality`: **PASS**
- `conformal_policy`: **PASS**
- `ab_no_regression`: **PASS**
- `fairness_relative`: **PASS**
- `fairness_absolute_business`: **PASS**
- `survival_quality`: **PASS**
- `export_contracts`: **PASS**

## Artifact Changes
- `data/processed/ifrs9_scenario_summary.parquet`: hash_changed=True, baseline_exists=True, current_exists=True
- `data/processed/model_comparison.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `data/processed/pipeline_summary.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `data/processed/portfolio_robustness_frontier.parquet`: hash_changed=True, baseline_exists=True, current_exists=True
- `data/processed/portfolio_robustness_summary.parquet`: hash_changed=True, baseline_exists=True, current_exists=True
- `models/conformal_lgd_ead_status.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `models/conformal_policy_status.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `models/fairness_audit_status.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `models/governance_status.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `models/survival_summary.pkl`: hash_changed=True, baseline_exists=True, current_exists=True
- `reports/dvc/metrics_summary.json`: hash_changed=True, baseline_exists=True, current_exists=True
- `reports/gpu_benchmark/gpu_bench_meta.json`: hash_changed=True, baseline_exists=True, current_exists=True

## Conformal Diagnostics
- Statistical warnings (non-blocking): `kupiec_pvalue_90`, `kupiec_pvalue_95`, `christoffersen_pvalue_90`, `christoffersen_pvalue_95`
