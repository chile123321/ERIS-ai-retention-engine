# Data contract

The versioned machine-readable contract is `configs/data/data_contract_v1.yaml`.
Its API section describes the existing 25-feature serving boundary. Numeric ranges
and nominal categories came from development aggregates, not authoritative HR
rules: `contract_status=dataset_derived`, `hr_business_approved=false`,
`production_approved=false`. Unknown nominal values receive a warning; missing
features are rejected. `record_id` is passthrough only and never a model feature.
Column-level business types, missing-value handling, units, and logical constraints
still require an authoritative HR data dictionary and approval. The web backend
must map its source fields to the unchanged canonical feature schema before calling
the ML API; it must not assume the IBM benchmark boundaries are business policy.
