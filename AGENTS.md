# Codex working rules

- Never read the locked final test during EDA or model development.
- Never modify raw data or the established split without explicit authorization.
- Put every learned preprocessing operation in a scikit-learn `Pipeline`.
- Fit feature selection, imputation, encoding, scaling, and resampling only inside each training fold.
- Never commit datasets, model artifacts, employee-level predictions, or secrets.
- Never include employee identifiers as model features.
- Record the data version, feature-set version, model configuration, random seed, and library versions for every training result.
- Do not deploy a model without an evaluation report.
- Describe SHAP as an explanation of prediction behavior, never as evidence of causality.
- Change only files relevant to the assigned task.
- Before finishing, run appropriate checks and report changed files and results.
