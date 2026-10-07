# Local typing boundary

`shap/__init__.pyi` describes only the TreeExplainer surface ERIS uses: a binary
XGBoost classifier, raw output, tree-path-dependent perturbations, a float64
contribution matrix and its scalar/array expected value. It is not a complete
stub for every SHAP explainer/model/multi-output option. There is no wildcard
`Any`, `__getattr__`, import suppression or catch-all keyword argument.

The pinned SHAP version has no PEP 561 metadata or published stub distribution.
The declaration is checked against the installed signatures; existing synthetic
explainability/bundle/API tests check shapes, finiteness and raw-margin additivity.
Extend and test this stub before using additional SHAP APIs. It is static-only:
runtime imports still resolve to the installed package, not this directory.

Other dependencies use separately installed versioned stubs. The sklearn stubs
do not declare `StratifiedKFold.n_splits`, so code uses the public equivalent
`get_n_splits()`. Pandas scalar/frame casts describe invariants established by
schema validation and the report constructors; they do not convert data or change
scores. Clone/search casts retain the actual Pipeline type. No source file is
excluded from strict Mypy checking.
