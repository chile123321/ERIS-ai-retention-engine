# ERIS AI Retention Engine

AI/ML service for validating HR data, developing employee-attrition models, packaging approved models, and serving inference through FastAPI.

## Scope

This repository owns data contracts and validation, development-only EDA, reusable preprocessing, model training/evaluation, calibration, threshold selection, explainability, model bundles, and the inference API. React UI, Node.js business services, Supabase migrations, authentication, user management, dashboards, and application reporting are out of scope.

## Current data state

- IBM HR raw data: 1,470 rows and 35 columns; target `Attrition` maps `No=0`, `Yes=1`.
- Candidate feature set v0: at most 25 features.
- Development set: 1,176 rows (`986 No`, `190 Yes`).
- Locked final test: 294 rows (`247 No`, `47 Yes`).

The existing raw data and split are immutable. The locked final test must not be opened or used for EDA, feature decisions, tuning, calibration, or threshold selection. It is reserved for a final, explicitly authorized evaluation.

## Feature strategies

- **V1 Full:** the approved set of up to 25 IBM HR candidate features.
- **V2 Practical:** features that ERIS can reliably collect and represent in its application schema.
- **V3 Minimal:** a small, easy-to-collect and easy-to-standardize feature set.

V2 and V3 remain intentionally undefined until their owners approve them.

## Planned setup

Requires Python 3.11 or 3.12. Dependency installation is intentionally not performed by this scaffold task.

```bash
python -m venv .venv
pip install -e ".[api,ml,dev]"
```

Planned commands:

```bash
make lint
make test
make serve
python scripts/prepare_data.py --help
python scripts/train.py --help
```

## Layout

`configs/` contains versioned contracts and experiment configuration; `src/eris_ml/` contains production code; `scripts/` contains entry points; `notebooks/` is exploration-only; `tests/` contains automated checks; local data and generated artifacts live under `data/` and `artifacts/` and are ignored by Git.

> **Status:** scaffold only. There is no trained model and prediction is unavailable.
