# API contract

- `GET /health`: process health; works without a model.
- `GET /model-info`: model readiness and version metadata; never loads a model at import time.
- `POST /predict`: returns HTTP 503 until an approved bundle is loaded. It must never emit fabricated predictions.

Prediction request fields will be defined from an approved feature contract. Employee identifiers and secrets are not model inputs.
