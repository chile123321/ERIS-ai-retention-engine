.PHONY: test lint format typecheck serve
test:
	pytest
lint:
	ruff check .
format:
	ruff format .
typecheck:
	mypy src
serve:
	uvicorn eris_ml.api.main:app --reload
