# Contributing

- `main` phải luôn chạy được.
- Dùng branch `feature/...`, `fix/...`, `docs/...` hoặc `chore/...`.
- Luôn cập nhật bằng `git pull --ff-only`; không push trực tiếp vào `main`.
- Gửi thay đổi qua Pull Request.
- Không commit data, model, prediction, secret hoặc `.env`.
- Dataset mới phải bắt đầu bằng provenance/audit, không bắt đầu bằng training.

Trước khi push:

```bash
uv sync --frozen --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src
git diff --check
```
