.PHONY: install test lint build docs-check serve web web-build clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check .

build:
	.venv/bin/python -m build
	.venv/bin/twine check dist/*

docs-check:
	lychee --include-fragments README.md docs

# The API with auto-reload. Needs the web extra; run `make web` beside it.
serve:
	.venv/bin/uvicorn server.asgi:app --host 127.0.0.1 --port 8000 \
		--reload --reload-dir server --reload-dir pipeline

# Vite on :5173, HMR, proxying /api to :8000.
web:
	npm --prefix web run dev

# Write web/dist, which `dub-server` serves.
web-build:
	npm --prefix web run build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov
