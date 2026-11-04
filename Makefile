.PHONY: install test lint build docs-check clean

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

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov
