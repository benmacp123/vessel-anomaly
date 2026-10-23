# Every target runs from committed files; no target needs an account.
# `make help` lists them.

.PHONY: help install setup lock test test-all lint format format-check all

help: ## List the targets and what each writes
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'


# uv manages the venv at venv/ (the portfolio's convention) rather than
# .venv; activate it before the other targets.
export UV_PROJECT_ENVIRONMENT = venv

install: ## Sync venv/ to uv.lock exactly (all groups)
	uv sync --frozen

lock: ## Re-resolve uv.lock after editing pyproject.toml
	uv lock

setup: install ## Same as install


test: ## Run the fast tests with coverage
	PYTHONPATH=src python -m pytest -vv -m "not slow" --cov=vessels --cov=cloud tests/

test-all: ## Run every test, including full-extract runs
	PYTHONPATH=src python -m pytest -vv tests/

lint: ## Pylint the packages; docstrings everywhere, tests included
	PYTHONPATH=src python -m pylint --disable=R,C src/vessels src/cloud
	# Every module, class and function needs a docstring, tests included.
	PYTHONPATH=src python -m pylint --disable=all \
		--enable=missing-module-docstring,missing-class-docstring,missing-function-docstring \
		src scripts tests

format: ## Format with black
	black src/ tests/ scripts/

format-check: ## Check formatting without changing files
	black --check src/ tests/ scripts/

all: install format lint test ## install, format, lint, test
