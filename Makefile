# Every target runs from committed files; no target needs an account.
# `make help` lists them.

.PHONY: help install setup lock test test-all lint format format-check report-cleaning coverage closures distance detectors evaluate score golden freeze demo demo-check demo-down all

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

report-cleaning: ## Write results/cleaning/ (rules, segments) for the tuning block
	PYTHONPATH=src python scripts/report_cleaning.py --block tuning

coverage: ## Write results/coverage/ (grid, summary, map) from the tuning block
	PYTHONPATH=src python scripts/build_us_coverage.py

closures: ## Write results/closures/ (towing in closures, cell coverage) from the tuning block
	PYTHONPATH=src python scripts/report_closures.py

distance: ## Write data/reference/us/anchorages.csv and results/distance/ from the tuning block
	PYTHONPATH=src python scripts/build_us_distance.py

detectors: ## Write results/detectors/ (closure, gap, loitering, encounter counts) from the tuning block
	PYTHONPATH=src python scripts/report_detectors.py

evaluate: ## Write results/evaluation/tuning_* (alert rates, planted recall, realism, delay); about 40 min
	PYTHONPATH=src python scripts/evaluate.py --block tuning

score: ## Score the held-out block once, against the committed frozen config (see README)
	PYTHONPATH=src python scripts/evaluate.py --block heldout

freeze: ## Write config/frozen.json and its hash from the code (commit before held-out scoring)
	PYTHONPATH=src python -m vessels.evaluation.config

golden: ## Rewrite tests/golden/*.json from the hand-built scenarios (deliberate changes only)
	PYTHONPATH=src:tests python tests/golden.py

format: ## Format with black
	black src/ tests/ scripts/

format-check: ## Check formatting without changing files
	black --check src/ tests/ scripts/

# The demo runs in Docker on a fresh volume each time; DAYS and SPEED
# override the replay (`make demo DAYS=1 SPEED=3600`).
demo: ## Build the image and replay tuning days through the service; map at http://localhost:8000
	docker compose down --volumes
	docker compose up --build --detach
	@echo "map: http://localhost:8000  (logs: docker compose logs -f processor)"

demo-check: ## Once the demo's processor has exited, check its alerts equal batch
	docker compose run --rm --no-deps processor python scripts/check_demo.py \
		--spool /demo/spool --state /demo/state --alerts /demo/alerts --days $${DAYS:-3}

demo-down: ## Stop the demo and remove its volume
	docker compose down --volumes

all: install format lint test ## install, format, lint, test
