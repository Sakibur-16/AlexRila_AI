# Development shortcuts. Everything here also runs standalone -- nothing in the
# project requires make.

.DEFAULT_GOAL := help
PYTHON ?= python
IMAGE ?= receipt-ocr:local

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install runtime and development dependencies
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: run
run: ## Run the API locally with reload
	$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: serve
serve: ## Run the development server via the app entrypoint (honours HOST/PORT)
	$(PYTHON) -m app.main

.PHONY: test
test: ## Run the full test suite
	$(PYTHON) -m pytest

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit

.PHONY: test-golden
test-golden: ## Run golden regression tests
	$(PYTHON) -m pytest tests/golden

.PHONY: coverage
coverage: ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov=app --cov-report=term-missing --cov-report=html

.PHONY: lint
lint: ## Check formatting and lint rules
	$(PYTHON) -m ruff check app tests scripts
	$(PYTHON) -m ruff format --check app tests scripts

.PHONY: format
format: ## Apply formatting and safe lint fixes
	$(PYTHON) -m ruff format app tests scripts
	$(PYTHON) -m ruff check --fix app tests scripts

.PHONY: typecheck
typecheck: ## Run static type checking
	$(PYTHON) -m mypy app

.PHONY: golden
golden: ## Regenerate golden expectations (review the diff before committing)
	$(PYTHON) scripts/generate_golden.py

.PHONY: golden-check
golden-check: ## Fail if golden output drifted
	$(PYTHON) scripts/generate_golden.py --check

.PHONY: check-llm
check-llm: ## Verify LLM_MODEL / VISION_MODEL exist on the provider
	$(PYTHON) scripts/check_llm.py

.PHONY: contract
contract: ## Export openapi.json + sample responses for backend handover
	$(PYTHON) scripts/export_openapi.py

.PHONY: evaluate
evaluate: ## Report extraction accuracy over the fixture dataset
	$(PYTHON) scripts/evaluate.py

.PHONY: check
check: lint typecheck test golden-check ## Everything CI runs

.PHONY: docker-build
docker-build: ## Build the production image
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run: ## Run the production image
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE)

.PHONY: docker-smoke
docker-smoke: docker-build ## Build the image and verify it serves /health
	docker run --rm -d --name receipt-ocr-smoke -p 8000:8000 $(IMAGE)
	@sleep 8
	@curl --fail --silent http://localhost:8000/health && echo "\nhealth ok"
	@curl --fail --silent http://localhost:8000/ready | head -c 400; echo
	docker stop receipt-ocr-smoke

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
