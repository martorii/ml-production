.PHONY: help install train test lint typecheck image serve up down logs traffic drift clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install dev dependencies and the package itself (editable)
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

train: ## Train the model and write artifacts/
	python -m tickets.train

lint: ## Ruff
	ruff check tickets tests scripts

typecheck: ## Mypy
	mypy tickets scripts

test: ## Run the full test suite (includes the model quality gate)
	pytest -v

image: train ## Train, then build the image around that exact model
	docker build -t ticket-router:local .

serve: ## Run the API locally with reload
	uvicorn tickets.api:app --reload --port 8000

# Trains first: the Dockerfile copies artifacts/ rather than training, so the compose
# build needs a model on disk to bake in.
up: train ## Start API + Prometheus + Grafana
	docker compose up -d --build
	@echo "API      http://localhost:8000/docs"
	@echo "Metrics  http://localhost:8000/metrics"
	@echo "Promethe http://localhost:9090"
	@echo "Grafana  http://localhost:3000 (admin/admin)"

down: ## Stop the stack
	docker compose down

logs: ## Tail the API logs
	docker compose logs -f api

traffic: ## Send normal traffic
	python scripts/simulate_traffic.py --n 400 --rps 25

drift: ## Send drifted traffic (new vocabulary, longer tickets, billing surge)
	python scripts/simulate_traffic.py --n 400 --rps 25 --shift 1.0

clean: ## Remove artifacts and caches
	rm -rf artifacts/*.joblib artifacts/*.json artifacts/*.csv .pytest_cache **/__pycache__
