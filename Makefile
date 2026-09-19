.PHONY: help install seed verify export test lint format \
        api migrate frontend frontend-install frontend-test \
        docker docker-up docker-down clean

help:
	@echo "hr-talent-evaluator -- scraped candidate records in, hiring recommendations out"
	@echo ""
	@echo "  Quick start"
	@echo "  make install          Install backend dependencies"
	@echo "  make seed             Load and rank the 9-candidate demo pool"
	@echo "  make api              Run the API on :8000"
	@echo "  make frontend         Run the dashboard on :5173"
	@echo ""
	@echo "  Checks"
	@echo "  make verify           Configuration + PII scrubber report"
	@echo "  make test             Backend test suite (warnings are errors)"
	@echo "  make lint             Ruff lint + format check"
	@echo "  make format           Ruff autofix + format"
	@echo "  make frontend-test    Frontend unit tests"
	@echo ""
	@echo "  Other"
	@echo "  make export           Print the demo pool as import-ready JSON"
	@echo "  make migrate          Apply Alembic migrations"
	@echo "  make docker-up        Build + start the full stack"

install:
	pip install -r requirements.txt

seed:
	python -m app.main seed

verify:
	python -m app.main verify

export:
	python -m app.main export

test:
	pytest -q

lint:
	ruff check app tests
	ruff format --check app tests

format:
	ruff check --fix app tests
	ruff format app tests

migrate:
	alembic upgrade head

api: migrate
	uvicorn app.api.app:app --reload --host 0.0.0.0 --port 8000

frontend-install:
	cd frontend && npm install

frontend:
	cd frontend && npm run dev

frontend-test:
	cd frontend && npm run test

docker:
	docker build -f Dockerfile.api -t hr-talent-evaluator-api:2.0.0 .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache *.egg-info data
