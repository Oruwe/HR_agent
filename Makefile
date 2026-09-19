.PHONY: help install demo verify bench test lint docker clean \
        api api-install migrate frontend frontend-install frontend-test \
        docker-up docker-down

help:
	@echo "hr-talent-evaluator"
	@echo ""
	@echo "  Offline agent / worker"
	@echo "  make install          Install core dependencies (no services required)"
	@echo "  make demo             Run a full offline screening interview"
	@echo "  make verify           OpenGAP / latency budget / security compliance report"
	@echo "  make bench            Reproduce the latency benchmark"
	@echo "  make test             Full backend test suite (warnings are errors)"
	@echo "  make lint             Ruff lint"
	@echo "  make docker           Build the production worker/demo image"
	@echo ""
	@echo "  API Gateway"
	@echo "  make api-install      Install full API + DB + STT dependencies"
	@echo "  make migrate          Apply Alembic migrations (creates ./data/app.db by default)"
	@echo "  make api              Run the API Gateway locally on :8000"
	@echo ""
	@echo "  Frontend (Candidate App + Admin Dashboard)"
	@echo "  make frontend-install Install frontend dependencies"
	@echo "  make frontend         Run the frontend dev server on :5173"
	@echo "  make frontend-test    Run frontend unit tests"
	@echo ""
	@echo "  Full stack via Docker Compose"
	@echo "  make docker-up        Build + start API and frontend (docker compose up --build)"
	@echo "  make docker-down      Stop the stack"

install:
	pip install "pydantic>=2.6" "PyYAML>=6.0" "numpy>=1.26" "pytest>=8" ruff

demo:
	python -m app.main demo

verify:
	python -m app.main verify

bench:
	python scripts/benchmark.py --iterations 60

test:
	pytest -q

lint:
	ruff check app tests scripts

docker:
	docker build -t hr-talent-evaluator:1.0.0 .

api-install:
	pip install -r requirements.txt

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

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache *.egg-info
