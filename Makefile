.PHONY: help install dev test eval lint fe-install fe-dev up down

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:        ## Install backend deps
	pip install -r requirements.txt -r requirements-dev.txt

dev:            ## Run backend (FastAPI) on :8000
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:           ## Run backend test suite
	python -m pytest -q

eval:           ## Run the retrieval eval harness
	python evals/run_eval.py --dataset evals/golden_qa.example.jsonl

fe-install:     ## Install frontend deps
	cd frontend && npm install

fe-dev:         ## Run frontend (Next.js) on :3000
	cd frontend && npm run dev

up:             ## Start full stack with Docker Compose
	docker compose up --build

down:           ## Stop Docker Compose
	docker compose down
