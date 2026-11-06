.PHONY: install demo dashboard seed test lint typecheck eval check clean

install:
	uv sync

demo: seed dashboard

seed:
	uv run loan-triage seed

dashboard:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest --cov --cov-report=term-missing:skip-covered

lint:
	uv run ruff format --check src tests app scripts
	uv run ruff check src tests app scripts

typecheck:
	uv run mypy

eval:
	uv run loan-triage eval --seeds 3 --output reports/eval.json

check: lint typecheck test eval

clean:
	rm -rf data reports .coverage htmlcov
