.PHONY: install test lint demo model figures
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check .
demo:
	python run_demo.py
model:
	python run_model.py
figures:
	python scripts/make_figures.py
