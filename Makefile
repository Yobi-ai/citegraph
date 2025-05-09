.PHONY: setup clean data train test lint format help

# Variables
PYTHON := python
CONDA_ENV := citegraph

help:
	@echo "Available commands:"
	@echo "  make setup    - Create conda environment and install dependencies"
	@echo "  make clean    - Remove Python cache files and build artifacts"
	# @echo "  make data     - Download and prepare the dataset"
	# @echo "  make train    - Train the model"
	# @echo "  make test     - Run tests"
	@echo "  make lint     - Run code quality checks (ruff, isort, mypy)"
	@echo "  make format   - Format code using isort and ruff"

setup:
	conda create -n $(CONDA_ENV) python=3.11 -y
	conda activate $(CONDA_ENV) && pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# data:
# 	$(PYTHON) src/data/ingestion.py
# 	$(PYTHON) src/data/cleaning.py
# 	$(PYTHON) src/data/build_features.py

# train:
# 	$(PYTHON) src/models/model1/train.py

# test:
# 	pytest tests/

lint:
	ruff check .
	isort . --check-only
	mypy .

format:
	isort .
	ruff format .
