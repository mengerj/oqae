.PHONY: help install install-dev test lint format type-check clean setup-env

# Default target
help:
	@echo "Available commands:"
	@echo "  setup-env     - Set up development environment"
	@echo "  install       - Install production dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  test          - Run tests with coverage"
	@echo "  lint          - Run linting (flake8)"
	@echo "  format        - Format code (black + isort)"
	@echo "  type-check    - Run type checking (mypy)"
	@echo "  pre-commit    - Run pre-commit hooks"
	@echo "  clean         - Clean build artifacts"
	@echo "  ci            - Run full CI pipeline locally"

# Environment setup
setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"
	@echo "Then run: make install-dev"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]
	pre-commit install

# Testing
test:
	pytest -v --cov=src --cov-report=html --cov-report=term-missing

test-watch:
	pytest-watch -- -v --cov=src

# Code quality
lint:
	flake8 src tests

format:
	black src tests
	isort src tests

type-check:
	mypy src

pre-commit:
	pre-commit run --all-files

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# CI pipeline
ci: format lint type-check test
	@echo "✅ All checks passed!"

# Development workflow helpers
issue:
	@echo "Creating new issue..."
	@read -p "Issue title: " title; \
	read -p "Issue body: " body; \
	gh issue create --title "$$title" --body "$$body"

pr:
	@echo "Creating pull request..."
	@gh pr create --fill

# Auto-branch creation from issue
branch-from-issue:
	@read -p "Issue number: " issue_num; \
	issue_title=$$(gh issue view $$issue_num --json title --jq '.title'); \
	branch_name=$$(echo "$$issue_title" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$$//g'); \
	git checkout -b "feature/$$branch_name-$$issue_num"; \
	echo "Created branch: feature/$$branch_name-$$issue_num" 