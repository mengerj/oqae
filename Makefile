.PHONY: help install install-dev test lint format type-check clean setup-env

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "🏗️  Environment Setup (uv-based):"
	@echo "  setup-env     - Set up development environment with uv"
	@echo "  install       - Install production dependencies with uv"
	@echo "  install-dev   - Install development dependencies with uv"
	@echo "                  Then activate with: source .venv/bin/activate"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  test          - Run tests with coverage"
	@echo "  test-watch    - Run tests in watch mode"
	@echo "  lint          - Run linting (flake8)"
	@echo "  format        - Format code (autopep8 + black + isort) - AUTO-FIXES"
	@echo "  format-check  - Check code formatting (matches CI)"
	@echo "  type-check    - Run type checking (mypy strict)"
	@echo "  pre-commit    - Run pre-commit hooks"
	@echo "  ci            - Run full CI pipeline locally (matches GitHub)"
	@echo ""
	@echo "🔄 Workflow Management:"
	@echo "  issue              - Create new GitHub issue"
	@echo "  pr                 - Create pull request"
	@echo "  branch-from-issue  - Create branch from GitHub issue"
	@echo "  workflow-status    - Check current workflow status"
	@echo "  check-workflows    - Analyze workflow failures with suggestions"
	@echo "  auto-fix          - Automatically fix workflow failures"
	@echo "  auto-fix-push     - Auto-fix and push changes"
	@echo ""
	@echo "🚀 Template Setup (for new repositories):"
	@echo "  setup-template       - Interactive setup for new project"
	@echo "  setup-template-clean - Setup new project and remove examples"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean         - Clean build artifacts"

# Environment setup
setup-env:
	uv sync --all-extras
	@echo "Virtual environment created with uv. Activate with: source .venv/bin/activate"
	@echo "Pre-commit hooks installed automatically"

# Installation
install:
	uv sync

install-dev:
	uv sync --all-extras
	uv run pre-commit install

# Testing
test:
	uv run pytest -v --cov=src --cov-report=html --cov-report=term-missing

test-watch:
	uv run pytest-watch -- -v --cov=src

# Code quality
lint:
	@echo "🔍 Running flake8 linting..."
	uv run flake8 src tests

format:
	@echo "🔧 Auto-fixing code formatting and style issues..."
	uv run autopep8 --in-place --recursive --aggressive --aggressive --max-line-length=88 \
		--select=E1,E2,E3,E4,E5,E7,W1,W2,W3 src tests
	uv run black src tests
	uv run isort src tests
	@echo "✅ Code formatting and style fixes applied"

format-check:
	@echo "🔍 Checking code formatting..."
	uv run black --check src tests
	uv run isort --check-only src tests

type-check:
	@echo "🔍 Running mypy type checking..."
	uv run mypy src/omvqvae

pre-commit:
	uv run pre-commit run --all-files

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

# CI pipeline - matches GitHub workflow exactly
ci:
	@echo "🚀 Running CI pipeline (matches GitHub workflow)..."
	@echo ""
	@echo "📋 Step 1: Code formatting check..."
	@$(MAKE) format-check || (echo "❌ Format check failed" && exit 1)
	@echo ""
	@echo "📋 Step 2: Linting with flake8..."
	@$(MAKE) lint || (echo "❌ Linting failed" && exit 1)
	@echo ""
	@echo "📋 Step 3: Type checking..."
	@$(MAKE) type-check || (echo "❌ Type check failed" && exit 1)
	@echo ""
	@echo "📋 Step 4: Running tests..."
	@$(MAKE) test || (echo "❌ Tests failed" && exit 1)
	@echo ""
	@echo "✅ All CI checks passed! (matches GitHub CI)"

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

# Workflow monitoring and fixing
check-workflows:
	@uv run python scripts/check_workflows.py --suggest-fixes

check-workflows-json:
	@uv run python scripts/check_workflows.py --json --suggest-fixes

auto-fix:
	@uv run python scripts/auto_fix_workflow.py --branch $$(git branch --show-current) --commit

auto-fix-push:
	@uv run python scripts/auto_fix_workflow.py --branch $$(git branch --show-current) --commit --push

workflow-status:
	@echo "📊 Current Workflow Status:"
	@uv run python scripts/check_workflows.py
	@echo ""
	@echo "🔗 Recent workflow runs:"
	@gh run list --limit 5

# Template setup (for new repositories created from template)
setup-template:
	@echo "🚀 Setting up new project from template..."
	uv run python scripts/setup_template.py

setup-template-clean:
	@echo "🚀 Setting up new project from template (removing examples)..."
	uv run python scripts/setup_template.py --remove-examples
