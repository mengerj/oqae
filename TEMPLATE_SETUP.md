# 🚀 Template Setup Guide

Welcome to your new Cursor AI-optimized Python project! This template provides a complete development environment with automated workflows, quality assurance, and AI integration.

## ✅ Quick Start Checklist

After creating a new repository from this template, follow these steps:

### 1. 📝 **Update Project Information**

**Edit `pyproject.toml`:**
```toml
[project]
name = "your-project-name"  # ← Change this
version = "0.1.0"
description = "Your project description"  # ← Change this
authors = [
    {name = "Your Name", email = "your.email@example.com"}  # ← Change this
]

[project.urls]
Homepage = "https://github.com/yourusername/your-repo"  # ← Change this
Repository = "https://github.com/yourusername/your-repo.git"  # ← Change this
Issues = "https://github.com/yourusername/your-repo/issues"  # ← Change this
```

**Edit `README.md`:**
- Update project title and description
- Replace repository URLs
- Update badges with your repository path
- Customize the features list for your project

**Edit `src/__init__.py`:**
```python
"""Your Project Name - Brief Description."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
```

### 2. 🏗️ **Set Up Development Environment**

```bash
# Clone your new repository
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Set up development environment
make setup-env
source venv/bin/activate  # On Windows: venv\Scripts\activate
make install-dev

# Verify setup works
make ci
```

### 3. 🧪 **Replace Example Code**

**Replace the example calculator:**
1. Delete `src/calculator.py` (example module)
2. Delete `tests/test_calculator.py` (example tests)
3. Create your own modules in `src/`
4. Create corresponding tests in `tests/`

**Keep the project structure:**
```
your-repo/
├── src/
│   ├── __init__.py          # ← Update this
│   └── your_modules.py      # ← Add your code here
├── tests/
│   ├── __init__.py          # ← Keep this
│   └── test_your_modules.py # ← Add your tests here
```

### 4. 🔧 **Configure GitHub Repository**

**Enable GitHub Actions:**
- Go to your repository Settings → Actions → General
- Allow all actions and reusable workflows

**Set up branch protection (optional but recommended):**
- Go to Settings → Branches
- Add rule for `main` branch
- Require status checks to pass before merging
- Require pull request reviews

### 5. 🤖 **Cursor AI Integration**

**Open in Cursor:**
```bash
# If you have Cursor installed
cursor .
```

**Start developing with AI assistance:**
```bash
# Create new feature branch
make branch-from-issue

# Develop with TDD
make test-watch  # Keep this running while developing

# Use Cursor to:
# - "Write tests for [functionality]"
# - "Implement [feature] following clean architecture"
# - "Refactor this code for better readability"
```

### 6. 📊 **Verify Everything Works**

```bash
# Run full CI pipeline
make ci

# Check workflow monitoring
make workflow-status

# Create a test issue and PR
make issue
make pr
```

## 🎯 **Available Commands**

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup-env` | Set up development environment |
| `make install-dev` | Install development dependencies |
| `make test` | Run tests with coverage |
| `make ci` | Run full CI pipeline locally |
| `make workflow-status` | Check GitHub workflow status |
| `make auto-fix` | Automatically fix workflow failures |
| `make branch-from-issue` | Create branch from GitHub issue |
| `make pr` | Create pull request |

## 🔄 **Development Workflow**

1. **Create Issue**: Use GitHub templates or `make issue`
2. **Create Branch**: `make branch-from-issue`
3. **Develop with TDD**: `make test-watch`
4. **Quality Check**: `make ci`
5. **Create PR**: `make pr`
6. **Auto-fix failures**: `make auto-fix` (if CI fails)

## 📚 **Template Features**

✅ **Complete Python Setup**
- Python 3.11+ with virtual environment
- Modern package configuration (pyproject.toml)
- Development and production dependencies

✅ **Code Quality Assurance**
- Black formatting (88 char line length)
- isort import sorting
- flake8 linting
- mypy type checking (strict mode)
- bandit security scanning
- pre-commit hooks

✅ **Testing Framework**
- pytest with comprehensive configuration
- Test coverage reporting (80%+ target)
- HTML coverage reports
- Example test structure

✅ **GitHub Actions CI/CD**
- Multi-Python testing (3.11, 3.12)
- Automated quality checks
- Security scanning
- Auto-formatting workflow
- Auto-fix capability

✅ **Cursor AI Optimization**
- Type hints everywhere
- Comprehensive docstrings
- Clean architecture examples
- AI-friendly project structure

✅ **Developer Experience**
- One-command setup
- Automated workflows
- Real-time monitoring
- Comprehensive documentation

## 🛠️ **Customization**

### Adding New Dependencies

**For production:**
```bash
# Edit pyproject.toml
[project]
dependencies = [
    "your-package>=1.0.0",
]
```

**For development:**
```bash
# Edit pyproject.toml
[project.optional-dependencies]
dev = [
    # ... existing packages
    "your-dev-package>=1.0.0",
]
```

### Adding New Quality Tools

**Edit `.pre-commit-config.yaml`:**
```yaml
repos:
  # ... existing repos
  - repo: https://github.com/your-tool/repo
    rev: v1.0.0
    hooks:
      - id: your-tool
```

**Edit `.github/workflows/ci.yml`:**
```yaml
- name: Run your tool
  run: |
    your-tool src tests
```

### Custom Makefile Commands

**Add to `Makefile`:**
```makefile
your-command:
	@echo "Running your custom command"
	your-tool --options
```

## 🚨 **Troubleshooting**

### Common Issues

**Virtual environment issues:**
```bash
rm -rf venv
make setup-env
source venv/bin/activate
make install-dev
```

**GitHub CLI not working:**
```bash
gh auth login
gh auth status
```

**Pre-commit hooks failing:**
```bash
pre-commit autoupdate
pre-commit install
make pre-commit
```

### Getting Help

1. Check the comprehensive documentation in `docs/`
2. Review GitHub issues and workflows
3. Use Cursor AI for code-specific help
4. Create issues using the provided templates

## 🎉 **You're Ready!**

Your Cursor AI-optimized Python project is ready for development. The template provides:

- **Zero-friction setup** with one command
- **Automated quality assurance**
- **AI-optimized development** workflows
- **Professional CI/CD** pipeline
- **Comprehensive documentation**

Happy coding with Cursor AI! 🚀
