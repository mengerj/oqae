# 🚀 Paris 2025 - Cursor AI Development Showcase

A comprehensive Python development environment optimized for Cursor AI agents, featuring automated workflows, quality assurance, and best practices.

[![CI](https://github.com/mengerj/paris-2025/actions/workflows/ci.yml/badge.svg)](https://github.com/mengerj/paris-2025/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mengerj/paris-2025/branch/main/graph/badge.svg)](https://codecov.io/gh/mengerj/paris-2025)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- 🤖 **Cursor AI Optimized**: Tailored for AI-assisted development
- 🧪 **Test-Driven Development**: Comprehensive testing setup with pytest
- 🔄 **Automated Workflows**: GitHub Actions for CI/CD, formatting, and security
- 📦 **Modern Python**: Python 3.11+ with strict type checking
- 🏗️ **Clean Architecture**: Scalable structure with separation of concerns
- 📊 **Quality Assurance**: Linting, formatting, and security scanning
- 🚀 **Zero-Friction Setup**: One-command environment setup

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Git
- [GitHub CLI](https://cli.github.com/) (optional, for automated workflows)

### Setup

```bash
# Clone the repository
git clone https://github.com/mengerj/paris-2025.git
cd paris-2025

# Set up development environment
make setup-env
source venv/bin/activate  # On Windows: venv\Scripts\activate
make install-dev

# Verify setup
make ci
```

### First Steps

```bash
# View available commands
make help

# Run tests
make test

# Format code
make format

# Run full quality checks
make ci

# Create a new feature branch from an issue
make branch-from-issue
```

## 🏗️ Project Structure

```
paris-2025/
├── src/                    # Source code
│   ├── __init__.py
│   └── calculator.py       # Example module with clean architecture
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_calculator.py  # Comprehensive test examples
├── docs/                   # Documentation
│   └── DEVELOPMENT_WORKFLOW.md
├── .github/                # GitHub workflows and templates
│   ├── workflows/
│   │   ├── ci.yml         # Main CI pipeline
│   │   └── auto-format.yml # Automated formatting
│   ├── ISSUE_TEMPLATE/    # Issue templates
│   └── PULL_REQUEST_TEMPLATE.md
├── .vscode/               # Cursor/VS Code settings
│   ├── settings.json      # Optimized for AI development
│   └── launch.json        # Debug configurations
├── pyproject.toml         # Project configuration
├── Makefile              # Development commands
├── .pre-commit-config.yaml # Git hooks
└── requirements-dev.txt   # Development dependencies
```

## 🤖 Cursor AI Integration

This project is specifically designed to work seamlessly with Cursor AI:

### Key Features for AI Development

- 📝 **Comprehensive Type Hints**: All code includes type hints for better AI understanding
- 📚 **Detailed Docstrings**: Google-style docstrings with examples
- 🧪 **TDD-Ready**: Test structure optimized for AI-generated tests
- 🏗️ **Clean Architecture**: Clear separation of concerns for AI comprehension
- ⚙️ **Automated Workflows**: One-command operations for common tasks

### Cursor-Optimized Commands

```bash
# AI-friendly development workflow
make test-watch      # Continuous testing during development
make branch-from-issue  # Create branch from GitHub issue
make issue          # Create new issue interactively
make pr            # Create pull request with auto-generated description
```

### AI Prompting Examples

```python
# For Cursor AI: "Create a test for email validation with comprehensive edge cases"
def test_email_validation():
    validator = EmailValidator()
    assert validator.is_valid("user@example.com") is True
    assert validator.is_valid("invalid-email") is False

# For Cursor AI: "Implement EmailValidator following clean architecture principles"
```

## 🧪 Development Workflow

### Test-Driven Development

1. **Write Tests First** (Red)
   ```bash
   # Create test file
   touch tests/test_new_feature.py
   make test  # Should fail
   ```

2. **Implement Minimal Code** (Green)
   ```bash
   # Write minimal implementation
   make test  # Should pass
   ```

3. **Refactor** (Refactor)
   ```bash
   # Improve code quality
   make ci    # Ensure all checks pass
   ```

### Quality Assurance

All code is automatically checked for:

- **Formatting**: Black (88 char line length)
- **Import Sorting**: isort
- **Linting**: flake8
- **Type Checking**: mypy (strict mode)
- **Security**: bandit
- **Test Coverage**: pytest-cov (80%+ target)

### Git Workflow

```bash
# Create feature branch from issue
make branch-from-issue

# Make changes with TDD approach
make test-watch

# Quality check before commit
make ci

# Commit and push
git add .
git commit -m "feat: implement new feature"
git push origin feature/your-branch

# Create pull request
make pr
```

## 📋 Available Commands

### Development Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup-env` | Set up development environment |
| `make install-dev` | Install development dependencies |
| `make test` | Run tests with coverage |
| `make test-watch` | Run tests in watch mode |
| `make lint` | Run linting (flake8) |
| `make format` | Format code (black + isort) |
| `make type-check` | Run type checking (mypy) |
| `make ci` | Run full CI pipeline locally |
| `make clean` | Clean build artifacts |

### Workflow Commands

| Command | Description |
|---------|-------------|
| `make issue` | Create new GitHub issue |
| `make pr` | Create pull request |
| `make branch-from-issue` | Create branch from GitHub issue |
| `make workflow-status` | Check current workflow status |
| `make check-workflows` | Analyze workflow failures with suggestions |
| `make auto-fix` | Automatically fix workflow failures |
| `make auto-fix-push` | Auto-fix and push changes |

## 🔧 Configuration

### Tool Configuration

All tools are configured in `pyproject.toml`:

- **Black**: 88-character line length, Python 3.11 target
- **isort**: Black-compatible profile
- **mypy**: Strict type checking with comprehensive warnings
- **pytest**: Comprehensive coverage reporting
- **flake8**: Black-compatible linting

### Cursor/VS Code Settings

Optimized settings in `.vscode/settings.json`:

- Python interpreter path
- Automatic formatting on save
- Type checking enabled
- Test discovery configured
- File exclusions for cleaner workspace

## 🚀 GitHub Actions

### Continuous Integration (`ci.yml`)

- **Multi-Python Testing**: Python 3.11 and 3.12
- **Comprehensive Checks**: Linting, formatting, type checking, tests
- **Security Scanning**: Bandit and Safety checks
- **Coverage Reporting**: Codecov integration

### Auto-Formatting (`auto-format.yml`)

- **Automatic PRs**: Creates PRs for formatting fixes
- **Scheduled Runs**: Weekly maintenance
- **Zero Configuration**: Works out of the box

## 📊 Code Quality Metrics

- **Test Coverage**: 80%+ target with HTML reports
- **Type Coverage**: 100% for public APIs
- **Security**: Automated vulnerability scanning
- **Performance**: Benchmark tests for critical paths

## 🤝 Contributing

1. **Create an Issue**: Use provided templates
2. **Create Branch**: `make branch-from-issue`
3. **Develop with TDD**: Write tests first
4. **Quality Check**: `make ci`
5. **Create PR**: `make pr`

See [Development Workflow](docs/DEVELOPMENT_WORKFLOW.md) for detailed guidance.

## 📖 Documentation

- **[Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)**: Comprehensive guide for Cursor AI development
- **[Workflow Monitoring](docs/WORKFLOW_MONITORING.md)**: Auto-fix system for GitHub workflows
- **API Documentation**: Auto-generated from docstrings
- **Architecture Decisions**: Documented in code comments

## 🛠️ Troubleshooting

### Common Issues

**Tests not running:**
```bash
# Check Python path
echo $PYTHONPATH
# Reinstall dependencies
make clean && make install-dev
```

**Pre-commit hooks failing:**
```bash
pre-commit autoupdate
make pre-commit
```

**Type checking errors:**
```bash
# Install missing type stubs
pip install types-requests types-PyYAML
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Cursor AI** for enabling next-generation development workflows
- **Python Community** for excellent tooling and best practices
- **GitHub Actions** for seamless CI/CD integration

---

**Built with ❤️ for the Cursor AI community**
