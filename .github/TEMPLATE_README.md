# 🚀 Cursor AI Python Template

A comprehensive Python development template optimized for Cursor AI agents, featuring automated workflows, quality assurance, and best practices.

> **This is a GitHub template repository.** Click "Use this template" to create a new repository with this setup.

## ✨ What You Get

- 🤖 **Cursor AI Optimized** - Tailored for AI-assisted development
- 🧪 **Test-Driven Development** - Complete pytest setup with 80%+ coverage
- 🔄 **Automated Workflows** - GitHub Actions for CI/CD and auto-fixes
- 📦 **Modern Python** - Python 3.11+ with strict type checking
- 🏗️ **Clean Architecture** - Scalable structure with separation of concerns
- 📊 **Quality Assurance** - Black, isort, flake8, mypy, bandit
- 🚀 **Zero-Friction Setup** - One-command environment setup

## 🚀 Quick Start

### 1. Create Repository from Template

1. Click **"Use this template"** button above
2. Name your new repository
3. Clone your new repository:
   ```bash
   git clone https://github.com/yourusername/your-new-repo.git
   cd your-new-repo
   ```

### 2. Set Up Development Environment

```bash
# One-command setup
make setup-env
source venv/bin/activate  # On Windows: venv\Scripts\activate
make install-dev

# Verify everything works
make ci
```

### 3. Customize for Your Project

**📝 See [`TEMPLATE_SETUP.md`](TEMPLATE_SETUP.md) for detailed setup instructions**

## 🎯 Key Features for AI Development

### Cursor Integration
- **Type hints everywhere** for better AI understanding
- **Comprehensive docstrings** with examples
- **Clean architecture patterns** that AI can easily extend
- **Automated quality checks** to maintain code standards

### Example AI Prompts
```
"Write comprehensive tests for email validation with edge cases"
"Implement a clean architecture for user authentication"
"Refactor this code following SOLID principles"
```

### AI-Optimized Commands
```bash
make test-watch      # Continuous testing during development
make auto-fix        # Automatically fix workflow failures
make branch-from-issue  # Create branch from GitHub issue
make workflow-status    # Monitor CI status
```

## 📋 Template Structure

```
your-project/
├── src/                    # Your source code goes here
│   ├── __init__.py        # Update with your project info
│   └── calculator.py      # Example - replace with your code
├── tests/                  # Your tests go here
│   ├── __init__.py
│   └── test_calculator.py # Example - replace with your tests
├── .github/               # Pre-configured workflows
│   ├── workflows/
│   │   ├── ci.yml         # Main CI pipeline
│   │   ├── auto-format.yml # Auto-formatting
│   │   └── auto-fix.yml   # Auto-fix failures
│   └── ISSUE_TEMPLATE/    # Issue templates
├── docs/                  # Comprehensive documentation
├── scripts/               # Workflow monitoring tools
├── pyproject.toml         # Modern Python configuration
├── Makefile              # Development commands
└── TEMPLATE_SETUP.md     # Detailed setup guide
```

## 🛠️ Included Tools

### Code Quality
- **Black** - Code formatting (88 char line)
- **isort** - Import sorting
- **flake8** - Linting with Black compatibility
- **mypy** - Strict type checking
- **bandit** - Security scanning
- **pre-commit** - Git hooks

### Testing
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **80%+ coverage** requirement
- **HTML coverage reports**

### Automation
- **GitHub Actions** - CI/CD pipeline
- **Auto-fix workflows** - Automatic issue resolution
- **Workflow monitoring** - Real-time status checking
- **Automated PRs** - For formatting and fixes

## 🎯 Perfect For

- **AI-assisted development** with Cursor
- **Production-ready** Python projects
- **Team collaboration** with quality gates
- **Rapid prototyping** with professional standards
- **Learning** modern Python best practices

## 📚 Documentation

After creating your repository:

- **[TEMPLATE_SETUP.md](TEMPLATE_SETUP.md)** - Complete setup guide
- **[Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)** - Cursor AI workflows
- **[Workflow Monitoring](docs/WORKFLOW_MONITORING.md)** - Auto-fix system

## 🎉 Ready to Start?

1. **Use this template** to create your repository
2. **Follow the setup guide** in `TEMPLATE_SETUP.md`
3. **Start coding** with Cursor AI assistance!

---

**Built with ❤️ for the Cursor AI community**
