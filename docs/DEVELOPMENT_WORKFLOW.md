# Development Workflow with Cursor AI

This guide outlines the optimal development workflow for this project using Cursor AI agents, focusing on automation, quality, and efficiency.

## 🚀 Quick Start

### Initial Setup
```bash
# Clone and navigate to the repository
git clone https://github.com/mengerj/paris-2025.git
cd paris-2025

# Set up the development environment
make setup-env
source venv/bin/activate
make install-dev

# Verify setup
make ci
```

### Daily Workflow
```bash
# Start working on a new feature
make branch-from-issue  # Creates branch from GitHub issue
# Follow the prompts to select an issue

# Develop with TDD
make test-watch  # Run tests in watch mode
# Write tests first, then implement features

# Check quality before committing
make ci  # Runs full CI pipeline locally

# Create pull request
make pr
```

## 🤖 Cursor AI Integration

### Optimal Cursor Usage Patterns

#### 1. Test-Driven Development with AI
```python
# Start by describing what you want to test to Cursor:
# "Write a test for a function that validates email addresses"

def test_email_validation():
    validator = EmailValidator()
    assert validator.is_valid("user@example.com") is True
    assert validator.is_valid("invalid-email") is False

# Then ask Cursor to implement the validator
# "Implement the EmailValidator class to make this test pass"
```

#### 2. Cursor Commands for Common Tasks

Create custom Cursor commands in `.vscode/settings.json`:

```json
{
    "cursor.commandPalette.commands": [
        {
            "command": "Create Test File",
            "description": "Generate test file with TDD structure",
            "action": "createTestFile"
        },
        {
            "command": "Refactor with Clean Architecture",
            "description": "Refactor code following clean architecture principles",
            "action": "refactorCleanArchitecture"
        }
    ]
}
```

#### 3. AI Prompting Best Practices

**For Architecture Decisions:**
```
"Design a clean architecture for [feature] that:
- Separates concerns clearly
- Is testable and mockable
- Follows SOLID principles
- Has clear interfaces
- Is scalable for future requirements"
```

**For Test Creation:**
```
"Create comprehensive tests for [module] that cover:
- Happy path scenarios
- Edge cases
- Error conditions
- Integration points
- Performance requirements (if applicable)"
```

**For Refactoring:**
```
"Refactor this code to:
- Improve readability and maintainability
- Extract reusable components
- Add proper error handling
- Ensure type safety
- Follow project conventions"
```

## 🔄 Automated Workflows

### Issue-to-Branch Workflow

1. **Create Issue**: Use GitHub templates or `make issue`
2. **Auto-Branch**: Run `make branch-from-issue` to create feature branch
3. **Develop**: Write tests first, then implement features
4. **Quality Check**: `make ci` ensures code quality
5. **Pull Request**: `make pr` creates PR with auto-generated description

### Branch Protection & CI

- **Main branch** is protected
- All PRs require:
  - Passing CI checks
  - Code review approval
  - Up-to-date with main branch
  - No merge conflicts

### Automated Code Quality

- **Pre-commit hooks** run automatically on every commit
- **GitHub Actions** run full CI on every PR
- **Auto-formatting** creates PRs for code style fixes
- **Security scanning** runs on every push

## 🧪 Test-Driven Development

### TDD Cycle with Cursor

1. **Red**: Write a failing test
   ```bash
   # Ask Cursor: "Write a test for [functionality]"
   make test  # Should fail
   ```

2. **Green**: Make the test pass
   ```bash
   # Ask Cursor: "Implement the minimal code to make this test pass"
   make test  # Should pass
   ```

3. **Refactor**: Improve the code
   ```bash
   # Ask Cursor: "Refactor this code while keeping tests green"
   make test  # Should still pass
   ```

### Test Structure

```python
# tests/test_feature.py
"""
Test module for feature functionality.

This module demonstrates:
- Clear test organization
- Comprehensive coverage
- Descriptive test names
- Proper use of fixtures
"""

import pytest
from src.feature import Feature


class TestFeature:
    """Test suite for Feature class."""

    @pytest.fixture
    def feature(self) -> Feature:
        """Fixture providing a Feature instance."""
        return Feature()

    def test_feature_does_something_when_condition_met(self, feature: Feature) -> None:
        """Test that feature behaves correctly under normal conditions."""
        # Given (Arrange)
        input_data = "test_input"
        expected_output = "expected_result"

        # When (Act)
        result = feature.process(input_data)

        # Then (Assert)
        assert result == expected_output
```

## 🏗️ Clean Architecture Principles

### Project Structure
```
src/
├── domain/          # Business logic and entities
├── application/     # Use cases and application services
├── infrastructure/  # External dependencies
├── interfaces/      # Adapters and controllers
└── shared/         # Common utilities
```

### Dependency Rules
- **Inner layers** don't depend on outer layers
- **Interfaces** are defined in inner layers
- **Implementations** are in outer layers
- **Use dependency injection** for flexibility

### Example Clean Architecture Implementation

```python
# domain/entities.py
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class User:
    """Domain entity representing a user."""
    id: str
    name: str
    email: str

# domain/repositories.py
class UserRepository(ABC):
    """Abstract repository for user operations."""

    @abstractmethod
    def save(self, user: User) -> None:
        pass

    @abstractmethod
    def find_by_id(self, user_id: str) -> User | None:
        pass

# application/use_cases.py
class CreateUserUseCase:
    """Use case for creating a new user."""

    def __init__(self, user_repo: UserRepository):
        self._user_repo = user_repo

    def execute(self, name: str, email: str) -> User:
        user = User(id=generate_id(), name=name, email=email)
        self._user_repo.save(user)
        return user
```

## 📊 Quality Metrics

### Code Coverage
- Target: **80%+ coverage**
- Run: `make test` generates HTML coverage report
- View: Open `htmlcov/index.html`

### Code Quality Tools
- **Black**: Code formatting (88 char line length)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking (strict mode)
- **bandit**: Security scanning

### Performance Monitoring
```python
# Use pytest-benchmark for performance tests
def test_performance_critical_function(benchmark):
    result = benchmark(critical_function, large_input)
    assert result.is_valid()
```

## 🔧 Development Commands

### Essential Commands
```bash
make help           # Show all available commands
make setup-env      # Set up development environment
make install-dev    # Install development dependencies
make test          # Run tests with coverage
make test-watch    # Run tests in watch mode
make lint          # Run linting
make format        # Format code
make type-check    # Run type checking
make ci            # Run full CI pipeline
make clean         # Clean build artifacts
```

### Workflow Commands
```bash
make issue              # Create new GitHub issue
make pr                # Create pull request
make branch-from-issue  # Create branch from issue
```

### Advanced Usage
```bash
# Run specific test file
pytest tests/test_calculator.py -v

# Run tests with specific marker
pytest -m "slow" -v

# Generate coverage report
pytest --cov=src --cov-report=html

# Run type checking on specific file
mypy src/calculator.py

# Format specific file
black src/calculator.py
```

## 🎯 Best Practices

### Code Style
- **Line length**: 88 characters (Black default)
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all modules, classes, and functions
- **Variable names**: Descriptive and unambiguous

### Git Workflow
- **Commit messages**: Use conventional commits format
- **Branch naming**: `feature/description-issue-number`
- **Small commits**: Atomic changes with clear purposes
- **No direct commits** to main branch

### Error Handling
```python
# Create specific exception types
class DomainError(Exception):
    """Base exception for domain-specific errors."""
    pass

class ValidationError(DomainError):
    """Raised when validation fails."""
    pass

# Use proper error handling
def validate_email(email: str) -> str:
    if "@" not in email:
        raise ValidationError(f"Invalid email format: {email}")
    return email.lower()
```

### Documentation
- **README**: Project overview and quick start
- **Docstrings**: All public APIs documented
- **Type hints**: Self-documenting code
- **Examples**: Include usage examples in docstrings

## 🚨 Troubleshooting

### Common Issues

**Tests failing after setup:**
```bash
# Check Python path
echo $PYTHONPATH
# Should include /path/to/project/src

# Reinstall dependencies
make clean
make install-dev
```

**Pre-commit hooks failing:**
```bash
# Update pre-commit
pre-commit autoupdate
pre-commit install

# Run all hooks
make pre-commit
```

**Type checking errors:**
```bash
# Install type stubs
pip install types-requests types-PyYAML

# Check specific file
mypy src/problematic_file.py --show-error-codes
```

### Getting Help

1. **Check this documentation** first
2. **Review existing issues** on GitHub
3. **Ask Cursor AI** for code-specific help
4. **Create an issue** using the provided templates

## 🔄 Continuous Improvement

### Regular Tasks
- **Weekly**: Review and update dependencies
- **Monthly**: Analyze test coverage and quality metrics
- **Quarterly**: Review and update development practices

### Metrics to Track
- Test coverage percentage
- Code quality score (flake8, mypy)
- Build/test execution time
- Number of bugs found in production

This workflow is designed to maximize productivity while maintaining high code quality through automation and AI assistance.
