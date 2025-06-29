#!/usr/bin/env python3
"""
Setup script for new repositories created from the Cursor AI Python template.

This script helps users quickly customize their project by:
1. Updating project information in pyproject.toml
2. Updating __init__.py with project details
3. Optionally removing example code
4. Creating initial git commit with changes
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def get_user_input() -> Dict[str, str]:
    """Collect project information from user."""
    print("🚀 Setting up your new Cursor AI Python project!")
    print("Please provide the following information:\n")

    project_info = {}

    # Get project name
    while True:
        name = input("Project name (e.g., 'my-awesome-app'): ").strip()
        if name and re.match(r'^[a-z0-9\-_]+$', name):
            project_info['name'] = name
            break
        print("❌ Please use lowercase letters, numbers, hyphens, and underscores only")

    # Get project description
    project_info['description'] = input("Project description: ").strip()

    # Get author info
    project_info['author_name'] = input("Your name: ").strip()
    project_info['author_email'] = input("Your email: ").strip()

    # Get repository info
    default_username = subprocess.run(
        ['git', 'config', 'user.name'],
        capture_output=True, text=True
    ).stdout.strip().replace(' ', '').lower()

    repo_url = input(f"Repository URL (e.g., 'https://github.com/{default_username}/{name}'): ").strip()
    if not repo_url:
        repo_url = f"https://github.com/{default_username}/{project_info['name']}"

    project_info['repository'] = repo_url
    project_info['homepage'] = repo_url
    project_info['issues'] = f"{repo_url}/issues"

    return project_info


def update_pyproject_toml(project_info: Dict[str, str]) -> None:
    """Update pyproject.toml with project information."""
    print("\n📝 Updating pyproject.toml...")

    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Update project fields
    content = re.sub(r'name = ".*?"', f'name = "{project_info["name"]}"', content)
    content = re.sub(r'description = ".*?"', f'description = "{project_info["description"]}"', content)
    content = re.sub(
        r'authors = \[\s*{name = ".*?", email = ".*?"}\s*\]',
        f'authors = [{{"name": "{project_info["author_name"]}", "email": "{project_info["author_email"]}"}}]',
        content,
        flags=re.DOTALL
    )

    # Update URLs
    content = re.sub(r'Homepage = ".*?"', f'Homepage = "{project_info["homepage"]}"', content)
    content = re.sub(r'Repository = ".*?"', f'Repository = "{project_info["repository"]}.git"', content)
    content = re.sub(r'Issues = ".*?"', f'Issues = "{project_info["issues"]}"', content)

    pyproject_path.write_text(content)
    print("✅ Updated pyproject.toml")


def update_init_py(project_info: Dict[str, str]) -> None:
    """Update src/__init__.py with project information."""
    print("📝 Updating src/__init__.py...")

    init_path = Path("src/__init__.py")
    content = f'''"""{project_info["description"]}"""

__version__ = "0.1.0"
__author__ = "{project_info["author_name"]}"
__email__ = "{project_info["author_email"]}"
'''

    init_path.write_text(content)
    print("✅ Updated src/__init__.py")


def remove_example_code() -> None:
    """Remove example calculator code."""
    print("🗑️  Removing example code...")

    calculator_py = Path("src/calculator.py")
    test_calculator_py = Path("tests/test_calculator.py")

    if calculator_py.exists():
        calculator_py.unlink()
        print("✅ Removed src/calculator.py")

    if test_calculator_py.exists():
        test_calculator_py.unlink()
        print("✅ Removed tests/test_calculator.py")


def create_initial_module(project_info: Dict[str, str]) -> None:
    """Create initial module and test files."""
    print("📄 Creating initial module...")

    # Create main module
    module_name = project_info['name'].replace('-', '_')
    module_path = Path(f"src/{module_name}.py")

    module_content = f'''"""
{project_info["description"]}

This is the main module for {project_info["name"]}.
"""

from typing import Any


def hello_world() -> str:
    """Return a friendly greeting.

    Returns:
        str: A greeting message

    Example:
        >>> hello_world()
        'Hello from {project_info["name"]}!'
    """
    return f"Hello from {project_info['name']}!"


class {module_name.title().replace('_', '')}:
    """Main class for {project_info["name"]}."""

    def __init__(self, name: str = "World") -> None:
        """Initialize the class.

        Args:
            name: Name to greet
        """
        self.name = name

    def greet(self) -> str:
        """Return a personalized greeting.

        Returns:
            str: Personalized greeting message
        """
        return f"Hello {{self.name}} from {project_info['name']}!"
'''

    module_path.write_text(module_content)
    print(f"✅ Created src/{module_name}.py")

    # Create test file
    test_path = Path(f"tests/test_{module_name}.py")
    test_content = f'''"""
Tests for {module_name} module.
"""

import pytest
from src.{module_name} import hello_world, {module_name.title().replace('_', '')}


def test_hello_world():
    """Test hello_world function."""
    result = hello_world()
    assert result == "Hello from {project_info['name']}!"
    assert isinstance(result, str)


class Test{module_name.title().replace('_', '')}:
    """Test cases for {module_name.title().replace('_', '')} class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        obj = {module_name.title().replace('_', '')}()
        assert obj.name == "World"

    def test_init_custom(self):
        """Test initialization with custom name."""
        obj = {module_name.title().replace('_', '')}("Alice")
        assert obj.name == "Alice"

    def test_greet_default(self):
        """Test greet method with default name."""
        obj = {module_name.title().replace('_', '')}()
        result = obj.greet()
        assert result == "Hello World from {project_info['name']}!"

    def test_greet_custom(self):
        """Test greet method with custom name."""
        obj = {module_name.title().replace('_', '')}("Alice")
        result = obj.greet()
        assert result == "Hello Alice from {project_info['name']}!"

    @pytest.mark.parametrize("name,expected", [
        ("Alice", "Hello Alice from {project_info['name']}!"),
        ("Bob", "Hello Bob from {project_info['name']}!"),
        ("", "Hello  from {project_info['name']}!"),
    ])
    def test_greet_parametrized(self, name: str, expected: str):
        """Test greet method with various names."""
        obj = {module_name.title().replace('_', '')}(name)
        assert obj.greet() == expected
'''

    test_path.write_text(test_content)
    print(f"✅ Created tests/test_{module_name}.py")


def run_initial_setup() -> None:
    """Run initial setup commands."""
    print("\n🔧 Running initial setup...")

    try:
        # Install dependencies
        subprocess.run(['make', 'install-dev'], check=True, capture_output=True)
        print("✅ Installed development dependencies")

        # Run tests to make sure everything works
        subprocess.run(['make', 'test'], check=True, capture_output=True)
        print("✅ Tests passed")

        # Run formatting
        subprocess.run(['make', 'format'], check=True, capture_output=True)
        print("✅ Code formatted")

    except subprocess.CalledProcessError as e:
        print(f"❌ Setup command failed: {e}")
        print("You may need to run 'make setup-env' and activate the virtual environment first")


def commit_changes(project_info: Dict[str, str]) -> None:
    """Commit the initial changes."""
    print("\n📦 Committing changes...")

    try:
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run([
            'git', 'commit', '-m',
            f"feat: Initialize {project_info['name']} from template\n\n- Update project information\n- Remove example code\n- Add initial module structure"
        ], check=True)
        print("✅ Changes committed")

        print("\n🎉 Setup complete! Your project is ready for development.")
        print(f"\nNext steps:")
        print(f"1. Push to your repository: git push origin main")
        print(f"2. Open in Cursor: cursor .")
        print(f"3. Start developing: make test-watch")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git commit failed: {e}")
        print("You can commit the changes manually later")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup new project from Cursor AI Python template")
    parser.add_argument('--remove-examples', action='store_true',
                       help='Remove example calculator code')
    args = parser.parse_args()

    # Check if we're in a git repository
    if not Path('.git').exists():
        print("❌ This doesn't appear to be a git repository")
        print("Make sure you're in the root of your project directory")
        sys.exit(1)

    # Check for required files
    required_files = ['pyproject.toml', 'src/__init__.py']
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file {file} not found")
            print("Make sure you're using the Cursor AI Python template")
            sys.exit(1)

    # Get user input
    project_info = get_user_input()

    # Update files
    update_pyproject_toml(project_info)
    update_init_py(project_info)

    # Handle example code
    if args.remove_examples or input("\n🗑️  Remove example calculator code? (y/N): ").lower().startswith('y'):
        remove_example_code()
        create_initial_module(project_info)

    # Run setup
    if input("\n🔧 Run initial setup (install deps, run tests)? (Y/n): ").lower() not in ['n', 'no']:
        run_initial_setup()

    # Commit changes
    if input("\n📦 Commit changes to git? (Y/n): ").lower() not in ['n', 'no']:
        commit_changes(project_info)

    print(f"\n🚀 {project_info['name']} is ready! Happy coding with Cursor AI!")


if __name__ == "__main__":
    main()
