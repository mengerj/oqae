#!/usr/bin/env python3
"""
Automatic Workflow Fix Script

This script can automatically fix common workflow failures and
create commits with the fixes.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from check_workflows import WorkflowChecker


class AutoFixer:
    """Automatically fix common workflow issues."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.fixes_applied = []

    def run_command(
        self, cmd: list[str], cwd: Path = None
    ) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        if self.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=cwd
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            raise

    def fix_formatting(self) -> bool:
        """Fix code formatting issues."""
        try:
            print("🔧 Fixing code formatting...")

            # Run black
            self.run_command(["black", "src", "tests"])

            # Run isort
            self.run_command(["isort", "src", "tests"])

            # Check if there are changes
            result = self.run_command(["git", "diff", "--name-only"])
            if result.stdout.strip():
                self.fixes_applied.append("formatting")
                print("✅ Code formatting fixed")
                return True
            else:
                print("ℹ️  No formatting changes needed")
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to fix formatting: {e}")
            return False

    def fix_linting(self) -> bool:
        """Attempt to fix common linting issues."""
        try:
            print("🔧 Checking linting issues...")

            # Run flake8 to see current issues
            try:
                result = self.run_command(["flake8", "src", "tests"])
                print("ℹ️  No linting issues found")
                return False
            except subprocess.CalledProcessError as e:
                print("📋 Linting issues found:")
                print(e.stdout)

                # Some common fixes we can apply
                fixes_made = False

                # Fix unused imports (basic attempt)
                if "imported but unused" in e.stdout:
                    print("🔧 Attempting to fix unused imports...")
                    try:
                        self.run_command(
                            [
                                "autoflake",
                                "--remove-all-unused-imports",
                                "--in-place",
                                "-r",
                                "src",
                                "tests",
                            ]
                        )
                        fixes_made = True
                    except subprocess.CalledProcessError:
                        print(
                            "⚠️  autoflake not available, skipping unused import fixes"
                        )

                if fixes_made:
                    self.fixes_applied.append("linting")
                    print("✅ Some linting issues fixed")
                    return True
                else:
                    print("⚠️  Manual intervention required for linting issues")
                    return False

        except Exception as e:
            print(f"❌ Error checking linting: {e}")
            return False

    def fix_import_sorting(self) -> bool:
        """Fix import sorting issues."""
        try:
            print("🔧 Fixing import sorting...")

            self.run_command(["isort", "src", "tests", "--check-only"])
            print("ℹ️  Import sorting is correct")
            return False

        except subprocess.CalledProcessError:
            # isort check failed, so we need to fix it
            self.run_command(["isort", "src", "tests"])
            self.fixes_applied.append("import_sorting")
            print("✅ Import sorting fixed")
            return True

    def add_missing_type_hints(self) -> bool:
        """Add basic type hints to functions missing them."""
        try:
            print("🔧 Checking for missing type hints...")

            # Run mypy to see what's missing
            try:
                self.run_command(["mypy", "src"])
                print("ℹ️  Type checking passed")
                return False
            except subprocess.CalledProcessError as e:
                print("📋 Type checking issues found:")
                print(e.stdout)

                # This is complex to fix automatically, so we'll just report
                print("⚠️  Manual intervention required for type hints")
                return False

        except Exception as e:
            print(f"❌ Error checking types: {e}")
            return False

    def fix_test_issues(self) -> bool:
        """Try to fix basic test issues."""
        try:
            print("🔧 Checking test issues...")

            # Run tests to see what fails
            try:
                self.run_command(["pytest", "tests", "-v"])
                print("ℹ️  All tests passing")
                return False
            except subprocess.CalledProcessError as e:
                print("📋 Test failures found:")
                print(e.stdout)

                # Basic fixes we can attempt
                fixes_made = False

                # Check if it's just missing __init__.py files
                if "No module named" in e.stdout:
                    print("🔧 Ensuring __init__.py files exist...")

                    # Make sure all test directories have __init__.py
                    test_dirs = Path("tests").rglob("*/")
                    for test_dir in test_dirs:
                        if test_dir.is_dir():
                            init_file = test_dir / "__init__.py"
                            if not init_file.exists():
                                if not self.dry_run:
                                    init_file.touch()
                                print(f"Created {init_file}")
                                fixes_made = True

                if fixes_made:
                    self.fixes_applied.append("test_structure")
                    print("✅ Test structure issues fixed")
                    return True
                else:
                    print("⚠️  Manual intervention required for test failures")
                    return False

        except Exception as e:
            print(f"❌ Error checking tests: {e}")
            return False

    def commit_fixes(self, message: str = None) -> bool:
        """Commit the applied fixes."""
        if not self.fixes_applied:
            print("ℹ️  No fixes to commit")
            return False

        try:
            # Check if there are changes to commit
            result = self.run_command(["git", "diff", "--name-only"])
            if not result.stdout.strip():
                print("ℹ️  No changes to commit")
                return False

            # Add all changes
            self.run_command(["git", "add", "."])

            # Create commit message
            if not message:
                fixes_str = ", ".join(self.fixes_applied)
                message = f"fix: automatic workflow fixes ({fixes_str})"

            # Commit
            self.run_command(["git", "commit", "-m", message])

            print(f"✅ Committed fixes: {', '.join(self.fixes_applied)}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to commit fixes: {e}")
            return False

    def auto_fix_workflow_failures(self, branch: str = None) -> dict[str, Any]:
        """Automatically fix workflow failures for a branch."""
        print("🔍 Analyzing workflow failures...")

        checker = WorkflowChecker()
        analysis = checker.analyze_failures(branch)

        if analysis.get("status") == "no_runs":
            return {"status": "no_runs", "message": "No workflow runs found"}

        if analysis.get("conclusion") != "failure":
            return {"status": "no_failures", "message": "No failures to fix"}

        print(f"📋 Workflow failed for branch: {analysis['branch']}")

        # Apply fixes based on failed steps
        fixes_attempted = []

        for step in analysis.get("failed_steps", []):
            step_name = step.get("step_name", "").lower()

            if "black" in step_name or "format" in step_name:
                if "formatting" not in fixes_attempted:
                    self.fix_formatting()
                    fixes_attempted.append("formatting")

            elif "isort" in step_name or "import" in step_name:
                if "import_sorting" not in fixes_attempted:
                    self.fix_import_sorting()
                    fixes_attempted.append("import_sorting")

            elif "flake8" in step_name or "lint" in step_name:
                if "linting" not in fixes_attempted:
                    self.fix_linting()
                    fixes_attempted.append("linting")

            elif "mypy" in step_name or "type" in step_name:
                if "type_hints" not in fixes_attempted:
                    self.add_missing_type_hints()
                    fixes_attempted.append("type_hints")

            elif "test" in step_name or "pytest" in step_name:
                if "tests" not in fixes_attempted:
                    self.fix_test_issues()
                    fixes_attempted.append("tests")

        # Commit fixes if any were applied
        if self.fixes_applied:
            committed = self.commit_fixes()
            return {
                "status": "fixes_applied",
                "fixes": self.fixes_applied,
                "committed": committed,
                "message": f"Applied fixes: {', '.join(self.fixes_applied)}",
            }
        else:
            return {
                "status": "no_fixes_applied",
                "message": "No automatic fixes could be applied",
            }


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Automatically fix workflow failures")
    parser.add_argument("--branch", help="Fix failures for specific branch")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it",
    )
    parser.add_argument("--commit", action="store_true", help="Commit the fixes")
    parser.add_argument("--push", action="store_true", help="Push the fixes to remote")

    args = parser.parse_args()

    try:
        fixer = AutoFixer(dry_run=args.dry_run)
        result = fixer.auto_fix_workflow_failures(args.branch)

        print(f"\n📊 Result: {result['message']}")

        if result["status"] == "fixes_applied" and args.push and not args.dry_run:
            print("📤 Pushing fixes to remote...")
            subprocess.run(["git", "push"], check=True)
            print("✅ Fixes pushed successfully")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
