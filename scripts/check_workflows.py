#!/usr/bin/env python3
"""
GitHub Workflow Status Checker

This script provides programmatic access to GitHub workflow results
and can automatically suggest or make adjustments based on failures.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class WorkflowRun:
    """Represents a GitHub workflow run."""

    id: str
    name: str
    status: str
    conclusion: Optional[str]
    html_url: str
    head_sha: str
    head_branch: str
    event: str
    created_at: str
    updated_at: str


@dataclass
class WorkflowJob:
    """Represents a job within a workflow run."""

    id: str
    name: str
    status: str
    conclusion: Optional[str]
    html_url: str
    steps: list[dict[str, Any]]


class WorkflowChecker:
    """Check and analyze GitHub workflow results."""

    def __init__(self, repo: Optional[str] = None):
        self.repo = repo or self._get_current_repo()

    def _get_current_repo(self) -> str:
        """Get the current repository from git remote."""
        try:
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=True,
            )
            url = result.stdout.strip()
            # Extract owner/repo from URL
            if url.startswith("https://github.com/"):
                return url.replace("https://github.com/", "").replace(".git", "")
            elif url.startswith("git@github.com:"):
                return url.replace("git@github.com:", "").replace(".git", "")
            else:
                raise ValueError(f"Unable to parse repository URL: {url}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get repository URL: {e}")

    def get_workflow_runs(self, limit: int = 10) -> list[WorkflowRun]:
        """Get recent workflow runs."""
        try:
            result = subprocess.run(
                [
                    "gh",
                    "run",
                    "list",
                    "--repo",
                    self.repo,
                    "--limit",
                    str(limit),
                    "--json",
                    "id,name,status,conclusion,url,headSha,headBranch,event,createdAt,updatedAt",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            runs_data = json.loads(result.stdout)
            return [
                WorkflowRun(
                    id=run["id"],
                    name=run["name"],
                    status=run["status"],
                    conclusion=run.get("conclusion"),
                    html_url=run["url"],
                    head_sha=run["headSha"],
                    head_branch=run["headBranch"],
                    event=run["event"],
                    created_at=run["createdAt"],
                    updated_at=run["updatedAt"],
                )
                for run in runs_data
            ]
        except subprocess.CalledProcessError as e:
            print(f"Error fetching workflow runs: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing workflow runs JSON: {e}")
            return []

    def get_workflow_jobs(self, run_id: str) -> list[WorkflowJob]:
        """Get jobs for a specific workflow run."""
        try:
            result = subprocess.run(
                ["gh", "run", "view", run_id, "--repo", self.repo, "--json", "jobs"],
                capture_output=True,
                text=True,
                check=True,
            )

            data = json.loads(result.stdout)
            jobs_data = data.get("jobs", [])

            return [
                WorkflowJob(
                    id=job["id"],
                    name=job["name"],
                    status=job["status"],
                    conclusion=job.get("conclusion"),
                    html_url=job["url"],
                    steps=job.get("steps", []),
                )
                for job in jobs_data
            ]
        except subprocess.CalledProcessError as e:
            print(f"Error fetching workflow jobs: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing workflow jobs JSON: {e}")
            return []

    def get_failed_steps(self, run_id: str) -> list[dict[str, Any]]:
        """Get failed steps from a workflow run."""
        failed_steps = []
        jobs = self.get_workflow_jobs(run_id)

        for job in jobs:
            if job.conclusion == "failure":
                for step in job.steps:
                    if step.get("conclusion") == "failure":
                        failed_steps.append(
                            {
                                "job_name": job.name,
                                "step_name": step.get("name"),
                                "step_number": step.get("number"),
                                "conclusion": step.get("conclusion"),
                                "log_url": f"{job.html_url}#step:{step.get('number')}:1",
                            }
                        )

        return failed_steps

    def analyze_failures(self, branch: Optional[str] = None) -> dict[str, Any]:
        """Analyze workflow failures for a specific branch or latest runs."""
        runs = self.get_workflow_runs()

        if branch:
            runs = [run for run in runs if run.head_branch == branch]

        if not runs:
            return {"status": "no_runs", "message": "No workflow runs found"}

        latest_run = runs[0]
        analysis = {
            "run_id": latest_run.id,
            "name": latest_run.name,
            "status": latest_run.status,
            "conclusion": latest_run.conclusion,
            "branch": latest_run.head_branch,
            "url": latest_run.html_url,
            "failed_steps": [],
        }

        if latest_run.conclusion == "failure":
            analysis["failed_steps"] = self.get_failed_steps(latest_run.id)

        return analysis

    def suggest_fixes(self, analysis: dict[str, Any]) -> list[str]:
        """Suggest fixes based on workflow analysis."""
        suggestions = []

        for step in analysis.get("failed_steps", []):
            step_name = step.get("step_name", "").lower()
            job_name = step.get("job_name", "").lower()

            if "lint" in step_name or "flake8" in step_name:
                suggestions.append("Run `make format` to fix linting issues")
                suggestions.append("Run `make lint` to check for remaining issues")

            elif "black" in step_name or "format" in step_name:
                suggestions.append("Run `black src tests` to fix formatting")
                suggestions.append("Commit the formatting changes")

            elif "mypy" in step_name or "type" in step_name:
                suggestions.append("Add missing type hints")
                suggestions.append("Run `mypy src` to check type issues")

            elif "test" in step_name or "pytest" in step_name:
                suggestions.append("Run `make test` locally to debug test failures")
                suggestions.append("Check test coverage with `pytest --cov=src`")

            elif "install" in step_name or "dependencies" in step_name:
                suggestions.append("Check requirements.txt and pyproject.toml")
                suggestions.append("Ensure all dependencies are properly specified")

            elif "security" in step_name or "bandit" in step_name:
                suggestions.append("Run `bandit -r src/` to check security issues")
                suggestions.append(
                    "Fix security vulnerabilities or add # nosec comments"
                )

        return list(set(suggestions))  # Remove duplicates

    def create_status_report(self, branch: Optional[str] = None) -> str:
        """Create a human-readable status report."""
        analysis = self.analyze_failures(branch)

        if analysis["status"] == "no_runs":
            return "❌ No workflow runs found"

        status = analysis["status"]
        conclusion = analysis.get("conclusion")

        if status == "completed" and conclusion == "success":
            return f"✅ All workflows passing for branch: {analysis['branch']}"

        elif status == "completed" and conclusion == "failure":
            report = [
                f"❌ Workflow failed for branch: {analysis['branch']}",
                f"🔗 URL: {analysis['url']}",
                "",
                "Failed steps:",
            ]

            for step in analysis["failed_steps"]:
                report.append(f"  - {step['job_name']}: {step['step_name']}")

            suggestions = self.suggest_fixes(analysis)
            if suggestions:
                report.extend(["", "Suggested fixes:"])
                for suggestion in suggestions:
                    report.append(f"  - {suggestion}")

            return "\n".join(report)

        elif status == "in_progress":
            return f"🔄 Workflow in progress for branch: {analysis['branch']}"

        else:
            return f"❓ Unknown status: {status} for branch: {analysis['branch']}"


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Check GitHub workflow status")
    parser.add_argument("--branch", help="Check specific branch")
    parser.add_argument("--repo", help="Repository (owner/repo)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--suggest-fixes", action="store_true", help="Suggest fixes for failures"
    )

    args = parser.parse_args()

    try:
        checker = WorkflowChecker(args.repo)

        if args.json:
            analysis = checker.analyze_failures(args.branch)
            if args.suggest_fixes:
                analysis["suggestions"] = checker.suggest_fixes(analysis)
            print(json.dumps(analysis, indent=2))
        else:
            report = checker.create_status_report(args.branch)
            print(report)

            if args.suggest_fixes:
                analysis = checker.analyze_failures(args.branch)
                suggestions = checker.suggest_fixes(analysis)
                if suggestions:
                    print("\nSuggested fixes:")
                    for suggestion in suggestions:
                        print(f"  - {suggestion}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
