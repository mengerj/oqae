#!/usr/bin/env python3
"""
Robust GitHub CLI wrapper that handles environment issues and provides fallbacks.

This module provides a reliable interface to GitHub data that works around
common GitHub CLI issues like pager configuration, authentication, and
output formatting problems.
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Any, Union
import requests
from pathlib import Path


class RobustGitHubCLI:
    """Robust GitHub CLI wrapper with fallback mechanisms."""

    def __init__(self):
        """Initialize the GitHub CLI wrapper."""
        self.setup_environment()
        self.auth_token = self._get_auth_token()
        self.repo_info = self._get_repo_info()

    def setup_environment(self) -> None:
        """Set up environment variables for reliable GitHub CLI usage."""
        # Disable pager to prevent formatting issues
        os.environ['GH_PAGER'] = ''
        os.environ['PAGER'] = ''

        # Ensure proper locale for consistent output
        os.environ['LC_ALL'] = 'C'
        os.environ['LANG'] = 'C'

        # Force color off for consistent parsing
        os.environ['GH_FORCE_TTY'] = '0'
        os.environ['NO_COLOR'] = '1'

    def _get_auth_token(self) -> Optional[str]:
        """Get GitHub authentication token."""
        try:
            result = subprocess.run(
                ['gh', 'auth', 'token'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback to environment variable
        return os.environ.get('GITHUB_TOKEN')

    def _get_repo_info(self) -> Optional[Dict[str, str]]:
        """Get repository information."""
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Parse GitHub URL
                if 'github.com' in url:
                    if url.startswith('https://github.com/'):
                        repo_part = url.replace('https://github.com/', '').replace('.git', '')
                    elif url.startswith('git@github.com:'):
                        repo_part = url.replace('git@github.com:', '').replace('.git', '')
                    else:
                        return None

                    if '/' in repo_part:
                        owner, repo = repo_part.split('/', 1)
                        return {'owner': owner, 'repo': repo, 'full_name': repo_part}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def run_gh_command(self, args: List[str], fallback_to_api: bool = True) -> Optional[Dict[str, Any]]:
        """Run GitHub CLI command with robust error handling."""
        try:
            # Try GitHub CLI first
            result = subprocess.run(
                ['gh'] + args,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # Return raw output if not JSON
                    return {'output': result.stdout.strip()}
            else:
                print(f"GitHub CLI error: {result.stderr}", file=sys.stderr)

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"GitHub CLI execution error: {e}", file=sys.stderr)

        # Fallback to direct API calls if enabled
        if fallback_to_api and self.auth_token and self.repo_info:
            return self._api_fallback(args)

        return None

    def _api_fallback(self, gh_args: List[str]) -> Optional[Dict[str, Any]]:
        """Fallback to direct GitHub API calls."""
        if not self.repo_info:
            return None

        headers = {
            'Authorization': f'token {self.auth_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        try:
            # Map common GitHub CLI commands to API endpoints
            if gh_args[:2] == ['run', 'list']:
                return self._get_workflow_runs(headers)
            elif gh_args[:2] == ['pr', 'view'] and len(gh_args) >= 3:
                pr_number = gh_args[2]
                return self._get_pull_request(pr_number, headers)
            elif gh_args[:2] == ['issue', 'list']:
                return self._get_issues(headers)
        except requests.RequestException as e:
            print(f"API fallback error: {e}", file=sys.stderr)

        return None

    def _get_workflow_runs(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Get workflow runs via API."""
        url = f"https://api.github.com/repos/{self.repo_info['full_name']}/actions/runs"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def _get_pull_request(self, pr_number: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Get pull request via API."""
        url = f"https://api.github.com/repos/{self.repo_info['full_name']}/pulls/{pr_number}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def _get_issues(self, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Get issues via API."""
        url = f"https://api.github.com/repos/{self.repo_info['full_name']}/issues"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status."""
        print("🔍 Checking workflow status...")

        # Get recent workflow runs
        runs_data = self.run_gh_command([
            'run', 'list',
            '--limit', '10',
            '--json', 'status,conclusion,headBranch,workflowName,createdAt,url'
        ])

        if not runs_data:
            return {'error': 'Could not fetch workflow runs'}

        # Process workflow data - GitHub CLI returns list directly, API returns dict with workflow_runs
        if isinstance(runs_data, list):
            runs = runs_data
        elif isinstance(runs_data, dict) and 'workflow_runs' in runs_data:
            runs = runs_data['workflow_runs']
        else:
            runs = []

        status_summary = {
            'total_runs': len(runs),
            'failed_runs': [r for r in runs if r.get('conclusion') == 'failure'],
            'successful_runs': [r for r in runs if r.get('conclusion') == 'success'],
            'in_progress': [r for r in runs if r.get('status') == 'in_progress'],
            'recent_failure': None
        }

        # Find most recent failure
        failed_runs = status_summary['failed_runs']
        if failed_runs:
            status_summary['recent_failure'] = failed_runs[0]

        return status_summary

    def get_pr_status(self, pr_number: str) -> Dict[str, Any]:
        """Get pull request status with check details."""
        print(f"🔍 Checking PR #{pr_number} status...")

        pr_data = self.run_gh_command([
            'pr', 'view', pr_number,
            '--json', 'title,state,mergeable,statusCheckRollup,number,url'
        ])

        if not pr_data:
            return {'error': f'Could not fetch PR #{pr_number}'}

        # Extract check status
        checks = pr_data.get('statusCheckRollup', [])
        check_summary = {
            'total_checks': len(checks),
            'passing': [c for c in checks if c.get('conclusion') == 'SUCCESS'],
            'failing': [c for c in checks if c.get('conclusion') == 'FAILURE'],
            'pending': [c for c in checks if c.get('status') in ['QUEUED', 'IN_PROGRESS']],
            'all_passed': all(c.get('conclusion') == 'SUCCESS' for c in checks)
        }

        return {
            'pr_info': pr_data,
            'checks': check_summary,
            'is_mergeable': pr_data.get('mergeable') == 'MERGEABLE'
        }

    def diagnose_failure(self, run_info: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose workflow failure based on available information."""
        workflow_name = run_info.get('workflowName', 'Unknown')
        duration = run_info.get('duration', 0)

        diagnosis = {
            'workflow': workflow_name,
            'likely_cause': 'Unknown',
            'suggested_fixes': [],
            'confidence': 'low'
        }

        # Pattern-based diagnosis
        if workflow_name == 'CI':
            if duration < 30:  # Quick failure usually means linting/formatting
                diagnosis.update({
                    'likely_cause': 'Code quality issues (linting, formatting, or type checking)',
                    'suggested_fixes': [
                        'Run `make format` to fix formatting issues',
                        'Run `make lint` to check for linting errors',
                        'Run `make type-check` to fix type issues',
                        'Run `make ci` locally to reproduce the issue'
                    ],
                    'confidence': 'high'
                })
            elif 30 <= duration < 120:  # Medium duration could be test failures
                diagnosis.update({
                    'likely_cause': 'Test failures or dependency issues',
                    'suggested_fixes': [
                        'Run `make test` to check for test failures',
                        'Check for missing dependencies in requirements.txt',
                        'Verify test environment setup'
                    ],
                    'confidence': 'medium'
                })

        return diagnosis


def main():
    """Main function for testing the robust GitHub CLI."""
    gh = RobustGitHubCLI()

    print("🚀 Testing Robust GitHub CLI")
    print(f"Repository: {gh.repo_info}")
    print(f"Authentication: {'✅' if gh.auth_token else '❌'}")

    # Test workflow status
    workflow_status = gh.get_workflow_status()
    print(f"\n📊 Workflow Status:")
    print(f"Total runs: {workflow_status.get('total_runs', 0)}")
    print(f"Failed runs: {len(workflow_status.get('failed_runs', []))}")
    print(f"Successful runs: {len(workflow_status.get('successful_runs', []))}")

    if workflow_status.get('recent_failure'):
        print(f"\n🚨 Recent Failure Detected:")
        failure = workflow_status['recent_failure']
        diagnosis = gh.diagnose_failure(failure)
        print(f"Workflow: {diagnosis['workflow']}")
        print(f"Likely cause: {diagnosis['likely_cause']}")
        print(f"Confidence: {diagnosis['confidence']}")


if __name__ == "__main__":
    main()
