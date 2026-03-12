"""
Submit insurance-bcf tests to Databricks — full pytest output captured.

Credentials are loaded from ~/.config/burning-cost/databricks.env.
"""

import base64
import os
import sys
import time

# Load Databricks credentials from env file
_env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    NotebookTask,
    RunLifeCycleState,
    RunResultState,
    SubmitTask,
)
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
HOST = os.environ.get("DATABRICKS_HOST", "")

notebook_content = """\
# Databricks notebook source
import subprocess
import sys

# Install dependencies
subprocess.run(
    ["pip", "install", "--quiet",
     "stochtree>=0.4.0",
     "git+https://github.com/burning-cost/insurance-bcf.git",
     "pytest", "jinja2", "matplotlib"],
    check=True, capture_output=True
)
print("Install OK")

# Clone the repo for tests
subprocess.run(["rm", "-rf", "/tmp/insurance-bcf"], capture_output=True)
result = subprocess.run(
    ["git", "clone", "--depth=1",
     "https://github.com/burning-cost/insurance-bcf.git",
     "/tmp/insurance-bcf"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("Clone STDERR:", result.stderr)
    raise Exception("Clone failed")
print("Clone OK")

# Run tests
r = subprocess.run(
    ["python", "-m", "pytest",
     "/tmp/insurance-bcf/tests/",
     "--tb=short", "-v", "--no-header", "-p", "no:warnings"],
    capture_output=True, text=True,
    cwd="/tmp/insurance-bcf"
)

full_output = r.stdout + ("\\nSTDERR:\\n" + r.stderr if r.stderr else "")
print(full_output)

with open("/tmp/pytest_output.txt", "w") as f:
    f.write(full_output)

result_snippet = full_output[-5000:]
dbutils.notebook.exit(result_snippet)
"""

notebook_path = "/Workspace/Shared/insurance-bcf-tests-v1"
print("Uploading notebook...")
encoded = base64.b64encode(notebook_content.encode()).decode()
w.workspace.import_(
    path=notebook_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded,
    overwrite=True,
)
print("Upload OK")

print("Submitting (serverless)...")
run_response = w.jobs.submit(
    run_name="insurance-bcf-pytest",
    tasks=[
        SubmitTask(
            task_key="pytest",
            notebook_task=NotebookTask(notebook_path=notebook_path),
        )
    ],
)

run_id = run_response.response.run_id
print(f"Run ID: {run_id}")
print(f"URL: {HOST}#job/runs/{run_id}")

print("\nPolling...")
while True:
    run = w.jobs.get_run(run_id=run_id)
    state = run.state
    lc = state.life_cycle_state
    print(f"  [{time.strftime('%H:%M:%S')}] {lc.value if lc else '?'}")
    if lc in (RunLifeCycleState.TERMINATED, RunLifeCycleState.SKIPPED, RunLifeCycleState.INTERNAL_ERROR):
        result_state = state.result_state
        print(f"  Result: {result_state.value if result_state else '?'}")
        print(f"  Msg: {state.state_message}")
        break
    time.sleep(30)

print("\n=== Task output ===")
try:
    tasks = run.tasks
    if tasks:
        task_run_id = tasks[0].run_id
        output = w.jobs.get_run_output(run_id=task_run_id)
        if output.notebook_output and output.notebook_output.result:
            print("NOTEBOOK RESULT:")
            print(output.notebook_output.result)
        if output.logs:
            print("LOGS:")
            print(output.logs[-6000:])
        if output.error:
            print("ERROR:", output.error)
        if output.error_trace:
            print("TRACE:", output.error_trace[-2000:])
except Exception as e:
    print(f"Output error: {e}")

success = (state.result_state == RunResultState.SUCCESS)
print(f"\n=== TEST {'PASSED' if success else 'FAILED'} ===")
sys.exit(0 if success else 1)
