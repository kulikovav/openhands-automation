"""Tests for the dispatcher dispatch-retry behavior (former image self-check).

Asserts the retry behavior is present in the dispatcher and config: the
deployment sizes the queue through AUTOMATION_MAX_CONCURRENT_RUNS, so a change
that drops the requeue must fail here.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

dispatcher_path = REPO_ROOT / "openhands" / "automation" / "dispatcher.py"
config_path = REPO_ROOT / "openhands" / "automation" / "config.py"

dispatcher_source = dispatcher_path.read_text(encoding="utf-8")
config_source = config_path.read_text(encoding="utf-8")


def test_config_exposes_retry_settings():
    for setting in (
        "dispatcher_max_dispatch_retries",
        "AUTOMATION_DISPATCHER_MAX_DISPATCH_RETRIES",
    ):
        assert setting in config_source, f"{setting} is missing from the config"


def test_config_keeps_max_concurrent_runs():
    # The deployment sizes the queue through AUTOMATION_MAX_CONCURRENT_RUNS, an
    # upstream setting, so a bump that drops it must fail the test.
    assert "max_concurrent_runs" in config_source, "max_concurrent_runs is missing"


def test_dispatcher_requeues_transient_failures_only():
    assert "_requeue_transient_failure" in dispatcher_source, "the requeue is missing"

    tree = ast.parse(dispatcher_source)
    fail_helper = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_fail"
        ),
        None,
    )
    assert fail_helper is not None, "_fail is missing from the dispatcher"
    fail_source = ast.get_source_segment(dispatcher_source, fail_helper) or ""
    assert '"transient"' in fail_source, "_fail must requeue only transient failures"
    assert "retryable" in fail_source, "_fail must gate the requeue on retryable"


def test_execution_context_failure_is_retryable():
    # A failure to get the execution context is the dispatch-phase failure the
    # retry exists for; the entrypoint start is deliberately not retryable.
    assert "Failed to get execution context" in dispatcher_source
    execution_context_fail = dispatcher_source.split(
        'await _fail(\n            "Failed to get execution context"', 1
    )
    assert len(execution_context_fail) == 2, (
        "the execution context failure must go through _fail"
    )
    assert "retryable=True" in execution_context_fail[1][:200], (
        "the execution context failure must be retryable"
    )
