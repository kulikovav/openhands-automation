"""Tests for the presets' SIGTERM interrupt handler (former image self-check).

The bash service enforces the automation timeout with SIGTERM; the handler
must interrupt the running conversation and exit 143, and must be registered
before the run starts. Runs the handler from each preset against stubs.
"""

import ast
import sys
import threading
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

PRESETS = (
    "openhands/automation/presets/prompt/sdk_main.py",
    "openhands/automation/presets/plugin/sdk_main.py",
)


class StubConversation:
    """Records whether the handler interrupted the running conversation."""

    def __init__(self):
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True


class StubOs:
    """Records the exit code the handler passes to os._exit."""

    def __init__(self):
        self.exit_code = None

    def _exit(self, code):
        self.exit_code = code


def imports_signal(tree):
    """Return whether the module imports the signal module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "signal" for alias in node.names
        ):
            return True
    return False


def find_handler(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_interrupt_on_sigterm":
            return node
    return None


def find_registration_line(tree):
    """Return the line of signal.signal(signal.SIGTERM, _interrupt_on_sigterm)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "signal"
            and isinstance(func.value, ast.Name)
            and func.value.id == "signal"
        ):
            continue
        if len(node.args) != 2:
            continue
        first, second = node.args
        if (
            isinstance(first, ast.Attribute)
            and first.attr == "SIGTERM"
            and isinstance(first.value, ast.Name)
            and first.value.id == "signal"
            and isinstance(second, ast.Name)
            and second.id == "_interrupt_on_sigterm"
        ):
            return node.lineno
    return None


def find_run_line(tree):
    """Return the line of the first conversation.run() call."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "conversation"
        ):
            return node.lineno
    return None


@pytest.mark.parametrize("relative", PRESETS)
def test_sigterm_handler_interrupts_and_exits_143(relative):
    path = REPO_ROOT / relative
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert imports_signal(tree), f"{relative}: signal is not imported"

    registration_line = find_registration_line(tree)
    assert registration_line is not None, (
        f"{relative}: SIGTERM handler is not registered"
    )
    run_line = find_run_line(tree)
    assert run_line is not None, f"{relative}: conversation.run() is missing"
    assert registration_line < run_line, (
        f"{relative}: SIGTERM handler is registered after conversation.run()"
    )

    handler = find_handler(tree)
    assert handler is not None, f"{relative}: _interrupt_on_sigterm is missing"

    conversation = StubConversation()
    stub_os = StubOs()
    namespace = {
        "conversation": conversation,
        "os": stub_os,
        "sys": sys,
        "threading": threading,
    }
    exec(
        compile(ast.Module(body=[handler], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace["_interrupt_on_sigterm"](15, None)
    assert conversation.interrupted, f"{relative}: handler did not interrupt"
    assert stub_os.exit_code == 143, f"{relative}: handler did not exit with 143"
