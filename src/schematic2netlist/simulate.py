"""ngspice execution and output parsing with a failure taxonomy.

Categories (Phase D metric 5): ok | parse_error | singular_matrix |
floating_node | timeout | ngspice_missing | ngspice_error.
"""

from __future__ import annotations

import subprocess

FAILURE_CATEGORIES = (
    "ok",
    "parse_error",
    "singular_matrix",
    "floating_node",
    "timeout",
    "ngspice_missing",
    "ngspice_error",
)


def parse_ngspice_output(stdout: str, stderr: str, returncode: int) -> tuple[bool, str]:
    """Classify an ngspice run. Returns (ok, category)."""
    out = (stdout or "").lower()
    err = (stderr or "").lower()
    combined = out + "\n" + err

    if "singular matrix" in combined:
        return False, "singular_matrix"
    if "floating" in combined:
        return False, "floating_node"
    if returncode != 0:
        return False, "parse_error"
    return True, "ok"


def run_ngspice(netlist_path: str, cfg: dict) -> tuple[bool, str]:
    """Run a batch-mode DC operating point analysis on a netlist."""
    s = cfg["simulate"]
    try:
        proc = subprocess.run(
            [s["ngspice_binary"], "-b", netlist_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=s["timeout_s"],
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "ngspice_missing"
    except Exception:
        return False, "ngspice_error"

    return parse_ngspice_output(
        proc.stdout.decode(errors="replace"),
        proc.stderr.decode(errors="replace"),
        proc.returncode,
    )
