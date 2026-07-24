"""ngspice execution and output parsing with a failure taxonomy.

Categories (Phase D metric 5): ok | parse_error | singular_matrix |
floating_node | timeout | ngspice_missing | ngspice_error.

With ``simulate.diagnostics`` enabled, the parser also extracts the
node/element ngspice named in the failure (e.g. the two nodes of a
singular matrix), which the ERC/repair layer can cite as evidence.
"""

from __future__ import annotations

import re
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

# "singular matrix: check nodes n1 and n2" / "check node n3"
_NODE_RE = re.compile(r"check nodes?\s+([\w.]+)(?:\s+and\s+([\w.]+))?", re.I)
# "Warning: node n3 is floating"
_FLOAT_RE = re.compile(r"node\s+([\w.]+)\s+is floating", re.I)


def parse_ngspice_output(
    stdout: str, stderr: str, returncode: int
) -> tuple[bool, str, dict]:
    """Classify an ngspice run. Returns (ok, category, diagnostics).

    ``diagnostics`` carries any node names extracted from the failure
    message under key "nodes" (empty when none / on success).
    """
    out = (stdout or "").lower()
    err = (stderr or "").lower()
    combined = out + "\n" + err
    diag: dict = {"nodes": []}

    if "singular matrix" in combined:
        m = _NODE_RE.search(combined)
        if m:
            diag["nodes"] = [g for g in m.groups() if g]
        return False, "singular_matrix", diag
    if "floating" in combined:
        m = _FLOAT_RE.search(combined)
        if m:
            diag["nodes"] = [m.group(1)]
        return False, "floating_node", diag
    if returncode != 0:
        return False, "parse_error", diag
    return True, "ok", diag


def run_ngspice(netlist_path: str, cfg: dict) -> tuple[bool, str]:
    """Run a batch-mode DC operating point analysis. Returns (ok, category)."""
    ok, category, _ = run_ngspice_diag(netlist_path, cfg)
    return ok, category


def run_ngspice_diag(netlist_path: str, cfg: dict) -> tuple[bool, str, dict]:
    """As :func:`run_ngspice` but also returns the diagnostics dict."""
    s = cfg["simulate"]
    try:
        proc = subprocess.run(
            [s["ngspice_binary"], "-b", netlist_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=s["timeout_s"],
        )
    except subprocess.TimeoutExpired:
        return False, "timeout", {"nodes": []}
    except FileNotFoundError:
        return False, "ngspice_missing", {"nodes": []}
    except Exception:
        return False, "ngspice_error", {"nodes": []}

    return parse_ngspice_output(
        proc.stdout.decode(errors="replace"),
        proc.stderr.decode(errors="replace"),
        proc.returncode,
    )
