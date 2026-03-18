"""WASM compatibility checks for the pattern designer notebook.

Validates that every package listed in the notebook's PEP 723 script header
and micropip install cell either:
  - ships as a pure-Python wheel on PyPI (safe for micropip), or
  - is a known Pyodide built-in (already available in WASM, no install needed).

Run with:  pytest tests/test_wasm_compat.py -v
These tests make network calls; skip them offline with:
  pytest -m "not network"
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "pattern_designer.py"

# Packages shipped as compiled wheels inside Pyodide itself.
# Micropip recognises these and skips PyPI lookup.
# Source: https://pyodide.org/en/stable/usage/packages-in-pyodide.html
PYODIDE_BUILTINS: set[str] = {
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "pillow",
    "pydantic",
    "pyodide",
    "micropip",
    "typing-extensions",
    "packaging",
    "certifi",
    "idna",
    "charset-normalizer",
    "requests",
    "six",
    "python-dateutil",
    "pytz",
    "attrs",
    "markupsafe",
    "jinja2",
    "pyparsing",
    "cycler",
    "kiwisolver",
    "fonttools",
    "contourpy",
    "traitlets",
    "comm",
    "ipython",
    "ipywidgets",
}

# Packages to skip in WASM checks (e.g. the project itself, installed from
# PyPI after publish; its WASM-compatibility is validated by its dep list).
SKIP_PYPI_CHECK: set[str] = {"pattern-fill"}

# Packages that are explicitly NOT safe in WASM (have native extensions,
# no pure-Python wheel, or pull in numba/LLVM etc.).
KNOWN_WASM_INCOMPATIBLE: set[str] = {
    "numba",
    "llvmlite",
    "anthropic",
    "pydantic-ai",
    "pydantic-ai-slim",
    "torch",
    "tensorflow",
    "jax",
    "umap-learn",
    "hdbscan",
    "stumpy",
}


def _normalise(name: str) -> str:
    """Canonicalise a package name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _strip_specifier(dep: str) -> str:
    """Return just the package name from a dependency specifier."""
    return re.split(r"[>=<!;\[]", dep.strip())[0].strip()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_script_header_deps(text: str) -> list[str]:
    """Extract deps from the PEP 723 ``# /// script`` block."""
    match = re.search(
        r"# /// script\s*\n(.*?)# ///",
        text,
        re.DOTALL,
    )
    if not match:
        return []
    block = match.group(1)
    deps: list[str] = []
    in_deps = False
    for line in block.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if stripped == "]":
                break
            # Strip surrounding quotes and trailing comma
            dep = re.sub(r'^["\'\s]+|["\',\s]+$', "", stripped)
            if dep:
                deps.append(dep)
    return deps


def parse_micropip_installs(text: str) -> list[str]:
    """Extract package specs from ``micropip.install(...)`` calls."""
    return re.findall(r'micropip\.install\(["\']([^"\']+)["\']', text)


# ---------------------------------------------------------------------------
# Network helper
# ---------------------------------------------------------------------------

def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


def has_pure_python_wheel(package_spec: str) -> tuple[bool, str]:
    """Return (ok, reason).

    ok=True  → package has at least one pure-Python wheel on PyPI
    ok=False → no pure-Python wheel found (will break micropip in WASM)
    """
    name = _normalise(_strip_specifier(package_spec))

    if name in {_normalise(p) for p in PYODIDE_BUILTINS}:
        return True, "pyodide built-in"

    if name in {_normalise(p) for p in SKIP_PYPI_CHECK}:
        return True, "project package — WASM-compatibility validated via its deps"

    try:
        data = _fetch_json(f"https://pypi.org/pypi/{name}/json")
    except Exception as exc:
        return False, f"PyPI lookup failed: {exc}"

    version = data["info"]["version"]
    releases = data["releases"].get(version, [])

    for wheel in releases:
        filename: str = wheel.get("filename", "")
        if not filename.endswith(".whl"):
            continue
        # Pure-Python wheels have "py3-none-any" or "py2.py3-none-any"
        # in their filename tag.
        if "-none-any.whl" in filename or filename.endswith("-py3-none-any.whl"):
            return True, f"pure-Python wheel found in {version}"

    # Some packages ship only sdist; micropip can handle pure-Python sdists
    # if there's no compiled code.  Flag these as a warning, not a hard fail,
    # unless they're in the known-incompatible list.
    if name in {_normalise(p) for p in KNOWN_WASM_INCOMPATIBLE}:
        return False, f"known WASM-incompatible package: {name}"

    # No wheel at all — likely a native extension package
    if not releases:
        return False, f"no release files found for {name}=={version}"

    has_wheel = any(f.get("filename", "").endswith(".whl") for f in releases)
    if not has_wheel:
        return True, "sdist only — likely pure Python (manual verification recommended)"

    return False, f"only native wheels found for {name}=={version}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestNotebookWasmCompat:
    @pytest.fixture(scope="class")
    def notebook_text(self):
        return NOTEBOOK.read_text()

    def test_notebook_exists(self, notebook_text):
        assert notebook_text, "Notebook is empty"

    def test_script_header_has_no_known_incompatible_packages(self, notebook_text):
        """Fail fast if a known WASM-incompatible package sneaks into the header."""
        deps = parse_script_header_deps(notebook_text)
        assert deps, "No deps found in script header — check parsing"

        bad = []
        for dep in deps:
            name = _normalise(_strip_specifier(dep))
            if name in {_normalise(p) for p in KNOWN_WASM_INCOMPATIBLE}:
                bad.append(dep)

        assert not bad, (
            f"Script header contains WASM-incompatible packages: {bad}\n"
            "Remove them or move to a non-WASM code path."
        )

    def test_micropip_installs_have_pure_python_wheels(self, notebook_text):
        """Every package in the micropip install cell must have a pure-Python wheel."""
        specs = parse_micropip_installs(notebook_text)
        assert specs, "No micropip.install() calls found — check parsing"

        failures = []
        for spec in specs:
            ok, reason = has_pure_python_wheel(spec)
            if not ok:
                failures.append(f"  {spec!r}: {reason}")

        assert not failures, (
            "The following micropip packages are not WASM-compatible:\n"
            + "\n".join(failures)
            + "\nSee https://pyodide.org/en/stable/usage/faq.html"
        )

    def test_script_header_packages_have_pure_python_wheels(self, notebook_text):
        """Every package in the script header must have a pure-Python wheel
        or be a Pyodide built-in."""
        deps = parse_script_header_deps(notebook_text)

        failures = []
        for dep in deps:
            ok, reason = has_pure_python_wheel(dep)
            if not ok:
                failures.append(f"  {dep!r}: {reason}")

        assert not failures, (
            "The following script-header packages are not WASM-compatible:\n"
            + "\n".join(failures)
            + "\nRemove them from the # /// script block or ensure a pure-Python wheel exists."
        )
