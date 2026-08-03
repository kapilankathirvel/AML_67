"""The Streamlit entry point must import under the launcher a host actually uses.

`streamlit run frontend/app.py` puts only the script's own folder on sys.path
(streamlit/web/bootstrap.py: sys.path.insert(0, dirname(main_script_path))).
The repo root never gets added, so every `from frontend import ...` and
`from backend import ...` in app.py depends on something else having put it
there.

Locally that something else is run_demo.py, which launches via
`python -m streamlit` -- and -m puts the working directory on sys.path. That
accident hides the problem completely: the app runs perfectly on a developer's
machine and dies on the first import on any host that uses the console script,
which is all of them.

This test removes the accident and checks the entry point still imports. It is
the only test in the suite that would have caught it; the failure is invisible
to every other one, because pytest runs from the repo root.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# What bootstrap.py does, and nothing else: only the script's folder is added.
_PROBE = textwrap.dedent(
    """
    import os, sys

    repo = {repo!r}
    sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(repo)]
    sys.path.insert(0, os.path.join(repo, "frontend"))

    assert not any(os.path.abspath(p) == os.path.abspath(repo) for p in sys.path), (
        "the probe failed to remove the repo root; the test would pass vacuously"
    )

    {body}
    """
)


def _run_isolated(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a subprocess whose sys.path mimics `streamlit run`.

    A subprocess rather than manipulating sys.path in-process, because pytest
    has already imported half of these modules and sys.modules would satisfy
    the imports from cache -- the test would pass without proving anything.

    PYTHONPATH is cleared for the same reason: a developer with the repo on it
    would get a green test for a broken deployment.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=str(_REPO_ROOT), body=body)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT.parent,  # never the repo root itself
        env=env,
    )


def test_the_probe_itself_detects_the_broken_case():
    """Guard the guard.

    If the isolation stopped working, every other test here would pass while
    proving nothing. `backend` is imported directly -- with no path fix of its
    own it MUST fail under these conditions.
    """
    proc = _run_isolated("import backend.main")
    assert proc.returncode != 0, "isolation is not working: backend imported without the repo root"
    assert "ModuleNotFoundError" in proc.stderr


def test_app_module_adds_the_repo_root_itself():
    """The fix: importing the entry point must work with only frontend/ on the path."""
    proc = _run_isolated(
        "import importlib.util, os\n"
        "spec = importlib.util.spec_from_file_location("
        "    'app_probe', os.path.join({repo!r}, 'frontend', 'app.py'))\n".format(repo=str(_REPO_ROOT))
        + "mod = importlib.util.module_from_spec(spec)\n"
        "import sys as _s\n"
        "# Execute only up to the sys.path fix and the imports; Streamlit calls\n"
        "# below would need a runtime, so stop before them by importing the\n"
        "# sibling modules the fix is there to enable.\n"
        "src = open(spec.origin, encoding='utf-8').read()\n"
        "head = src.split('# ---------------------------------------------------------------------------')[0]\n"
        "exec(compile(head, spec.origin, 'exec'), mod.__dict__)\n"
        "import frontend.api_client, frontend.components.plan_trace, backend.main\n"
        "print('OK')"
    )
    assert proc.returncode == 0, f"entry point does not bootstrap its own path:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_the_path_fix_precedes_the_package_imports():
    """Order is the whole point, and it is easy to lose to an import sorter.

    Moving the sys.path insert below `from frontend import api_client` would
    leave the file looking tidy and the deployment broken.
    """
    src = (_REPO_ROOT / "frontend" / "app.py").read_text(encoding="utf-8")

    fix_at = src.index("sys.path.insert(0, str(_REPO_ROOT))")
    first_package_import = min(
        src.index("from frontend import api_client"),
        src.index("from frontend.components"),
    )
    assert fix_at < first_package_import, (
        "the sys.path fix must come before the first frontend/backend import"
    )
