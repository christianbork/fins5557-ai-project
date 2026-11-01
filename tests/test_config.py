"""Tests for src/config.py."""

import os
import subprocess
import sys

from src.config import get_secret


def test_get_secret_prefers_environment(monkeypatch):
    monkeypatch.setenv("FINS_TEST_SECRET", "from-env")
    assert get_secret("FINS_TEST_SECRET", "default") == "from-env"


def test_get_secret_default_when_missing(monkeypatch):
    monkeypatch.delenv("FINS_TEST_SECRET", raising=False)
    assert get_secret("FINS_TEST_SECRET", "default") == "default"


def test_config_does_not_import_streamlit_outside_the_app():
    """Importing streamlit sets MPLBACKEND=Agg, which stops notebook charts from showing.
    Run in a fresh Python so other tests can't have imported streamlit already."""
    code = (
        "import os, sys; import src.config as c; c.get_secret('FINS_MISSING_KEY'); "
        "print('streamlit' in sys.modules, os.environ.get('MPLBACKEND'))"
    )
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.split() == ["False", "None"]
