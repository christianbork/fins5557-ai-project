"""Load settings and API keys without hardcoding them.

Locally: values come from a .env file (see .env.example).
On Streamlit Cloud: add the same keys under App settings -> Secrets.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return a secret from the environment, then Streamlit secrets, else default."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
