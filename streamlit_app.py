"""Root Streamlit entrypoint for cloud deployment.

This wrapper avoids path parsing issues in some deploy environments when
the app lives in nested directories with spaces.
"""

from __future__ import annotations

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "f93beda2-65f8-4f28-9d09-2eadd93e2b06" / "NBA All-Time Greatness Predictor" / "streamlit_app.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Expected app entrypoint not found: {TARGET}")

runpy.run_path(str(TARGET), run_name="__main__")
