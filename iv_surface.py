"""Public entry point for the implied-volatility surface application.

The implementation lives in :mod:`iv_engine`; run this file to start the app.
"""
from __future__ import annotations

import traceback

import iv_engine


def main() -> None:
    """Run the interactive implied-volatility surface workflow."""
    iv_engine.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        traceback.print_exc()
    finally:
        iv_engine.pause_before_exit()
