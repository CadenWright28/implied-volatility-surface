from __future__ import annotations

import traceback

import iv_engine


if __name__ == "__main__":
    try:
        iv_engine.main()
    except Exception as exc:
        print(f"\n{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        iv_engine.pause_before_exit()
