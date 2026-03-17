#!/usr/bin/env python3
"""
run.py — PINN Research Engine entry point
==========================================
Usage:
    python run.py [--port PORT] [--no-browser]

Opens http://localhost:8765 automatically in your default browser.
Press Ctrl+C to stop.
"""

import argparse
import socket
import sys
import threading
import time
import webbrowser


def find_free_port(start: int = 8765) -> int:
    for p in range(start, start + 20):
        with socket.socket() as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    return start


def open_browser(port: int, delay: float = 1.5):
    def _open():
        time.sleep(delay)
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=_open, daemon=True).start()


def check_deps():
    missing = []
    for pkg in ("torch", "fastapi", "uvicorn", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[PINN] Missing packages: {', '.join(missing)}")
        print(f"[PINN] Run:  pip install {' '.join(missing)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="PINN Research Engine")
    parser.add_argument("--port",       type=int,  default=0,
                        help="Port (default: auto-select from 8765)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser automatically")
    args = parser.parse_args()

    check_deps()

    import torch
    import uvicorn
    from src.server import app, DEVICE

    port = args.port if args.port else find_free_port(8765)

    # Patch port into the health / HTML responses at runtime
    # The frontend reads __PORT__ replaced at serve-time — handled in server.py

    print()
    print("=" * 58)
    print("  PINN Research Engine — Autonomous Self-Training")
    print(f"  Device  : {DEVICE}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  URL     : http://localhost:{port}")
    print("=" * 58)
    print()
    print("  Browser opens automatically.")
    print("  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        open_browser(port)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
