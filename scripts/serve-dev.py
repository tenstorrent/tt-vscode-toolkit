#!/usr/bin/env python3
"""
Dev HTTP server for the GH Pages site/ directory.

Adds the COOP/COEP headers required to enable SharedArrayBuffer in the browser.
SharedArrayBuffer is needed by Pyodide's threading support, which the
greenlet_shim.py uses for cooperative kernel simulation.

Usage:
    python3 scripts/serve-dev.py [port]   # default port 8000
"""

import http.server
import os
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
SITE_DIR = Path(__file__).parent.parent / "site"


class CORPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def end_headers(self):
        # These two headers together enable cross-origin isolation, which
        # unlocks SharedArrayBuffer (required for Pyodide threading / Atomics.wait).
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Suppress request noise; only show errors
        if args and isinstance(args[1], str) and not args[1].startswith("2"):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    os.chdir(SITE_DIR)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), CORPHandler)
    import socket
    hostname = socket.gethostname()
    print(f"Serving {SITE_DIR} at http://0.0.0.0:{PORT}/")
    print(f"  Local:   http://127.0.0.1:{PORT}/")
    print(f"  Network: http://{hostname}:{PORT}/")
    print("COOP/COEP headers enabled — SharedArrayBuffer active for Pyodide threading")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
