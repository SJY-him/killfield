#!/usr/bin/env python3
"""Local dev server for the viewer.

`python -m http.server` lets the browser cache ES modules heuristically, and
only some of them carry a `?v=` stamp: `build.sh` versions `viewer.js`,
`i18n.js` and the Hybrid weights, but nothing versions `src/mouse-aim.js`,
`src/opponent.js` or the `constants.js` those import in turn. Rebuild while one
of those is cached and the page dies on a stale module with an error that
points at the source rather than at the cache::

    SyntaxError: The requested module './src/mouse-aim.js'
    does not provide an export named 'AIM_MODE_AIM'

Stamping every module transitively would mean hashing them in dependency order,
because stamping a leaf rewrites its importers and so changes their hashes too.
For a machine that is only ever serving itself it is far simpler to refuse to
cache anything, which is what this does.

Usage:

    python viewer/serve.py            # http://127.0.0.1:8000
    python viewer/serve.py 8001       # another port
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import socket
import socketserver
import sys
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the viewer, telling the browser to keep none of it."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # One line per request is useful; the default's date prefix is not.
        sys.stderr.write("%s\n" % (fmt % args))


class Server(socketserver.TCPServer):
    # Without this a Ctrl-C leaves the port in TIME_WAIT and the next start
    # fails with "address already in use" for a minute or so.
    allow_reuse_address = True


def main(argv: list[str]) -> int:
    port = 8000
    if len(argv) > 1:
        try:
            port = int(argv[1])
        except ValueError:
            print(f"port must be a number, got {argv[1]!r}", file=sys.stderr)
            return 2
        if not 1 <= port <= 65535:
            print(f"port must be 1-65535, got {port}", file=sys.stderr)
            return 2

    handler = functools.partial(NoCacheHandler, directory=str(VIEWER_DIR))
    try:
        server = Server(("127.0.0.1", port), handler)
    except OSError as error:
        if error.errno in (socket.EADDRINUSE, 10048):  # 10048 is Windows'
            print(f"port {port} is already in use — stop that server, or:", file=sys.stderr)
            print(f"  python viewer/serve.py {port + 1}", file=sys.stderr)
            return 1
        raise

    print(f"  http://127.0.0.1:{port}   (serving {VIEWER_DIR}, caching off)")
    print("  Ctrl-C to stop\n")
    with server, contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
