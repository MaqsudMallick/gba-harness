"""Serve the local append-only Zelda decision trace dashboard."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DEFAULT_TRACE_DIR = ROOT.parent / "artifacts" / "zelda-live-trace"


class TraceDashboardHandler(SimpleHTTPRequestHandler):
    """Serve the dashboard and the selected trace directory."""

    server_version = "ZeldaTraceDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(
                ROOT / "zelda_trace_dashboard.html",
                "text/html; charset=utf-8",
                no_cache=True,
            )
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/trace":
            trace_path = self.server.trace_dir / "trace.json"  # type: ignore[attr-defined]
            if not trace_path.is_file():
                self._send_json(
                    {
                        "capture": {"call_count": 0, "execution_count": 0},
                        "calls": [],
                        "executions": [],
                    }
                )
                return
            self._send_file(trace_path, "application/json; charset=utf-8", no_cache=True)
            return
        if parsed.path.startswith("/trace/"):
            relative = Path(parsed.path.removeprefix("/trace/"))
            trace_dir = self.server.trace_dir.resolve()  # type: ignore[attr-defined]
            requested = (trace_dir / relative).resolve()
            if trace_dir not in requested.parents or not requested.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = "image/png" if requested.suffix.lower() == ".png" else "text/plain"
            self._send_file(requested, content_type, no_cache=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, value: object) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str, *, no_cache: bool = False) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if no_cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class TraceDashboardServer(ThreadingHTTPServer):
    trace_dir: Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    return result


def main() -> int:
    args = parser().parse_args()
    trace_dir = args.trace_dir.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    server = TraceDashboardServer((args.host, args.port), TraceDashboardHandler)
    server.trace_dir = trace_dir
    print(f"Zelda trace dashboard: http://{args.host}:{args.port}", flush=True)
    print(f"Trace directory: {trace_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
