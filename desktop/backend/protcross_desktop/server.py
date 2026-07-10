"""Small local HTTP API for the ProtCross Desktop sidecar."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .service import DesktopBackend


class DesktopRequestHandler(BaseHTTPRequestHandler):
    backend: DesktopBackend
    auth_token: str | None = None

    def do_OPTIONS(self) -> None:
        try:
            self._require_allowed_origin()
            self.send_response(204)
            self._cors_headers()
            self.send_header("content-length", "0")
            self.end_headers()
        except Exception as exc:
            self._error(exc)

    def do_GET(self) -> None:
        try:
            self._require_allowed_origin()
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path == "/health":
                self._json({"ok": True, "service": "protcross-desktop"})
            elif path == "/status":
                self._require_auth(parsed_url)
                self._json(self.backend.status())
            elif path == "/file":
                self._require_auth(parsed_url)
                values = parse_qs(parsed_url.query).get("path", [])
                if not values:
                    raise ValueError("Missing file path.")
                self._file(values[0])
            elif path.startswith("/batch/"):
                self._require_auth(parsed_url)
                parts = path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "batch" and parts[2] == "result":
                    values = parse_qs(parsed_url.query).get("input_structure", [])
                    if not values:
                        raise ValueError("Missing input_structure.")
                    self._json(self.backend.batch_item_result(parts[1], values[0]))
                elif len(parts) == 2:
                    query = parse_qs(parsed_url.query)
                    limit = _optional_int(query.get("limit", [None])[0])
                    offset = _optional_int(query.get("offset", [None])[0]) or 0
                    if limit is None:
                        self._json(self.backend.batch_status(parts[1], offset=offset))
                    else:
                        self._json(self.backend.batch_status(parts[1], limit=limit, offset=offset))
                else:
                    self._json({"error": "not found"}, status=404)
            elif path.startswith("/asset-download/"):
                self._require_auth(parsed_url)
                parts = path.strip("/").split("/")
                if len(parts) == 2:
                    self._json(self.backend.esm_download_status(parts[1]))
                else:
                    self._json({"error": "not found"}, status=404)
            else:
                self._json({"error": "not found"}, status=404)
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        try:
            self._require_allowed_origin()
            self._require_auth(urlparse(self.path))
            path = urlparse(self.path).path
            payload = self._payload()
            if path == "/license/confirm":
                self._json(self.backend.confirm_esm_license(**payload))
            elif path == "/backend/configure":
                self._json(self.backend.configure_backend(**payload))
            elif path == "/backend/test":
                self._json(self.backend.test_backend(**payload))
            elif path == "/assets/import-esm":
                self._json(self.backend.import_esm_weights(**payload))
            elif path == "/assets/import-checkpoint":
                self._json(self.backend.import_checkpoint(**payload))
            elif path == "/assets/import-pca":
                self._json(self.backend.import_pca(**payload))
            elif path == "/assets/download-esm/start":
                self._json(self.backend.start_esm_download(**payload))
            elif path == "/predict":
                self._json(self.backend.predict_single(**payload))
            elif path == "/inspect":
                self._json(self.backend.inspect_input_structure(**payload))
            elif path == "/batch":
                self._json(self.backend.submit_batch(**payload))
            elif path.startswith("/batch/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                self._json(self.backend.cancel_batch(job_id))
            elif path.startswith("/asset-download/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                self._json(self.backend.cancel_esm_download(job_id))
            elif path == "/diagnostics/export":
                self._json({"path": self.backend.export_diagnostics(**payload)})
            else:
                self._json({"error": "not found"}, status=404)
        except Exception as exc:
            self._error(exc)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if length == 0:
            return {}
        content_type = self.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise PermissionError("Desktop API POST requests must use application/json.")
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def _json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: str) -> None:
        file_path = self.backend.readable_output_file(path)
        stat = file_path.stat()
        self.send_response(200)
        self._cors_headers()
        suffix = file_path.suffix.lower()
        content_type = "chemical/x-mmcif" if suffix in {".cif", ".mmcif"} else "chemical/x-pdb"
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(stat.st_size))
        self.end_headers()
        with file_path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)

    def _cors_headers(self) -> None:
        origin = self.headers.get("origin")
        if origin and _origin_allowed(origin):
            self.send_header("access-control-allow-origin", origin)
            self.send_header("vary", "Origin")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "authorization, content-type, x-protcross-desktop-token")

    def _require_allowed_origin(self) -> None:
        origin = self.headers.get("origin")
        if origin and not _origin_allowed(origin):
            raise PermissionError(f"Origin is not allowed: {origin}")

    def _require_auth(self, parsed_url) -> None:
        if not self.auth_token:
            return
        auth_header = self.headers.get("authorization", "")
        header_token = self.headers.get("x-protcross-desktop-token")
        query_token = parse_qs(parsed_url.query).get("token", [None])[0]
        bearer = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else None
        if self.auth_token not in {header_token, query_token, bearer}:
            raise PermissionError("Missing or invalid desktop API token.")

    def _error(self, exc: Exception) -> None:
        if isinstance(exc, KeyError):
            status = 404
        elif isinstance(exc, PermissionError):
            status = 403
        elif isinstance(exc, (FileNotFoundError, ValueError, RuntimeError)):
            status = 400
        else:
            status = 500
        self._json({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, status=status)


def create_server(
    host: str,
    port: int,
    *,
    root: str | Path | None = None,
    token: str | None = None,
    backend: DesktopBackend | None = None,
) -> ThreadingHTTPServer:
    handler = type("BoundDesktopRequestHandler", (DesktopRequestHandler,), {})
    handler.backend = backend or DesktopBackend(root=root)
    handler.auth_token = token or os.environ.get("PROTCROSS_DESKTOP_TOKEN")
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ProtCross Desktop local backend API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--root", default=None)
    parser.add_argument("--token", default=os.environ.get("PROTCROSS_DESKTOP_TOKEN"))
    args = parser.parse_args(argv)

    server = create_server(args.host, args.port, root=args.root, token=args.token)
    actual_port = server.server_address[1]
    print(f"ProtCross Desktop backend listening on http://{args.host}:{actual_port}", flush=True)
    server.serve_forever()
    return 0


def _origin_allowed(origin: str | None) -> bool:
    if not origin:
        return True
    configured = os.environ.get("PROTCROSS_DESKTOP_ALLOWED_ORIGINS")
    allowed = {
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    }
    if configured:
        allowed.update(item.strip() for item in configured.split(",") if item.strip())
    return origin in allowed


def _optional_int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"Expected integer query value, got {value!r}") from exc
    if number < 0:
        raise ValueError("Integer query values must be >= 0.")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
