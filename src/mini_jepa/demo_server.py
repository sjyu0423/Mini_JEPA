"""Dependency-light localhost server for the Mini-JEPA M4 demo."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mini_jepa.demo import SCENARIOS, DemoEngine, DemoPaths

ASSET_DIR = Path(__file__).parent / "demo_assets"


class DemoRequestHandler(BaseHTTPRequestHandler):
    engine: DemoEngine

    def _send_bytes(
        self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(value).encode("utf-8"), "application/json", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(self.engine.status())
            return
        if parsed.path == "/api/scenario":
            query = parse_qs(parsed.query)
            scenario = query.get("name", ["ordinary"])[0]
            try:
                seed = int(query.get("seed", ["42"])[0])
                frames = min(max(int(query.get("frames", ["180"])[0]), 30), 600)
                if scenario not in SCENARIOS:
                    raise ValueError(f"unknown scenario: {scenario}")
                payload = self.engine.predict_scenario(scenario, seed, num_frames=frames)  # type: ignore[arg-type]
            except (TypeError, ValueError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return

        relative = "index.html" if parsed.path == "/" else parsed.path.removeprefix("/")
        asset_path = (ASSET_DIR / relative).resolve()
        if ASSET_DIR.resolve() not in asset_path.parents and asset_path != ASSET_DIR.resolve():
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if not asset_path.is_file():
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        self._send_bytes(asset_path.read_bytes(), content_type)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[demo] {self.address_string()} {format % args}")


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8877,
    device: str = "auto",
    paths: DemoPaths | None = None,
) -> None:
    engine = DemoEngine(paths or DemoPaths(), device_name=device)
    handler = type("ConfiguredDemoHandler", (DemoRequestHandler,), {"engine": engine})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Mini-JEPA demo running at http://{host}:{port}")
    print(
        f"Inference device: {engine.device}; checkpoints loaded in {engine.load_latency_ms:.1f} ms"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    defaults = DemoPaths()
    parser = argparse.ArgumentParser(description="Run the local Mini-JEPA prediction demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--m1-checkpoint", type=Path, default=defaults.m1_checkpoint)
    parser.add_argument("--m1-probe", type=Path, default=defaults.m1_probe)
    parser.add_argument("--m3-checkpoint", type=Path, default=defaults.m3_checkpoint)
    parser.add_argument("--m3-probe", type=Path, default=defaults.m3_probe)
    args = parser.parse_args()
    paths = DemoPaths(args.m1_checkpoint, args.m1_probe, args.m3_checkpoint, args.m3_probe)
    serve(host=args.host, port=args.port, device=args.device, paths=paths)


if __name__ == "__main__":
    main()
