import argparse
import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from wiw_manip.envs.libero_env_core import LocalLiberoEnv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("libero_env_server")

ENV = None


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _json_response(handler, payload, status=200):
    body = json.dumps(_to_jsonable(payload)).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    @staticmethod
    def _thread_id():
        return threading.get_ident()

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        global ENV
        if self.path == "/health":
            _json_response(self, {"ok": True, "configured": ENV is not None})
            return
        _json_response(self, {"error": "not found"}, status=404)

    def do_POST(self):
        global ENV
        payload = self._read_json()
        try:
            if self.path == "/configure":
                logger.info("POST /configure on thread_id=%s", self._thread_id())
                if ENV is not None:
                    ENV.close()
                ENV = LocalLiberoEnv(
                    eval_set=payload["eval_set"],
                    img_size=tuple(payload["img_size"]),
                    down_sample_ratio=payload["down_sample_ratio"],
                    log_path=payload["log_path"],
                    max_step=payload["max_step"],
                    action_repeat=int(payload.get("action_repeat", 50)),
                )
                _json_response(self, {"state": ENV.export_state()})
                return
            if ENV is None:
                _json_response(self, {"error": "server not configured"}, status=400)
                return
            if self.path == "/init_dataset_and_tasks":
                ENV.init_dataset_and_tasks(
                    eval_task=payload["eval_task"],
                    down_sample_ratio=payload["down_sample_ratio"],
                    log_path=payload["log_path"],
                )
                _json_response(self, {"state": ENV.export_state()})
                return
            if self.path == "/reset":
                ENV.reset()
                _json_response(self, {"state": ENV.export_state()})
                return
            if self.path == "/step":
                logger.info("POST /step on thread_id=%s", self._thread_id())
                _, reward, done, info = ENV.step(payload["action"], debug_payload=payload.get("debug_payload"))
                _json_response(self, {"state": ENV.export_state(), "reward": reward, "done": done, "info": info})
                return
            if self.path == "/render":
                logger.info("POST /render on thread_id=%s", self._thread_id())
                camera_views = payload.get("camera_views")
                images = ENV.render_images(camera_views)
                annotation_points = ENV.get_annotation_points(camera_views)
                encoded = {
                    key: base64.b64encode(ENV.encode_png_base64(image)).decode("utf-8")
                    for key, image in images.items()
                }
                _json_response(self, {"images": encoded, "annotation_points": annotation_points, "state": ENV.export_state()})
                return
            if self.path == "/close":
                ENV.close()
                ENV = None
                _json_response(self, {"ok": True})
                return
            _json_response(self, {"error": "not found"}, status=404)
        except Exception as exc:
            logger.exception("Request failed")
            _json_response(self, {"error": str(exc)}, status=500)

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    logger.info("Starting LIBERO env server (single-thread) on http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
