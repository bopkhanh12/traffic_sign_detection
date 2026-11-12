#!/usr/bin/env python3
"""Small localhost dashboard and image demo for the TT100K YOLO model."""

from __future__ import annotations

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "training_output" / "yolov8n"
HTML = ROOT / "web" / "index.html"
MAX_UPLOAD = 20 * 1024 * 1024
_model = None
_model_lock = threading.Lock()


def training_status() -> dict:
    metrics_path = OUTPUT / "results.csv"
    metrics = []
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            metrics = list(csv.DictReader(stream))
    weights = OUTPUT / "weights"
    available_weights = [name for name in ("best.pt", "last.pt") if (weights / name).is_file()]
    args_path = OUTPUT / "args.yaml"
    return {
        "run": "YOLOv8n",
        "status": "Đã có checkpoint" if available_weights else "Đang chờ epoch đầu tiên",
        "completed_epochs": len(metrics),
        "target_epochs": 10,
        "train_images": 5434,
        "validation_images": 600,
        "classes": 173,
        "weights": available_weights,
        "metrics": metrics[-1] if metrics else {},
        "updated": metrics_path.stat().st_mtime if metrics_path.is_file() else (args_path.stat().st_mtime if args_path.exists() else None),
    }


def detect(image_bytes: bytes) -> dict:
    global _model
    import io

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO

            weights = OUTPUT / "weights" / "best.pt"
            if not weights.is_file():
                weights = OUTPUT / "weights" / "last.pt"
            if not weights.is_file():
                raise RuntimeError("Model chưa có checkpoint. Hãy đợi ít nhất một epoch hoàn tất.")
            _model = YOLO(str(weights))
        results = _model.predict(source=image, verbose=False)
    result = results[0]
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            boxes.append({"class": result.names.get(class_id, str(class_id)), "confidence": round(float(box.conf.item()), 4), "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()]})
    return {"detections": boxes, "count": len(boxes)}


class Handler(BaseHTTPRequestHandler):
    server_version = "TT100KLocal/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            body = json.dumps(training_status(), ensure_ascii=False).encode()
            self._send(200, body, "application/json; charset=utf-8")
        elif path == "/" and HTML.is_file():
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_UPLOAD:
                raise ValueError("Ảnh phải nhỏ hơn 20 MB.")
            result = detect(self.rfile.read(length))
            self._send(200, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except (ValueError, OSError) as error:
            self._send(400, json.dumps({"error": str(error)}, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except RuntimeError as error:
            self._send(503, json.dumps({"error": str(error)}, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except Exception as error:
            self._send(500, json.dumps({"error": f"Không thể dự đoán: {error}"}, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


if __name__ == "__main__":
    address = ("127.0.0.1", 8001)
    print(f"TT100K localhost: http://{address[0]}:{address[1]}", flush=True)
    ThreadingHTTPServer(address, Handler).serve_forever()
