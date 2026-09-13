"""Local, shared, dynamically batched inference for Qwen Heading Gaussian."""
from __future__ import annotations

import argparse
from concurrent.futures import Future
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np
from PIL import Image

from oat.starvla_heading.evaluation import decode_image, file_hash, validate_actions, validate_metadata


@dataclass
class Pending:
    payload: dict
    future: Future


class BatchingPredictor:
    """One inference thread owns the GPU and batches concurrent simulator calls."""
    def __init__(self, predict_many, max_batch_size=4, max_wait_ms=10, max_pending=64):
        if max_batch_size < 1 or max_wait_ms < 0 or max_pending < max_batch_size:
            raise ValueError("Invalid inference batch/queue settings")
        self.predict_many = predict_many
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self.pending = queue.Queue(maxsize=max_pending)
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="policy-inference", daemon=True)
        self.thread.start()

    def submit(self, payload):
        if self.closed:
            raise RuntimeError("Inference server is shutting down")
        future = Future()
        self.pending.put_nowait(Pending(payload, future))
        return future

    def _run(self):
        while True:
            first = self.pending.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.max_wait
            stop_after_batch = False
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.pending.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                batch.append(item)
            try:
                outputs = self.predict_many([item.payload for item in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("Model returned a different number of batch results")
                for item, output in zip(batch, outputs):
                    item.future.set_result(output)
            except Exception as error:
                traceback.print_exc()
                for item in batch:
                    item.future.set_exception(error)
            if stop_after_batch:
                return

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.thread.join(timeout=60)


def validate_request(payload, metadata):
    if not isinstance(payload, dict) or payload.get("protocol_version") != 1:
        raise ValueError("Unsupported inference protocol")
    if payload.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
        raise ValueError("Request checkpoint differs from the frozen server checkpoint")
    if not isinstance(payload.get("request_id"), str) or not payload["request_id"]:
        raise ValueError("A request_id is required")
    if not isinstance(payload.get("seed"), int) or isinstance(payload["seed"], bool) or not 0 <= payload["seed"] < 2 ** 64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not isinstance(payload.get("language"), str) or not payload["language"].strip():
        raise ValueError("An actual language instruction is required")
    state = np.asarray(payload.get("state"), dtype=np.float32)
    if state.shape != (2, 9) or not np.isfinite(state).all():
        raise ValueError("Expected a finite (2, 9) raw state window")
    images = payload.get("images")
    if not isinstance(images, list) or len(images) != 4:
        raise ValueError("Expected four temporal-major, camera-major PNG images")
    size = metadata["image_size"]
    for image in images:
        if decode_image(image).shape != (size, size, 3):
            raise ValueError(f"Expected RGB images of shape ({size}, {size}, 3)")
    return payload


def request_examples(payloads):
    return [{"image": [Image.fromarray(decode_image(image)) for image in payload["images"]],
             "lang": payload["language"], "state": np.asarray(payload["state"], dtype=np.float32)}
            for payload in payloads]


class ModelPredictor:
    def __init__(self, checkpoint, device="cuda:0"):
        import torch
        from oat.starvla_heading.model import QwenHeadingGaussian
        checkpoint = Path(checkpoint).resolve()
        weights = checkpoint / "weights.pt"
        self.metadata = json.loads((checkpoint / "metadata.json").read_text())
        validate_metadata(self.metadata, benchmark="libero")
        if file_hash(weights) != self.metadata["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint weights do not match metadata SHA-256")
        config = json.loads((checkpoint / "config.json").read_text())
        if self.metadata.get("config_sha256") and file_hash(checkpoint / "config.json") != self.metadata["config_sha256"]:
            raise RuntimeError("Checkpoint config does not match metadata SHA-256")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("The full Qwen inference server requires a CUDA device")
        self.model = QwenHeadingGaussian(config, statistics=config.get("statistics"), pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.model.qwen.gradient_checkpointing_disable()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def __call__(self, payloads):
        import torch
        examples = request_examples(payloads)
        # Separate generators fix source randomness per request, independently
        # of worker scheduling and how the dynamic batch is assembled. BF16
        # rounding can still vary slightly with batch padding/kernel selection.
        noise = torch.stack([
            torch.randn((16, 7), device=self.device, dtype=torch.float32,
                        generator=torch.Generator(device=self.device).manual_seed(payload["seed"]))
            for payload in payloads
        ])
        with torch.inference_mode():
            actions = self.model.predict_action(examples, noise=noise)["actions"]
        if len(actions) != len(payloads):
            raise RuntimeError("Unexpected policy batch dimension")
        return [{"protocol_version": 1, "checkpoint_sha256": self.metadata["checkpoint_sha256"],
                 "request_id": payload["request_id"], "actions": validate_actions(action).tolist()}
                for payload, action in zip(payloads, actions)]


def make_handler(metadata, predictor, request_timeout=300):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Per-control-cycle access logs would dominate full Plus evaluation.
            if len(args) > 1 and str(args[1]) not in ("200",):
                super().log_message(format, *args)

        def reply(self, status, value):
            encoded = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/metadata":
                self.reply(200, metadata)
            elif self.path == "/health":
                self.reply(200, {"ready": not predictor.closed, "checkpoint_sha256": metadata["checkpoint_sha256"]})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/predict":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError("Request body must be between 1 byte and 16 MiB")
                payload = json.loads(self.rfile.read(size))
                validate_request(payload, metadata)
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, {"error": str(error)})
                return
            try:
                result = predictor.submit(payload).result(timeout=request_timeout)
                self.reply(200, result)
            except queue.Full:
                self.reply(503, {"error": "Inference queue is full"})
            except Exception as error:
                self.reply(500, {"error": f"{type(error).__name__}: {error}"})

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Frozen export directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-wait-ms", type=float, default=10)
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    model = ModelPredictor(args.checkpoint, args.device)
    predictor = BatchingPredictor(model, args.max_batch_size, args.max_wait_ms)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model.metadata, predictor, args.request_timeout))
    print(json.dumps({"ready": True, "address": f"http://{args.host}:{args.port}",
                      "parameters": model.parameter_count, "max_batch_size": args.max_batch_size,
                      "checkpoint_sha256": model.metadata["checkpoint_sha256"]}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        predictor.close()
    return 0
