from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
TOP_K = 10


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        payload = json.dumps({
            "chunk_size": CHUNK_SIZE,
            "overlap_ratio": OVERLAP_RATIO,
            "top_k": TOP_K,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass
