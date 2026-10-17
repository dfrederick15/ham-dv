"""Browser viewer: MJPEG stream + status JSON served over HTTP (works on any computer on the LAN)."""
from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>HamDV receiver</title>
<style>
body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:16px}
#v{width:100%;max-width:720px;image-rendering:auto;background:#000;border:1px solid #333}
.row{display:flex;gap:24px;flex-wrap:wrap;margin-top:12px}
.k{color:#888;font-size:12px}.v{font-size:20px}
#bar{height:8px;background:#333;width:200px;border-radius:4px;overflow:hidden}#fill{height:100%;background:#4c8;width:0}
</style></head><body>
<h2>HamDV &mdash; live video over the voice repeater</h2>
<img id="v" src="/stream.mjpg">
<div class="row">
<div><div class="k">Station</div><div class="v" id="cs">&mdash;</div></div>
<div><div class="k">State</div><div class="v" id="st">&mdash;</div></div>
<div><div class="k">SNR</div><div class="v" id="snr">&mdash;</div><div id="bar"><div id="fill"></div></div></div>
<div><div class="k">Mode</div><div class="v" id="mode">&mdash;</div></div>
<div><div class="k">Video</div><div class="v" id="vid">&mdash;</div></div>
<div><div class="k">Blocks OK</div><div class="v" id="blk">&mdash;</div></div>
<div><div class="k">Frames</div><div class="v" id="fr">&mdash;</div></div>
<div><div class="k">Drift</div><div class="v" id="sro">&mdash;</div></div>
</div>
<div class="row"><div><div class="k">Text</div><div class="v" id="txt">&mdash;</div></div></div>
<script>
async function poll(){try{const r=await fetch('/status.json');const s=await r.json();
document.getElementById('cs').textContent=s.callsign||'—';
document.getElementById('st').textContent=s.state;
document.getElementById('snr').textContent=isFinite(s.snr_db)?s.snr_db.toFixed(1)+' dB':'—';
document.getElementById('fill').style.width=Math.max(0,Math.min(100,(s.snr_db||0)*3))+'%';
document.getElementById('mode').textContent=s.mode||'—';
document.getElementById('vid').textContent=s.video||'—';
document.getElementById('blk').textContent=s.blocks_ok+' / '+s.blocks_total;
document.getElementById('fr').textContent=s.frames;
document.getElementById('sro').textContent=(s.sro_ppm||0).toFixed(0)+' ppm';
document.getElementById('txt').textContent=s.text||'—';}catch(e){}
setTimeout(poll,500);}poll();
</script></body></html>"""


class Viewer:
    def __init__(self, port: int = 8080, upscale: int = 4):
        self.port = port
        self.upscale = upscale
        self.jpeg = self._encode(np.full((72 * upscale, 96 * upscale, 3), 32, np.uint8))
        self.status = {}
        self.cond = threading.Condition()
        self.version = 0
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/status.json"):
                    body = json.dumps(viewer.status, default=lambda o: None).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/frame.jpg"):
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(viewer.jpeg)))
                    self.end_headers()
                    self.wfile.write(viewer.jpeg)
                elif self.path.startswith("/stream.mjpg"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = -1
                    try:
                        while True:
                            with viewer.cond:
                                viewer.cond.wait_for(lambda: viewer.version != last, timeout=1.0)
                                last = viewer.version
                                data = viewer.jpeg
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                             str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self.send_response(404)
                    self.end_headers()

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _encode(self, rgb: np.ndarray) -> bytes:
        from PIL import Image
        im = Image.fromarray(rgb)
        if self.upscale > 1:
            im = im.resize((rgb.shape[1] * self.upscale, rgb.shape[0] * self.upscale), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def update_frame(self, rgb: np.ndarray):
        data = self._encode(rgb)
        with self.cond:
            self.jpeg = data
            self.version += 1
            self.cond.notify_all()

    def update_status(self, status: dict):
        self.status = dict(status)

    def close(self):
        self.server.shutdown()
