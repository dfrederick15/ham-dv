"""Browser viewer: MJPEG stream + status JSON served over HTTP (works on any computer on the LAN)."""
from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .interp import interpolate, global_field

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
    """MJPEG/HTTP viewer with a motion-interpolated playout clock.

    Real frames arrive in bursts (one superframe at a time).  They are queued
    with nominal presentation times and played back `delay` seconds later, so
    that the *next* real frame is normally already known; a render thread at
    `smooth_fps` warps and blends the two bracketing frames (see interp.py).
    """

    def __init__(self, port: int = 8080, upscale: int = 4, smooth_fps: float = 30.0, delay: float = 1.7):
        self.port = port
        self.upscale = upscale
        self.jpeg = self._encode(np.full((72 * upscale, 96 * upscale, 3), 32, np.uint8))
        self.status = {}
        self.cond = threading.Condition()
        self.version = 0
        self.smooth_fps = smooth_fps
        self.delay = delay
        self.lock = threading.Lock()
        self.queue = []            # list of (present_time, rgb, field)
        self.clock_origin = None   # (wall time of first frame, its nominal media time)
        self.rendered = 0
        if smooth_fps > 0:
            threading.Thread(target=self._render_loop, daemon=True).start()
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

    def push_frame(self, rgb: np.ndarray, fps: float, field=None):
        """Queue a real frame for smooth playout (falls back to immediate display when smooth_fps == 0)."""
        if self.smooth_fps <= 0:
            self.update_frame(rgb)
            return
        now = time.time()
        with self.lock:
            if self.clock_origin is None or (self.queue and now - self.queue[-1][0] > 4.0) or not self.queue and now - self._last_present > 4.0:
                # (re)start the playout clock: present this frame `delay` seconds from now
                self.clock_origin = (now + self.delay, 0.0)
                self.queue = []
                self.media_pos = 0
            t_present = self.clock_origin[0] + self.media_pos / fps
            self.media_pos += 1
            # never schedule into the past (late burst): shift the clock instead
            if t_present < now:
                self.clock_origin = (now + self.delay - (self.media_pos - 1) / fps, 0.0)
                t_present = now + self.delay
            if field is None and self.queue:
                field = global_field(self.queue[-1][1], rgb)
            self.queue.append((t_present, rgb, field))
            self._last_present = t_present

    _last_present = 0.0
    media_pos = 0

    def _render_loop(self):
        period = 1.0 / self.smooth_fps
        prev = None
        while True:
            t0 = time.time()
            frame = None
            with self.lock:
                q = self.queue
                # drop frames that are two or more behind the display time
                while len(q) >= 2 and q[1][0] <= t0:
                    q.pop(0)
                if q:
                    if len(q) >= 2 and q[0][0] <= t0 < q[1][0]:
                        a, b = q[0], q[1]
                        t = (t0 - a[0]) / max(b[0] - a[0], 1e-3)
                        frame = interpolate(a[1], b[1], t, b[2])
                    elif q[0][0] <= t0:
                        frame = q[0][1]
            if frame is not None and frame is not prev:
                self.update_frame(frame)
                self.rendered += 1
                prev = frame
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

    def update_status(self, status: dict):
        self.status = dict(status)

    def close(self):
        self.server.shutdown()
