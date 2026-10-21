"""Live view of the lane pipeline over HTTP, meant to be reached through an SSH tunnel.

The server binds to 127.0.0.1 only, so nothing is exposed to the internet and no
security-group port is needed. From your laptop:

    ssh -L 8080:localhost:8080 ec2-user@<server>
    then open http://localhost:8080

Endpoints:
    /             page with both streams + live metrics
    /overlay.mjpg camera overlay (MJPEG stream)
    /bev.mjpg     bird's-eye view (MJPEG stream)
    /metrics      latest metrics as JSON
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2 as cv


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>LLD live</title>
<style>
  body { background:#111; color:#ddd; font-family:monospace; margin:16px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
  img { border:1px solid #333; max-width:100%; }
  table { border-collapse:collapse; }
  td { padding:2px 10px 2px 0; }
  .bad { color:#f55; } .ok { color:#5d5; } .na { color:#888; }
  canvas { background:#1a1a1a; border:1px solid #333; }
</style></head>
<body>
<div class="row">
  <img src="/overlay.mjpg" width="640" height="640">
  <img src="/bev.mjpg" width="640" height="640">
  <div>
    <table id="m"></table>
    <p>offset (m), last 200 frames</p>
    <canvas id="plot" width="400" height="160"></canvas>
  </div>
</div>
<script>
const history = [];
let lastSeq = -1, lastDone = false, lastSession = null;
const fmt = v => (typeof v === "number") ? v.toFixed(2) : String(v);

function drawPlot() {
  const c = document.getElementById("plot"), g = c.getContext("2d");
  const range = 2.0;  // +/- metres shown
  const y = v => c.height / 2 - (Math.max(-range, Math.min(range, v)) / range) * (c.height / 2);
  g.clearRect(0, 0, c.width, c.height);
  g.strokeStyle = "#444";
  for (const v of [0, __DEP__, -__DEP__]) { g.beginPath(); g.moveTo(0, y(v)); g.lineTo(c.width, y(v)); g.stroke(); }
  history.forEach((p, i) => {
    g.fillStyle = !p.valid ? "#666" : (p.departure ? "#f55" : "#5d5");
    g.fillRect(i * c.width / 200, y(p.offset) - 1, 2, 2);
  });
}

async function poll() {
  try {
    const m = await (await fetch("/metrics")).json();
    if ((m.seq !== lastSeq || m.done !== lastDone) && m.metrics) {
      const isNewFrame = m.seq !== lastSeq;
      lastSeq = m.seq; lastDone = m.done;
      const x = m.metrics;
      const cls = !x.lane_valid ? "na" : (x.departure ? "bad" : "ok");
      document.getElementById("m").innerHTML = `
        <tr><td>session</td><td>${m.session ?? ""}</td></tr>
        <tr><td>frame</td><td>${m.index}${m.total != null ? " / " + m.total : ""}</td></tr>
        <tr><td>file</td><td>${m.frame}</td></tr>
        ${m.lag_s != null ? `<tr><td>lag</td><td class="${m.lag_s > 5 ? "bad" : "ok"}">${fmt(m.lag_s)} s</td></tr>` : ""}
        <tr><td>offset</td><td>${fmt(x.offset_m)} m</td></tr>
        <tr><td>L / R off</td><td>${fmt(x.left_off_m)} / ${fmt(x.right_off_m)} m</td></tr>
        <tr><td>road width</td><td>${fmt(x.road_width_m)} m</td></tr>
        <tr><td>valid</td><td class="${x.lane_valid ? "ok" : "bad"}">${x.lane_valid}</td></tr>
        <tr><td>status</td><td class="${cls}">${x.departure_text}</td></tr>
        <tr><td>run</td><td>${m.done ? "finished" : "running"}</td></tr>`;
      if (m.session !== lastSession) { history.length = 0; lastSession = m.session; }
      if (isNewFrame) {
        history.push({offset: x.offset_m, valid: x.lane_valid, departure: x.departure});
        if (history.length > 200) history.shift();
        drawPlot();
      }
    }
  } catch (e) { /* server gone or tunnel closed; keep trying */ }
  setTimeout(poll, 300);
}
poll();
</script>
</body></html>
"""


class LiveView:
    """Holds the latest frames/metrics and serves them to any connected browser."""

    def __init__(self, host="127.0.0.1", port=8080, jpeg_quality=80, departure_threshold=0.4):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.page = PAGE.replace("__DEP__", str(departure_threshold)).encode()

        self._cond = threading.Condition()
        self._seq = 0
        self._jpegs = {"overlay": None, "bev": None}
        self._info = {"seq": 0, "metrics": None, "done": False}
        self._stopped = False
        self._server = None
        self._thread = None

    def start(self):
        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", view.page)
                elif path == "/metrics":
                    with view._cond:
                        body = json.dumps(view._info).encode()
                    self._send(200, "application/json", body)
                elif path in ("/overlay.mjpg", "/bev.mjpg"):
                    view._stream(self, path[1:].split(".")[0])
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # keep the per-frame console output readable

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"live view on http://{self.host}:{self.port}  "
              f"(from your laptop: ssh -L {self.port}:localhost:{self.port} <user>@<server>)")
        return self

    def _stream(self, handler, name):
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        last = -1
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._seq != last or self._stopped, timeout=5)
                    if self._stopped:
                        return
                    if self._seq == last:
                        continue
                    last, jpg = self._seq, self._jpegs[name]
                if jpg is None:
                    continue
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser tab closed / tunnel dropped

    def update(self, overlay, bev, metrics, **info):
        """Publish a new frame. info: extra fields for /metrics (index, total, frame, ...)."""
        params = [cv.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok1, overlay_jpg = cv.imencode(".jpg", overlay, params)
        ok2, bev_jpg = cv.imencode(".jpg", bev, params)
        with self._cond:
            self._seq += 1
            if ok1:
                self._jpegs["overlay"] = overlay_jpg.tobytes()
            if ok2:
                self._jpegs["bev"] = bev_jpg.tobytes()
            self._info = {"seq": self._seq, "metrics": metrics, "done": False, **info}
            self._cond.notify_all()

    def finish(self):
        """Mark the run as finished; the last frame stays visible while the server is up."""
        with self._cond:
            self._info = {**self._info, "done": True}

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
