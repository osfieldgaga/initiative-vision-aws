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

Public demo (LIVE_TOKEN_SECRET set): behind Caddy on demo.initiativelabs.cc, every request needs a
token signed with that secret -- "<expiry unix time>.<hex HMAC-SHA256(secret, expiry)>". The
website mints one after its password check and opens /?token=...; the page then keeps it in an
httpOnly cookie, so the streams and /metrics are covered too. Without the secret (the default)
nothing is checked: localhost and the SSH tunnel work as before. For testing:

    LIVE_TOKEN_SECRET=... python live_view.py token [hours]      # prints a token
"""
import hashlib
import hmac
import json
import os
import sys
import threading
import time
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
    <p>signs</p>
    <table id="signs"></table>
    <p>objects, nearest first</p>
    <table id="objs"></table>
  </div>
</div>
<script>
const history = [];
let lastSeq = -1, lastDone = false, lastSession = null;
const fmt = v => (typeof v === "number") ? v.toFixed(2) : (v == null ? "n/a" : String(v));
const fmtM = v => v == null ? "n/a" : fmt(v) + " m";

function drawPlot() {
  const c = document.getElementById("plot"), g = c.getContext("2d");
  const range = 2.0;  // +/- metres shown
  const y = v => c.height / 2 - (Math.max(-range, Math.min(range, v)) / range) * (c.height / 2);
  g.clearRect(0, 0, c.width, c.height);
  g.strokeStyle = "#444";
  for (const v of [0, __DEP__, -__DEP__]) { g.beginPath(); g.moveTo(0, y(v)); g.lineTo(c.width, y(v)); g.stroke(); }
  history.forEach((p, i) => {
    if (p.offset == null) return;  // lane(s) not detected, nothing to plot
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
        <tr><td>offset</td><td>${fmtM(x.offset_m)}</td></tr>
        <tr><td>L / R off</td><td>${fmtM(x.left_off_m)} / ${fmtM(x.right_off_m)}</td></tr>
        <tr><td>road width</td><td>${fmtM(x.road_width_m)}</td></tr>
        <tr><td>valid</td><td class="${x.lane_valid ? "ok" : "bad"}">${x.lane_valid}</td></tr>
        <tr><td>status</td><td class="${cls}">${x.departure_text}</td></tr>
        ${m.skipped ? `<tr><td>skipped</td><td class="bad">${m.skipped} frames (too old, replay to fill in)</td></tr>` : ""}
        ${m.backlog != null ? `<tr><td>backlog</td><td class="${m.backlog > 3 ? "bad" : "ok"}">${m.backlog} frames queued</td></tr>` : ""}
        ${m.timing && m.timing.pi_to_s3_s != null ? `<tr><td>lag breakdown</td><td></td></tr>
        ${m.timing.capture_to_write_s != null ? `<tr><td>&nbsp; capture → write</td><td>${fmt(m.timing.capture_to_write_s)} s</td></tr>` : ""}
        <tr><td>&nbsp; Pi → S3</td><td>${fmt(m.timing.pi_to_s3_s)} s</td></tr>
        <tr><td>&nbsp; S3 → queue</td><td>${fmt(m.timing.s3_to_queue_s)} s</td></tr>
        <tr><td>&nbsp; in queue</td><td>${fmt(m.timing.queue_s + m.timing.batch_wait_s)} s</td></tr>
        <tr><td>&nbsp; processing</td><td>${fmt(m.timing.total_s)} s</td></tr>` : ""}
        ${m.timing ? `<tr><td>per frame</td><td>${fmt(m.timing.total_s)} s</td></tr>
        <tr><td>&nbsp; download</td><td>${fmt(m.timing.download_s)} s</td></tr>
        <tr><td>&nbsp; inference</td><td>${fmt(m.timing.infer_s)} s</td></tr>
        ${m.timing.signs_s != null ? `<tr><td>&nbsp; signs</td><td>${fmt(m.timing.signs_s)} s</td></tr>` : ""}
        <tr><td>&nbsp; post-proc</td><td>${fmt(m.timing.post_s)} s</td></tr>
        <tr><td>&nbsp; outputs</td><td>${fmt(m.timing.output_s)} s</td></tr>` : ""}
        <tr><td>run</td><td>${m.done ? "finished" : "running"}</td></tr>`;
      const signs = x.signs || [];
      document.getElementById("signs").innerHTML = signs.length
        ? signs.map(s => `<tr><td class="${s.type === "stop" ? "bad" : "ok"}">` +
                         `${s.type === "stop" ? "STOP" : "SPEED LIMIT " + s.value}</td>` +
                         `<td class="na">score ${fmt(s.score)}, read "${s.text}"</td></tr>`).join("")
        : `<tr><td class="na">none</td></tr>`;
      const objs = x.objects || [];
      document.getElementById("objs").innerHTML = objs.length
        ? `<tr class="na"><td>#</td><td>label</td><td>score</td><td>box (x1,y1,x2,y2)</td></tr>` +
          objs.map((o, i) => `<tr><td>${i + 1}</td><td>${o.label}</td><td>${fmt(o.score)}</td>` +
                            `<td class="na">${o.box.map(v => Math.round(v)).join(", ")}</td></tr>`).join("")
        : `<tr><td class="na">none</td></tr>`;
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


TOKEN_COOKIE = "lv_token"
DENIED_PAGE = (b"<!doctype html><meta charset=utf-8><title>Initiative Vision</title>"
               b"<body style='background:#111;color:#ddd;font-family:sans-serif;margin:40px'>"
               b"<h2>Initiative Vision live demo</h2><p>Open it from "
               b"<a style='color:#6af' href='https://initiativelabs.cc/ivision-demo'>initiativelabs.cc/ivision-demo</a>.</p>")


def make_token(secret, hours=2.0, now=None):
    expiry = int((now or time.time()) + hours * 3600)
    return f"{expiry}.{hmac.new(secret.encode(), str(expiry).encode(), hashlib.sha256).hexdigest()}"


def token_valid(secret, token, now=None):
    """True if token was signed with secret and hasn't expired."""
    try:
        expiry, sig = token.split(".", 1)
        expected = hmac.new(secret.encode(), expiry.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected) and int(expiry) > (now or time.time())
    except (AttributeError, ValueError):
        return False


class LiveView:
    """Holds the latest frames/metrics and serves them to any connected browser."""

    def __init__(self, host="127.0.0.1", port=8080, jpeg_quality=80, departure_threshold=0.4,
                 token_secret=None):
        self.host = host
        self.port = port
        self.token_secret = token_secret  # None = no access check (localhost / SSH tunnel)
        self.last_request = 0.0           # time.time() of the last request, for the idle auto-stop
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
            def _token(self):
                """(token from ?token= or the cookie, came_from_query)."""
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                for part in query.split("&"):
                    if part.startswith("token="):
                        return part[len("token="):], True
                for part in self.headers.get("Cookie", "").split(";"):
                    name, _, value = part.strip().partition("=")
                    if name == TOKEN_COOKIE:
                        return value, False
                return None, False

            def do_GET(self):
                path = self.path.split("?")[0]
                view.last_request = time.time()
                if view.token_secret:
                    token, from_query = self._token()
                    if not token_valid(view.token_secret, token):
                        self._send(401, "text/html; charset=utf-8", DENIED_PAGE)
                        return
                    if from_query:  # keep it in a cookie and drop it from the address bar
                        self.send_response(303)
                        self.send_header("Location", path)
                        self.send_header("Set-Cookie", f"{TOKEN_COOKIE}={token}; Path=/; HttpOnly; Secure; "
                                                       f"SameSite=Lax; Max-Age={int(token.split('.')[0]) - int(time.time())}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
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


if __name__ == "__main__" and sys.argv[1:2] == ["token"]:
    secret = os.environ.get("LIVE_TOKEN_SECRET")
    if not secret:
        raise SystemExit("set LIVE_TOKEN_SECRET")
    print(make_token(secret, float(sys.argv[2]) if len(sys.argv) > 2 else 2.0))
