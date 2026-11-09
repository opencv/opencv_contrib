import atexit
import html
import threading
import time
import traceback
import uuid
from collections.abc import Mapping

import cv2 as cv


__all__ = [
    "LiveViewError",
    "LiveViewSession",
    "show",
    "camera",
    "sessions",
    "close_all",
]


class LiveViewError(RuntimeError):
    pass


_ACTIVE_SESSIONS = {}
_ACTIVE_LOCK = threading.Lock()


def _normalize_mode(mode):
    value = "auto" if mode is None else str(mode).lower()
    aliases = {
        "rtc": "webrtc",
        "web_rtc": "webrtc",
        "web-rtc": "webrtc",
        "jpg": "snapshot",
        "jpeg": "snapshot",
    }
    value = aliases.get(value, value)
    if value not in ("auto", "mjpeg", "webrtc", "snapshot"):
        raise ValueError("mode must be one of: auto, mjpeg, webrtc, snapshot")
    return value


def _normalize_fps(fps):
    if fps in (None, "auto"):
        return 30.0
    value = float(fps)
    if value <= 0:
        raise ValueError("fps must be positive")
    return value


def _normalize_timeout(value, name):
    if value is None or value is False:
        return None
    result = float(value)
    if result < 0:
        raise ValueError("{} must be non-negative".format(name))
    return result


def _normalize_public_url(url, public_host):
    if not public_host:
        return url
    return (url.replace("http://0.0.0.0", "http://{}".format(public_host))
               .replace("http://127.0.0.1", "http://{}".format(public_host))
               .replace("http://localhost", "http://{}".format(public_host)))


def _join_url(base, route):
    return "{}/{}".format(base.rstrip("/"), route.lstrip("/"))


def _channel_url(server_url, name, transport):
    if transport == "snapshot":
        return _join_url(server_url, "frame/{}.jpg".format(name))
    if transport == "webrtc":
        return _join_url(server_url, "webrtc/{}".format(name))
    return _join_url(server_url, "stream/{}.mjpeg".format(name))


def _create_server(host, port, mode):
    enable_webrtc = mode in ("auto", "webrtc")
    server = None
    try:
        server = cv.liveview.createServer(host, int(port), enable_webrtc)
        server.start()
        if mode == "snapshot":
            return server, "snapshot"
        return server, "webrtc" if enable_webrtc else "mjpeg"
    except Exception:
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass
        if mode != "auto" or not enable_webrtc:
            raise
        server = cv.liveview.createServer(host, int(port), False)
        server.start()
        return server, "mjpeg"


def _display_html(body):
    try:
        from IPython.display import HTML, display
    except Exception:
        return False
    display(HTML(body))
    return True


class LiveViewSession:
    def __init__(self, server, feed, name, mode, transport, fps, public_host,
                 source_id, on_close=None, close_when_idle=True,
                 idle_timeout=3.0, connect_timeout=30.0):
        self.name = name
        self.mode = mode
        self.transport = transport
        self.fps = fps
        self.public_host = public_host
        self.session_id = str(uuid.uuid4())
        self.state = "starting"
        self.diagnostics = []
        self._server = server
        self._feed = feed
        self._source_id = source_id
        self._on_close = on_close
        self._close_when_idle = bool(close_when_idle)
        self._idle_timeout = _normalize_timeout(idle_timeout, "idle_timeout")
        self._connect_timeout = _normalize_timeout(connect_timeout, "connect_timeout")
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._channels = set()
        self._error = None
        self._thread = threading.Thread(target=self._run,
                                        name="OpenCVLiveViewPump",
                                        daemon=True)
        self._thread.start()
        self._monitor_thread = None
        if self._close_when_idle:
            self._monitor_thread = threading.Thread(target=self._monitor_idle,
                                                    name="OpenCVLiveViewIdleMonitor",
                                                    daemon=True)
            self._monitor_thread.start()

    @property
    def id(self):
        return self.session_id

    @property
    def source(self):
        return self._source_id

    @property
    def server(self):
        return self._server

    @property
    def url(self):
        return _normalize_public_url(self._server.url(), self.public_host)

    @property
    def server_url(self):
        return self.url

    @property
    def channel_url(self):
        return self.url_for(self.name, self.transport)

    @property
    def mjpeg_url(self):
        return self.url_for(self.name, "mjpeg")

    @property
    def snapshot_url(self):
        return self.url_for(self.name, "snapshot")

    @property
    def webrtc_url(self):
        return self.url_for(self.name, "webrtc")

    @property
    def channels(self):
        with self._lock:
            return sorted(self._channels)

    @property
    def error(self):
        return self._error

    @property
    def urls(self):
        return {
            "server": self.server_url,
            "channel": self.channel_url,
            "mjpeg": self.mjpeg_url,
            "snapshot": self.snapshot_url,
            "webrtc": self.webrtc_url,
        }

    def url_for(self, name, mode=None):
        transport = self.transport if mode in (None, "auto") else _normalize_mode(mode)
        if transport == "auto":
            transport = self.transport
        return _normalize_public_url(_channel_url(self._server.url(), name, transport),
                                     self.public_host)

    def wait_ready(self, timeout=5.0):
        if not self._ready.wait(timeout):
            raise LiveViewError("LiveView feeder did not publish a frame before timeout")
        if self._error is not None:
            raise LiveViewError("LiveView feeder failed: {}".format(self._error))
        return self

    def html(self, name=None, mode=None, width=640, height=640):
        channel = name or self.name
        transport = self.transport if mode in (None, "auto") else _normalize_mode(mode)
        if transport == "snapshot":
            url = self.url_for(channel, "snapshot")
            body = '<img src="{}?t={}" width="{}">'.format(
                html.escape(url), time.time(), int(width))
        elif transport == "webrtc":
            url = self.url_for(channel, "webrtc")
            body = ('<p><a href="{0}" target="_blank">Open LiveView channel</a></p>'
                    '<iframe src="{0}" width="960" height="{1}"></iframe>').format(
                        html.escape(url), int(height))
        else:
            url = self.url_for(channel, "mjpeg")
            body = '<img src="{}" width="{}">'.format(html.escape(url), int(width))
        return "<h3>LiveView {}</h3>\n{}".format(html.escape(channel), body)

    def _repr_html_(self):
        return self.html()

    def show(self, name=None, mode=None, width=640, height=640):
        _display_html(self.html(name=name, mode=mode, width=width, height=height))
        if self.state == "ready":
            self.state = "displayed"
        return self

    def close(self, timeout=3.0):
        with self._lock:
            if self.state in ("closing", "closed"):
                return
            self.state = "closing"
        self._closed.set()
        self._ready.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
        if (self._monitor_thread is not None and self._monitor_thread.is_alive() and
                threading.current_thread() is not self._monitor_thread):
            self._monitor_thread.join(timeout)
        try:
            self._server.stop()
        finally:
            if self._on_close is not None:
                try:
                    self._on_close()
                except Exception as exc:
                    self.diagnostics.append("close callback failed: {}".format(exc))
            with self._lock:
                self.state = "closed"
            with _ACTIVE_LOCK:
                if _ACTIVE_SESSIONS.get(self._source_id) is self:
                    del _ACTIVE_SESSIONS[self._source_id]

    def release(self, timeout=3.0):
        self.close(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _run(self):
        interval = 1.0 / self.fps
        while not self._closed.is_set():
            started = time.monotonic()
            try:
                result = self._feed()
                if result is not None:
                    self._publish_result(result)
                    if not self._ready.is_set():
                        self._ready.set()
                        with self._lock:
                            self.state = "ready"
            except Exception as exc:
                self._error = exc
                self.diagnostics.append(traceback.format_exc())
                with self._lock:
                    self.state = "failed"
                self._ready.set()
                self._closed.set()
                break
            elapsed = time.monotonic() - started
            delay = interval - elapsed
            if delay > 0:
                self._closed.wait(delay)

    def _publish_result(self, result):
        if isinstance(result, Mapping):
            if not result:
                return
            for channel, frame in result.items():
                self._publish_one(str(channel), frame)
            return
        self._publish_one(self.name, result)

    def _publish_one(self, name, frame):
        if frame is None:
            return
        self._server.publish(name, frame)
        with self._lock:
            self._channels.add(name)

    def _viewer_count(self):
        if hasattr(self._server, "viewerCount"):
            try:
                return int(self._server.viewerCount(self.name))
            except TypeError:
                return int(self._server.viewerCount())
        return 0

    def _has_ever_had_viewer(self):
        if hasattr(self._server, "hasEverHadViewer"):
            return bool(self._server.hasEverHadViewer())
        if hasattr(self._server, "lastViewerConnectedTick"):
            return int(self._server.lastViewerConnectedTick()) > 0
        return self._viewer_count() > 0

    def _monitor_idle(self):
        if not self._ready.wait(self._connect_timeout):
            if not self._closed.is_set():
                self.diagnostics.append("closed before first frame timeout")
                self.close()
            return

        ready_at = time.monotonic()
        idle_since = None
        while not self._closed.is_set():
            if self._error is not None:
                return
            try:
                viewers = self._viewer_count()
                ever_connected = self._has_ever_had_viewer()
            except Exception as exc:
                self.diagnostics.append("viewer monitor failed: {}".format(exc))
                self._closed.wait(0.25)
                continue

            now = time.monotonic()
            if viewers > 0:
                idle_since = None
            elif ever_connected:
                if idle_since is None:
                    idle_since = now
                if self._idle_timeout is not None and now - idle_since >= self._idle_timeout:
                    self.diagnostics.append("closed after LiveView viewer idle timeout")
                    self.close()
                    return
            elif self._connect_timeout is not None and now - ready_at >= self._connect_timeout:
                self.diagnostics.append("closed after LiveView viewer connect timeout")
                self.close()
                return

            self._closed.wait(0.1)


def _replace_existing(source_id, replace, reuse, mode):
    with _ACTIVE_LOCK:
        existing = _ACTIVE_SESSIONS.get(source_id)
    if existing is None or existing.state == "closed":
        return None
    if reuse and existing.mode == mode:
        return existing
    if not replace:
        raise LiveViewError("LiveView session already exists for {}".format(source_id))
    existing.close()
    return None


def _start_session(feed, name, mode, fps, host, public_host, port, replace,
                   reuse, display, wait, timeout, source_id, on_close,
                   close_when_idle, idle_timeout, connect_timeout):
    if not callable(feed):
        raise TypeError("feed must be callable")
    resolved_mode = _normalize_mode(mode)
    source = source_id or "frames:{}".format(name)
    existing = _replace_existing(source, replace=replace, reuse=reuse, mode=resolved_mode)
    if existing is not None:
        if display:
            existing.show(name=name, mode=resolved_mode)
        return existing

    server, transport = _create_server(host, port, resolved_mode)
    session = LiveViewSession(server=server,
                              feed=feed,
                              name=str(name),
                              mode=resolved_mode,
                              transport=transport,
                              fps=_normalize_fps(fps),
                              public_host=public_host,
                              source_id=source,
                              on_close=on_close,
                              close_when_idle=close_when_idle,
                              idle_timeout=idle_timeout,
                              connect_timeout=connect_timeout)
    with _ACTIVE_LOCK:
        _ACTIVE_SESSIONS[source] = session
    if wait:
        try:
            session.wait_ready(timeout)
        except Exception:
            session.close()
            raise
    if display:
        session.show(name=name, mode=resolved_mode)
    return session


def show(feed, name="view", mode="auto", fps="auto", host="127.0.0.1",
         public_host=None, port=0, replace=True, reuse=False, show=True,
         wait=True, timeout=5.0, source_id=None, on_close=None,
         close_when_idle=True, idle_timeout=3.0, connect_timeout=30.0):
    return _start_session(feed=feed,
                          name=name,
                          mode=mode,
                          fps=fps,
                          host=host,
                          public_host=public_host,
                          port=port,
                          replace=replace,
                          reuse=reuse,
                          display=show,
                          wait=wait,
                          timeout=timeout,
                          source_id=source_id,
                          on_close=on_close,
                          close_when_idle=close_when_idle,
                          idle_timeout=idle_timeout,
                          connect_timeout=connect_timeout)


def camera(device=0, transform=None, name="camera", mode="auto", fps="auto",
           host="127.0.0.1", public_host=None, port=0, replace=True,
           reuse=False, show=True, wait=True, timeout=5.0,
           close_when_idle=True, idle_timeout=3.0, connect_timeout=30.0):
    cap = cv.VideoCapture(device)
    if not cap.isOpened():
        cap.release()
        raise LiveViewError("cannot open camera: {}".format(device))

    def feed():
        ok, frame = cap.read()
        if not ok:
            return None
        if transform is not None:
            return transform(frame)
        return frame

    return _start_session(feed=feed,
                          name=name,
                          mode=mode,
                          fps=fps,
                          host=host,
                          public_host=public_host,
                          port=port,
                          replace=replace,
                          reuse=reuse,
                          display=show,
                          wait=wait,
                          timeout=timeout,
                          source_id="camera:{}".format(device),
                          on_close=cap.release,
                          close_when_idle=close_when_idle,
                          idle_timeout=idle_timeout,
                          connect_timeout=connect_timeout)


def sessions():
    with _ACTIVE_LOCK:
        return dict(_ACTIVE_SESSIONS)


def close_all():
    for session in list(sessions().values()):
        session.close()


atexit.register(close_all)
