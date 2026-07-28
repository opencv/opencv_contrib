# LiveView

LiveView publishes named visual outputs from a running OpenCV process so they
can be inspected in a browser or notebook while the user iterates on frame
processing code.

The module is intended for fast development feedback, not as a general-purpose
streaming service. A typical loop is:

```text
sensor or existing frame source
  -> OpenCV / ML / robotics processing
  -> visual feedback frame
  -> LiveView browser or notebook display
```

## Python Workflow

The primary Python API is feeder based. The user owns the frame source and
processing logic. LiveView owns the display session, server, feeder pump,
transport URLs, notebook HTML, replacement, and idle shutdown.

```python
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("could not open camera")

def feed():
    ok, frame = cap.read()
    if not ok:
        return None
    overlay = process(frame)
    return overlay

view = cv.liveview.show(
    feed,
    name="overlay",
    mode="auto",
    host="0.0.0.0",
    public_host="192.168.1.10",
    replace=True,
)

view
```

`feed()` should return one OpenCV frame or `None`. Returning `None` keeps the
session alive and waits for a later frame.

For related views from one processing tick, return a dictionary:

```python
def feed():
    ok, frame = cap.read()
    if not ok:
        return None
    mask = segment(frame)
    overlay = draw_overlay(frame, mask)
    return {
        "camera": frame,
        "mask": mask,
        "overlay": overlay,
    }

view = cv.liveview.show(feed, name="overlay")
```

## Notebook Iteration

For notebook use, keep long-lived external resources such as `VideoCapture`
outside the cell that is repeatedly edited. Rerun only the cell that defines
`feed()` and calls `cv.liveview.show(..., replace=True)`.

Setup cell:

```python
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("could not open camera")
```

Rerunnable cell:

```python
def feed():
    ok, frame = cap.read()
    if not ok:
        return None
    return process(frame)

view = cv.liveview.show(feed, name="overlay", replace=True)
view
```

`replace=True` closes the previous LiveView session with the same `name` before
the new session is installed. This disowns the previous feeder loop and server.
It does not release `cap`, because `cap` is owned by user code in this
workflow. This distinction is intentional: LiveView must not destroy a stable
camera object that the notebook continues to use across processing iterations.

If LiveView creates the source, LiveView releases it. If user code creates the
source, user code owns it.

## Source Ownership

`cv.liveview.show(feed, ...)`:

- owns the LiveView session;
- owns the feeder thread;
- owns the HTTP/WebRTC server instance used by that session;
- owns replacement and idle close policy;
- does not infer or release arbitrary objects referenced by `feed()`.

Use `on_close` when a user-owned resource should be released with the session:

```python
cap = cv.VideoCapture(0)

def feed():
    ok, frame = cap.read()
    return frame if ok else None

view = cv.liveview.show(feed, on_close=cap.release)
```

`cv.liveview.camera(...)` is a convenience wrapper for simple camera display.
It creates the `VideoCapture` internally, so it releases that capture when the
session closes:

```python
view = cv.liveview.camera(0, mode="auto")
```

Use `camera()` for quick inspection. Use `show(feed)` for real processing
pipelines.

## Session Lifecycle

`cv.liveview.show()` returns a `LiveViewSession`.

Useful properties and methods:

```python
view.server_url
view.channel_url
view.mjpeg_url
view.snapshot_url
view.webrtc_url
view.urls
view.channels
view.html()
view.show()
view.close()
view.release()
```

Sessions are context managers:

```python
with cv.liveview.show(feed, show=False) as view:
    print(view.channel_url)
```

Global helpers:

```python
cv.liveview.sessions()
cv.liveview.close_all()
```

`close()` is idempotent. Closing a session stops the feeder thread, stops the
server, tears down active transports, runs explicit `on_close`, and removes the
session from the active session registry.

## Replacement

By default, `source_id` is derived from the displayed name:

```text
frames:<name>
```

Starting another session with the same name and `replace=True` closes the
previous session before registering the new one:

```python
view = cv.liveview.show(feed, name="rectangles", replace=True)
```

Use an explicit `source_id` only when several names should map to the same
logical session, or when the default `frames:<name>` identity is not enough.

## Idle Close

LiveView tracks long-lived viewers:

- MJPEG stream clients;
- WebRTC sessions.

Snapshot requests count as activity, but not as active viewers.

Python sessions can close themselves when no viewer connects or after all
viewers disconnect:

```python
view = cv.liveview.show(
    feed,
    close_when_idle=True,
    connect_timeout=30.0,
    idle_timeout=3.0,
)
```

Defaults are selected for notebook and browser iteration:

- `close_when_idle=True`
- `connect_timeout=30.0`
- `idle_timeout=3.0`

Disable idle close for long-running publishers:

```python
view = cv.liveview.show(feed, close_when_idle=False)
```

## Transports

`mode` selects the display transport:

```python
mode="auto"
mode="mjpeg"
mode="webrtc"
mode="snapshot"
```

`auto` prefers WebRTC when the server is built with WebRTC support, otherwise
uses MJPEG. If WebRTC startup fails in `auto`, the Python wrapper falls back to
MJPEG.

## C++ API

The C++ API is centered on explicit publishing:

```cpp
cv::Ptr<cv::liveview::Server> view =
    cv::liveview::createServer("127.0.0.1", 0);

view->start();
view->publish("camera", frame);
view->publish("edges", edges);
std::cout << view->channelUrl("camera") << std::endl;
```

The server exposes:

- `/` index page;
- `/healthz`;
- `/channels.json`;
- `/frame/<name>.jpg`;
- `/stream/<name>.mjpeg`;
- `/webrtc/<name>` when WebRTC is enabled.

Viewer accounting is available through the public server facade:

```cpp
server->viewerCount();
server->viewerCount("camera");
server->hasViewers();
server->hasEverHadViewer();
server->lastViewerConnectedTick();
server->lastViewerDisconnectedTick();
server->lastViewerActivityTick();
```

These counters are used by Python session idle-close policy and are useful for
tests and diagnostics.

## Build Options

HTTP backend:

```text
LIVEVIEW_HTTP_BACKEND=AUTO|CIVETWEB|BOOST
```

Video encoder:

```text
LIVEVIEW_VIDEO_ENCODER=AUTO|OFF|FFMPEG
```

WebRTC:

```text
LIVEVIEW_WEBRTC=AUTO|OFF|ON
```

WebRTC requires the video encoder path and GStreamer WebRTC development
packages. When WebRTC is disabled, snapshot and MJPEG remain available.

## Tests

The module test target exercises the server, route handling, channel storage,
encoding paths, transport negotiation, viewer accounting, and Python session
helpers.

```text
modules/liveview/test/
```

These tests use local synthetic frames and localhost HTTP requests.
