# LiveView

The `liveview` module publishes named live visual outputs from running OpenCV
applications.

The C++ core API is centered on publishing frames:

```cpp
cv::liveview::Server view("127.0.0.1", 0);
view.start();
view.publish("camera", frame);
view.publish("edges", edges);
```

The module provides HTTP snapshot and MJPEG routes. WebRTC is available when the
optional WebRTC build dependencies are enabled.

The primary Python workflow is feeder-based. The user supplies the frame
production function and LiveView owns the display session:

```python
import cv2 as cv

def feed():
    ok, frame = cap.read()
    if not ok:
        return None
    overlay = run_pipeline(frame)
    return overlay

view = cv.liveview.show(feed, name="overlay", mode="auto")
```

The feeder may also return multiple named views from the same pipeline tick:

```python
def feed():
    ok, frame = cap.read()
    if not ok:
        return None
    mask = segment(frame)
    return {"camera": frame, "mask": mask}

view = cv.liveview.show(feed)
```

`cv.liveview.camera(...)` is a convenience wrapper over the same feeder/session
machinery. It is not the core abstraction.
