# LiveView

The `liveview` module publishes named live visual outputs from running OpenCV
applications.

The public API is centered on:

```cpp
cv::liveview::Server view("127.0.0.1", 0);
view.start();
view.publish("camera", frame);
view.publish("edges", edges);
```

The module provides HTTP snapshot and MJPEG routes. WebRTC is available when the
optional WebRTC build dependencies are enabled.
