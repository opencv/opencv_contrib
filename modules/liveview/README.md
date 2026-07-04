# LiveView

The `liveview` module is a planned OpenCV contrib module for publishing named
live visual outputs from running OpenCV applications.

The intended direction is a small public API centered on:

```cpp
cv::liveview::Server view;
view.publish("camera", frame);
view.publish("edges", edges);
```

This skeleton intentionally contains no transport implementation yet. The first
development target is the core channel model, followed by HTTP snapshot/MJPEG
viewing and an optional WebRTC transport for robotics and ROS2 demos.
