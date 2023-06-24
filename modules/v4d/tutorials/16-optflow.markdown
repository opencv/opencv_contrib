# Optflow-Demo {#v4d_optflow}

@prev_tutorial{v4d_pedestrian}
@next_tutorial{v4d_beauty}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

Optical flow visualization on top of a video. Uses background subtraction (OpenCV) to isolate areas with motion, detects features to track (OpenCV), calculates the optical flow (OpenCV), uses nanovg for rendering (OpenGL) and post-processes the video (OpenCV).

\htmlinclude "../samples/example_v4d_optflow-demo.html"

@include samples/optflow-demo.cpp



