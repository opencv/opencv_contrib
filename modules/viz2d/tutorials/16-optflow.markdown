# Optflow-Demo {#viz2d_optflow}

@prev_tutorial{viz2d_pedestrian}
@next_tutorial{viz2d_beauty}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

Optical flow visualization on top of a video. Uses background subtraction (OpenCV/OpenCL) to isolate areas with motion, detects features to track (OpenCV/OpenCL), calculates the optical flow (OpenCV/OpenCL), uses nanovg for rendering (OpenGL) and post-processes the video (OpenCL). Decodes/encodes on the GPU (VAAPI).

@youtube{uPYTRPZnocw}

@include samples/cpp/optflow/optflow-demo.cpp



