# Optflow-Demo {#v4d_optflow}

@prev_tutorial{v4d_pedestrian}
@next_tutorial{v4d_beauty}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

Optical flow visualization on top of a video. Uses background subtraction (OpenCV/OpenCL) to isolate areas with motion, detects features to track (OpenCV/OpenCL), calculates the optical flow (OpenCV/OpenCL), uses nanovg for rendering (OpenGL) and post-processes the video (OpenCL). Decodes/encodes on the GPU (VAAPI).

@youtube{k-NA6R9SBvo}

@include samples/optflow-demo.cpp



