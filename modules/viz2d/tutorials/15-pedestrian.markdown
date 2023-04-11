# Pedestrian-Demo {#viz2d_pedestrian}

@prev_tutorial{viz2d_font}
@next_tutorial{viz2d_optflow}

|    |    |
| -: | :- |
| Original author | Amir Hassan |
| Compatibility | OpenCV >= 4.7 |

Pedestrian detection using HOG with a linear SVM, non-maximal suppression and tracking using KCF. Uses nanovg for rendering (OpenGL), detects using a linear SVM (OpenCV/OpenCL), filters resuls using NMS (CPU) and tracks using KCF (CPU). Decodes/encodes on the GPU (VAAPI).

@youtube{PIUQIG6Qlrw}

@include samples/cpp/pedestrian/pedestrian-demo.cpp


