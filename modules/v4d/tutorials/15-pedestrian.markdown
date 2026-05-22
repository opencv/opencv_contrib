# Pedestrian-Demo {#v4d_pedestrian}

@prev_tutorial{v4d_font}
@next_tutorial{v4d_optflow}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

Pedestrian detection using HOG with a linear SVM, non-maximal suppression and tracking using KCF. Uses nanovg for rendering (OpenGL), detects using a linear SVM (OpenCV), filters resuls using NMS and tracks using KCF.

\htmlinclude "../samples/example_v4d_pedestrian-demo.html"

@include samples/pedestrian-demo.cpp


