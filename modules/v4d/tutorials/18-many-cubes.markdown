# Many_Cubes-Demo {#v4d_many_cubes}

@prev_tutorial{v4d_cube}
@next_tutorial{v4d_video}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

Renders 10 rainbow cubes on blueish background using OpenGL and applies a glow effect using OpenCV. The special thing about this demo is the each cube is renderered in a different OpenGL-context with its independent OpenGL-state. That said, this for sure isn't the most efficient way to draw multiple copies but serves well to demonstrate how independent OpenGL contexts/states can be used.

\htmlinclude "../samples/example_v4d_many_cubes-demo.html"

@include samples/many_cubes-demo.cpp

