# OpenGL Rendering {#v4d_render_opengl}

@prev_tutorial{v4d_vector_graphics_and_fb}
@next_tutorial{v4d_font_rendering}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

## Render a blue screen using OpenGL
This example simply paints the screen blue using OpenGL without shaders for brevity. One important detail of this example is that states are being preserved between invocations of a context type (in this case the ```gl``` context).

\htmlinclude "../samples/example_v4d_render_opengl.html"

@include samples/render_opengl.cpp


