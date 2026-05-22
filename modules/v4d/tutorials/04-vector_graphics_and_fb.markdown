# Render vector graphics and manipulate the framebuffer {#v4d_vector_graphics_and_fb}

@prev_tutorial{v4d_vector_graphics}
@next_tutorial{v4d_render_opengl}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

## Vector graphics and framebuffer manipulation
The framebuffer can be accessed directly to manipulate data created in other contexts. In this case vector graphics is rendered to the framebuffer through NanoVG and then blurred using an ```fb`` context.

\htmlinclude "../samples/example_v4d_vector_graphics_and_fb.html"

@include samples/vector_graphics_and_fb.cpp
