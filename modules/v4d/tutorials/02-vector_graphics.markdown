# Render vector graphics {#v4d_vector_graphics}

@prev_tutorial{v4d_display_image}
@next_tutorial{v4d_render_opengl}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

## Vector graphics
Through the nvg context javascript-canvas-like rendering is possible.

@include samples/vector_graphics.cpp

![The result](doc/vector_graphics.png)

## Vector graphics and framebuffer manipulation
The framebuffer can be accessed directly to manipulate data created in other contexts. In this case vector graphics is rendered to the framebuffer through NanoVG and then blurred using an ```fb`` context.

@include samples/vector_graphics_and_fb.cpp

![The result](doc/vector_graphics_and_fb.png)
