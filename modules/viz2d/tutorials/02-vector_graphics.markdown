# Render vector graphics using Viz2D {#viz2d_vector_graphics}

## Vector graphics
Through the nvg context javascript-canvas-like rendering is possible.

@include samples/cpp/vector_graphics.cpp

![The result](doc/vector_graphics.png)

## Vector graphics and framebuffer manipulation
The framebuffer can be accessed directly to manipulate data created in other contexts. In this case vector graphics is rendered to the framebuffer through NanoVG and then blurred using an ```fb`` context.

@include samples/cpp/vector_graphics_and_fb.cpp

![The result](doc/vector_graphics_and_fb.png)
