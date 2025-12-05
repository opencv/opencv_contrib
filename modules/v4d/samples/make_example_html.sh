#!/bin/bash

set -e

./example_v4d_capture.sh "Beauty Demo" beauty-demo > example_v4d_beauty-demo.html
./example_v4d_nocapture.sh "Cube Demo" cube-demo > example_v4d_cube-demo.html
./example_v4d_nocapture.sh "Custom Source and Sink" custom_source_and_sink > example_v4d_custom_source_and_sink.html
./example_v4d_nocapture.sh "Display an Image through the FB Context" display_image_fb > example_v4d_display_image_fb.html
./example_v4d_nocapture.sh "Display an Image through the Video-Pipeline" display_image > example_v4d_display_image.html
./example_v4d_nocapture.sh "Display an Image through NanoVG" display_image_nvg > example_v4d_display_image_nvg.html
./example_v4d_nocapture.sh "Font Demo" font-demo > example_v4d_font-demo.html
./example_v4d_nocapture.sh "Font rendering with Form-based GUI" font_with_gui > example_v4d_font_with_gui.html
./example_v4d_nocapture.sh "Font rendering" font_rendering > example_v4d_font_rendering.html
./example_v4d_nocapture.sh "Many Cubes Demo" many_cubes-demo > example_v4d_many_cubes-demo.html
./example_v4d_capture.sh "NanoVG Demo" nanovg-demo > example_v4d_nanovg-demo.html
./example_v4d_capture.sh "Sparse Optical Flow Demo" optflow-demo > example_v4d_optflow-demo.html
./example_v4d_capture.sh "Pedestrian Demo" pedestrian-demo > example_v4d_pedestrian-demo.html
./example_v4d_nocapture.sh "Render OpenGL Blue Screen" render_opengl > example_v4d_render_opengl.html
./example_v4d_capture.sh "Mandelbrot Shader Demo" shader-demo > example_v4d_shader-demo.html
./example_v4d_nocapture.sh "Vector Graphics and Frambuffer access" vector_graphics_and_fb > example_v4d_vector_graphics_and_fb.html
./example_v4d_nocapture.sh "Vector Graphics" vector_graphics > example_v4d_vector_graphics.html
./example_v4d_capture.sh "Video Demo" video-demo > example_v4d_video-demo.html
./example_v4d_capture.sh "Video Editing" video_editing > example_v4d_video_editing.html
