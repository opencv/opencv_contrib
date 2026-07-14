# Display an image using the video pipeline {#v4d_display_image_pipeline}

@prev_tutorial{v4d}
@next_tutorial{v4d_display_image_fb}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

## Using the video pipeline
Actually there are several ways to display an image using V4D. The most convenient way is to use the video pipeline to feed an image to V4D. That has the advantage that the image is automatically resized (preserving aspect ratio) to framebuffer size and color converted (the framebuffer is BGRA while video frames are expected to be BGR).

\htmlinclude "../samples/example_v4d_display_image.html"

@include samples/display_image.cpp
