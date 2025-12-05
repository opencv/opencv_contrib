# Display an image using NanoVG {#v4d_display_image_nvg}

@prev_tutorial{v4d_display_image_fb}
@next_tutorial{v4d_vector_graphics}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

## Using NanoVG to display images
Instead of feeding to the video pipeline or doing a direct framebuffer access we can use NanoVG to display an image. It is not as convinient as the other methods but it is very fast and flexible.

\htmlinclude "../samples/example_v4d_display_image_nvg.html"

@include samples/display_image_nvg.cpp
