# Display an image using direct framebuffer access {#v4d_display_image_fb}

@prev_tutorial{v4d_display_image_pipeline}
@next_tutorial{v4d_display_image_nvg}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

## Using direct framebuffer access
Instead of feeding to the video pipeline we can request the framebuffer in a ```fb``` context and copy the image to it. But first we must manually resize and color convert the image.

\htmlinclude "../samples/example_v4d_display_image_fb.html"

@include samples/display_image_fb.cpp
