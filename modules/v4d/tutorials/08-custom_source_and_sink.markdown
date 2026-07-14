# Custom Source and Sink {#v4d_custom_source_and_sink}

@prev_tutorial{v4d_video_editing}
@next_tutorial{v4d_font_with_gui}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

## Reading and writing to V4D using custom sources and sinks
In the previous tutorial we used a default video source and a video sink to stream a video through V4D which can be manipulated using OpenGL, NanoVG or OpenCV. In this example we are creating a custom source that generates rainbow frames. For each time the source is invoked the frame is colored a slightly different color. Additionally the custom sink saves individual images instead of a video (only in native builds).

\htmlinclude "../samples/example_v4d_custom_source_and_sink.html"

@include samples/custom_source_and_sink.cpp



