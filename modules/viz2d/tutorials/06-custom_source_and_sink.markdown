# Custom Source and Sink {#viz2d_custom_source_and_sink}

@prev_tutorial{viz2d_video_editing}
@next_tutorial{viz2d_font_with_gui}

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

## Reading and writing to Viz2D using custom Sources and Sinks
In the previous tutorial we used a default video source and a video sink to stream a video through Viz2D which can be manipulated using OpenGL, NanoVG or OpenCV. In this example we are creating a custom Source that generates rainbow frames. For each time the Source is invoked the frame is colored a slightly different color. Additionally the custom Sink saves individual images instead of a video.

@include samples/cpp/custom_source_and_sink.cpp

![The result](doc/custom_source_and_sink.png)

