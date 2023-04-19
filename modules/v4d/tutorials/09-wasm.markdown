# WebAssembly Support {#v4d_webassembly_support}

[TOC]

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

# What is WebAssembly?
It is possible to compile C++ (but also other languages) for the browser. The resulting binaries contain usually WebAssembly (WASM) which the browser knows to execute.

# So what makes it special for OpenCV and V4D?
For OpenCV there has been the possibility to run code in the browser for a while using [OpenCV.js](https://docs.opencv.org/4.x/d0/d84/tutorial_js_usage.html). But OpenCV.js merely offers the OpenCV APIs and visualization and GUI has to be done by other means (e.g. HTML5 Canvas). That is where V4D steps in because it has been written with WebAssembly in mind and cleverly uses [OpenGL](https://en.wikipedia.org/wiki/OpenGL) in a fashion that translates well to [WebGL](https://en.wikipedia.org/wiki/WebGL). V4D enables you to write graphical applications that run native as well as in the browser.


