# V4D {#v4d}

[TOC]

|    |    |
| -: | :- |
| Original author | Amir Hassan (kallaballa) <amir@viel-zu.org> |
| Compatibility | OpenCV >= 4.7 |

# What is V4D?
V4D offers a way of writing graphical (on- and offscreen) high performance applications with OpenCV. It is light-weight and unencumbered by QT or GTK licenses. It features vector graphics using [NanoVG](https://github.com/inniyah/nanovg) a GUI based on [ImGUI](https://github.com/ocornut/imgui) and (on supported systems) OpenCL/OpenGL and OpenCL/VAAPI interoperability. It should be included in [OpenCV-contrib](https://github.com/opencv/opencv_contrib) once it is ready.

# Why V4D?
Please refer to the online demos in the \ref v4d_tutorials and \ref v4d_demos section to see at a glance what it can do for you. **But note**: The online demos are slower than native builds and are sometimes missing features. If you want full performance (including hardware acceleration) you should really create a native build and test it.

* **OpenGL**: Easy access to OpenGL.
* **GUI**: Simple yet powerful user interfaces through ImGui.
* **Vector graphics**: Elegant and fast vector graphics through NanoVG.
* **Font rendering**: Loading of fonts and sophisticated rendering options.
* **Video pipeline**: Through a simple source/sink system videos can be efficently read, displayed, edited and saved.
* **Hardware acceleration**: Transparent hardware acceleration usage where possible. (e.g. CL-GL interop, VAAPI and CL-VAAPI interop). Actually it is possible to write programs that run almost entirely on the GPU, given driver-features are available.
* **No more highgui** with it's heavy dependencies, licenses and limitations.
* **\ref v4d_webassembly_support**.

# Design Notes
* V4D is not thread safe. Though it is possible to have several V4D objects in one or more threads and synchronize them using ```V4D::makeCurrent()```. This is a limitation of GLFW3/EGL. That said, OpenCV algorithms are multi-threaded as usual.
* V4D uses InputArray/OutputArray/InputOutputArray which gives you the option to work with Mat, std::vector and UMat. Anyway, you should prefer to use UMat whenever possible to automatically use hardware capabilities where available.
* Access to different subsystems (opengl, framebuffer, nanovg and imgui) is provided through "contexts". A context is simply a function that takes a functor, sets up the subsystem, executes the functor and tears-down the subsystem.
* ```V4D::run``` is not a context. It is an abstraction of a run loop that takes a functor and runs until the application terminates or the functor returns false. This is necessary for portability reasons.
* Contexts ***may not*** be nested.

For example, to create an OpenGL context and set the GL viewport:
@code{.cpp}
//Creates a V4D object for on screen rendering
Ptr<V4D> v4d = V4D::make(Size(WIDTH, HEIGHT), "GL viewport");

//Takes care of OpenGL states in the background
v4d->gl([](const Size sz) {
    glViewPort(0, 0, sz.width, sz.height);
});
@endcode

# GPU Support
* Intel Gen 8+ (Tested: Gen 11 + Gen 13) is supported best
* NVIDIA Ada Lovelace (Tested: GTX 4070 Ti) with proprietary drivers (535.104.05) and CUDA toolkit (12.2) works but video writing is very slow, unless: you change the codec to H264 or you create a gstreamer sink using nvenc.
* AMD: never tested

# Requirements
* C++20 (at the moment)
* OpenGL 3.2 Core (optionally Compat)/OpenGL ES 3.0/WebGL2

# Optional requirements
* Support for OpenCL 1.2
* Support for cl_khr_gl_sharing and cl_intel_va_api_media_sharing OpenCL extensions.

# Dependencies
* [My OpenCV 4.x fork](https://github.com/kallaballa/opencv) (It works with mainline OpenCV 4.x as well, but will miss some features)
* GLEW
* GLFW3
* NanoVG (included as a sub-repo)
* ImGui (included as a sub-repo)

# Optional: Dependencies for demos
* (At the time of writing) If you want CL-GL interop on a recent Intel Platform you might need to build [compute-runtime](https://github.com/intel/compute-runtime). The first version of compute-runtime shipping CL-GL interop is **23.13.26032**

# Tutorials {#v4d_tutorials}
The tutorials are designed to be read one after the other to give you a good overview over the key concepts of V4D. After that you can move on to the demos.

* \ref v4d_display_image_pipeline
* \ref v4d_display_image_fb
* \ref v4d_display_image_nvg
* \ref v4d_vector_graphics
* \ref v4d_vector_graphics_and_fb
* \ref v4d_render_opengl
* \ref v4d_font_rendering
* \ref v4d_video_editing
* \ref v4d_custom_source_and_sink
* \ref v4d_font_with_gui

# Demos {#v4d_demos}
The goal of the demos is to show how to use V4D to the fullest. Also they show how to use V4D to create programs that run mostly (the part the matters) on the GPU (when driver capabilities allow). They are also a good starting point for your own applications because they touch many key aspects and algorithms of OpenCV.

* \ref v4d_cube
* \ref v4d_many_cubes
* \ref v4d_video
* \ref v4d_nanovg
* \ref v4d_shader
* \ref v4d_font
* \ref v4d_pedestrian
* \ref v4d_optflow
* \ref v4d_beauty

# Instructions for Ubuntu 22.04.2 LTS
You need to build OpenCV with V4D

## Install required packages

```bash
apt install vainfo clinfo libqt5opengl5-dev freeglut3-dev ocl-icd-opencl-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libpostproc-dev libswresample-dev libswscale-dev libglfw3-dev libstb-dev libglew-dev cmake make git-core build-essential opencl-clhpp-headers pkg-config zlib1g-dev doxygen libxinerama-dev libxcursor-dev libxi-dev libva-dev yt-dlp wget intel-opencl-icd ca-certificates
```
## Optional: Install if you want to build your own packages
```bash
apt install ubuntu-dev-tools dh-cmake gdebi
```
## EITHER: Minimal V4D build without examples and demos

```bash
git clone --branch GCV https://github.com/kallaballa/opencv.git
git clone https://github.com/kallaballa/V4D.git
mkdir opencv/build
cd opencv/build
 cmake -DCMAKE_BUILD_TYPE=Release -DCV_TRACE=OFF -DBUILD_SHARED_LIBS=ON -DWITH_OPENGL=ON -DOPENCV_ENABLE_EGL=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DWITH_1394=OFF -DWITH_ADE=OFF -DWITH_VTK=OFF -DWITH_EIGEN=OFF -DWITH_GTK=OFF -DWITH_GTK_2_X=OFF -DWITH_IPP=OFF -DWITH_JASPER=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_OPENVX=OFF -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF-DWITH_TBB=OFF -DWITH_TIFF=OFF -DWITH_OPENCL=ON -DWITH_OPENCL_SVM=ON -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_GPHOTO2=OFF -DWITH_LAPACK=OFF -DWITH_ITT=OFF -DWITH_QUIRC=ON -DBUILD_ZLIB=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_calib3d=OFF -DBUIlD_opencv_ccalib=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_features2d=OFF -DBUILD_opencv_flann=OFF -DBUILD_opencv_gapi=OFF -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=OFF -DBUILD_opencv_imgcodecs=ON -DBUILD_opencv_shape=OFF -DBUILD_opencv_videoio=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_highgui=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_stitching=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_js=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_alphamat=OFF -DBUILD_opencv_aruco=OFF -DBUILD_opencv_barcode=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_cudaarithm=OFF -DBUILD_opencv_cudabgsegm=OFF -DBUILD_opencv_cudacodec=OFF -DBUILD_opencv_cudafeatures2d=OFF -DBUILD_opencv_cudafilters=OFF -DBUILD_opencv_cudaimgproc=OFF -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_cudaobjdetect=OFF -DBUILD_opencv_cudaoptflow=OFF -DBUILD_opencv_cudastereo=OFF -DBUILD_opencv_cudawarping=OFF -DBUILD_opencv_cudev=OFF -DBUILD_opencv_cvv=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DBUILD_opencv_dnn_superres=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=OFF -DBUILD_opencv_freetype=OFF -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_hfs=OFF -DBUILD_opencv_img_hash=OFF -DBUILD_opencv_intensity_transform=OFF -DBUILD_opencv_julia=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_mcc=OFF -DBUILD_opencv_optflow=OFF -DBUILD_opencv_ovis=OFF -DBUILD_opencv_phase_unwrapping=OFF -DBUILD_opencv_plot=OFF -DBUILD_opencv_quality=OFF -DBUILD_opencv_rapid=OFF -DBUILD_opencv_README.md=OFF -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_sfm=OFF -DBUILD_opencv_shape=OFF -DBUILD_opencv_stereo=OFF -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=OFF -DBUILD_opencv_tracking=OFF -DBUILD_opencv_videostab=OFF -DBUILD_opencv_viz=OFF -DBUILD_opencv_wechat_qrcode=OFF -DBUILD_opencv_xfeatures2d=OFF -DBUILD_opencv_ximgproc=OFF -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PACKAGE=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF -DWITH_PTHREADS_PF=ON -DCV_ENABLE_INTRINSICS=ON -DOPENCV_EXTRA_MODULES_PATH=../../V4D/modules/ ..
make -j8
sudo make install
```

## OR: Full V4D build with examples, demos and debian packages (takes a while)

```bash
git clone --branch GCV https://github.com/kallaballa/opencv.git
git clone https://github.com/kallaballa/V4D.git
mkdir opencv/build
cd opencv/build
cmake -DINSTALL_BIN_EXAMPLES=ON -DOPENCV_CUSTOM_PACKAGE_INFO=ON -DCPACK_PACKAGE_VERSION_MAJOR=4 -DCPACK_PACKAGE_VERSION_MINOR=8 -DCPACK_PACKAGE_VERSION_PATCH=0 -DCPACK_PACKAGE_VERSION=4:8.0-kallaballa -DCMAKE_BUILD_TYPE=Release -DCPACK_PACKAGE_CONTACT="amir@viel-zu.org" -DOPENCV_GENERATE_PKGCONFIG=ON -DCPACK_PACKAGE_VENDOR=kallaballa -DCPACK_DEBIAN_PACKAGE_DEPENDS="libqt5opengl5,freeglut3,ocl-icd-libopencl1,libavcodec58,libavdevice58,libavfilter7,libavformat58,libavutil56,libpostproc55,libswresample3,libswscale5,libglfw3,libstb0,libglew2.2,zlib1g,libxinerama1,libxcursor1,libxi6,libva2,intel-opencl-icd,ca-certificates" -DINSTALL_CREATE_DISTRIB=ON -DCPACK_BINARY_DEB=ON -DCV_TRACE=OFF -DBUILD_SHARED_LIBS=ON -DWITH_OPENGL=ON -DOPENCV_ENABLE_EGL=ON -DOPENCV_ENABLE_GLX=ON -DOPENCV_FFMPEG_ENABLE_LIBAVDEVICE=ON -DWITH_QT=ON -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON -DWITH_VA=ON -DWITH_VA_INTEL=ON -DWITH_1394=OFF -DWITH_ADE=OFF -DWITH_VTK=OFF -DWITH_EIGEN=OFF -DWITH_GTK=OFF -DWITH_GTK_2_X=OFF -DWITH_IPP=OFF -DWITH_JASPER=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_OPENVX=OFF -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF-DWITH_TBB=OFF -DWITH_TIFF=OFF -DWITH_OPENCL=ON -DWITH_OPENCL_SVM=ON -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_GPHOTO2=OFF -DWITH_LAPACK=OFF -DWITH_ITT=OFF -DWITH_QUIRC=ON -DBUILD_ZLIB=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_calib3d=ON -DBUIlD_opencv_ccalib=OFF -DBUILD_opencv_dnn=ON -DBUILD_opencv_features2d=ON -DBUILD_opencv_flann=ON -DBUILD_opencv_gapi=OFF -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=ON -DBUILD_opencv_imgcodecs=ON -DBUILD_opencv_shape=OFF -DBUILD_opencv_videoio=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_highgui=ON -DBUILD_opencv_superres=OFF -DBUILD_opencv_stitching=ON -DBUILD_opencv_java=OFF -DBUILD_opencv_js=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_alphamat=OFF -DBUILD_opencv_aruco=OFF -DBUILD_opencv_barcode=OFF -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF -DBUILD_opencv_ccalib=ON -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_cudaarithm=OFF -DBUILD_opencv_cudabgsegm=OFF -DBUILD_opencv_cudacodec=OFF -DBUILD_opencv_cudafeatures2d=OFF -DBUILD_opencv_cudafilters=OFF -DBUILD_opencv_cudaimgproc=OFF -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_cudaobjdetect=OFF -DBUILD_opencv_cudaoptflow=OFF -DBUILD_opencv_cudastereo=OFF -DBUILD_opencv_cudawarping=OFF -DBUILD_opencv_cudev=OFF -DBUILD_opencv_cvv=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn_objdetect=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DBUILD_opencv_dnn_superres=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=ON -DBUILD_opencv_freetype=OFF -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_hfs=OFF -DBUILD_opencv_img_hash=OFF -DBUILD_opencv_intensity_transform=OFF -DBUILD_opencv_julia=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_mcc=OFF -DBUILD_opencv_optflow=ON -DBUILD_opencv_ovis=OFF -DBUILD_opencv_phase_unwrapping=OFF -DBUILD_opencv_plot=ON -DBUILD_opencv_quality=OFF -DBUILD_opencv_rapid=OFF -DBUILD_opencv_README.md=OFF -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_sfm=OFF -DBUILD_opencv_shape=OFF -DBUILD_opencv_stereo=OFF -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_superres=OFF -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=OFF -DBUILD_opencv_tracking=ON -DBUILD_opencv_videostab=OFF -DBUILD_opencv_viz=OFF -DBUILD_opencv_wechat_qrcode=OFF -DBUILD_opencv_xfeatures2d=OFF -DBUILD_opencv_ximgproc=ON -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF -DBUILD_EXAMPLES=ON -DBUILD_PACKAGE=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=ON -DWITH_PTHREADS_PF=ON -DCV_ENABLE_INTRINSICS=ON -DOPENCV_EXTRA_MODULES_PATH=../../V4D/modules/ ..
make -j8
sudo make install
```
## Build debian packages
```bash
cpack DEB
```

## Download the example videos
```bash
# big buck bunny video
wget -O bunny.webm https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.1080p.vp9.webm
# dance video
yt-dlp -o dance.webm "https://www.youtube.com/watch?v=yg6LZtNeO_8"
# kristen video
yt-dlp -o kristen.webm "https://www.youtube.com/watch?v=hUAT8Jm_dvw&t=11s"
```

## Run the examples and demos
```
# Examples
bin/example_v4d_display_image
bin/example_v4d_display_image_fb
bin/example_v4d_vector_graphics
bin/example_v4d_vector_graphics_and_fb
bin/example_v4d_render_opengl
bin/example_v4d_font_rendering
bin/example_v4d_video_editing
bin/example_v4d_custom_source_and_sink
bin/example_v4d_font_with_gui

# Demos
bin/example_v4d_cube-demo
bin/example_v4d_many_cubes-demo
bin/example_v4d_video-demo bunny.webm
bin/example_v4d_nanovg-demo bunny.webm
bin/example_v4d_shader-demo bunny.webm
bin/example_v4d_font-demo
bin/example_v4d_pedestrian-demo dance.webm
bin/example_v4d_optflow-demo dance.webm
bin/example_v4d_beauty-demo kristen.webm

```

# Attribution
* The author of the bunny video is the **Blender Foundation** ([Original video](https://www.bigbuckbunny.org)).
* The author of the dance video is **GNI Dance Company** ([Original video](https://www.youtube.com/watch?v=yg6LZtNeO_8)).
* The author of the video used in the beauty-demo video is **Kristen Leanne** ([Original video](https://www.youtube.com/watch?v=hUAT8Jm_dvw&t=11s)).
* The author of cxxpool is **Copyright (c) 2022 Christian Blume**: ([LICENSE](https://github.com/bloomen/cxxpool/blob/master/LICENSE))
* The author of the roboto font family is **Google Inc.** ([LICENSE](https://github.com/googlefonts/roboto/blob/main/LICENSE))
