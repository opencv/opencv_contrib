Introduction to Julia OpenCV Binding {#tutorial_julia}
=======================================

OpenCV
------

OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. Distributed under permissive license, OpenCV makes it easy for businesses to utilize and modify the code.

The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery and establish markers to overlay it with augmented reality, etc. OpenCV has more than 47 thousand people of user community and estimated number of downloads exceeding 18 million. The library is used extensively in companies, research groups and by governmental bodies.

Julia
-------------
Julia is a high-performance, high-level, and dynamic programming language that specializes in tasks relateted numerical, and scientefic computing. However, It can also be used for general programming with GUI and web programming. Julia can be considered as a combination of rapid interpreter style prototyping capability of Python with the raw speed of C because of its special "just-ahead-of-time" compilation.

Inspite of all this, Julia severely lacks in a lot of traditional computer vision and image processing algorithms. This also hampers the usage of Julia in any pipeline that requires computer vision. The OpenCV bindings for Julia aims to solve this problem.

The Bindings
-----------------------
The OpenCV bindings for Julia are created automatically using Python scripts at configure time and then installed with the Julia package manager on the system. These bindings cover most of the important functionality present in the core, imgproc, imgcodecs, highgui, videio, calib3d, and dnn modules. These bindings depend on CxxWrap.jl and the process for usage and compilation is explained in detail below. The Bindings have been tested on Ubuntu and Mac. Windows might work but is not officially tested and supported right now.

The generation process and the method by which the binding works are similar to the Python bindings. The only major difference is that CxxWrap.jl does not support optional arguments. As a consequence, it's necessary to define the optional arguments in Julia code which adds a lot of additional complexity.

How To Install The Bindings
-----------------------
The easiest and recommended way to install the bindings is using Julia's inbuilt package manager. OpenCV is available as a registered package for Julia and is supported on all major platforms and architectures. The following steps checked for correctness on Julia v1.6.1

TO install start the Julia REPL. Hit `]` and then type `add OpenCV`.

```bash
$ julia
...
julia> ]
pkg> add OpenCV
```

How To Build The Bindings
-----------------------
Before you can build the bindings, make sure that you know how to build OpenCV with all the functionality you require and the contrib modules except the Julia Bindings. As mentioned before, the Julia bindings are not officially supported on Windows right now and a better alternative would be to try it with WSL/WSL2.

The pre-requisites for the Julia Bindings are:
 - [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl)
 - [libcxxwrap-julia](https://github.com/JuliaInterop/libcxxwrap-julia)
 - Python
 - Julia

It is recommended to use Julia 1.4+ and the latest versions of CxxWrap.jl and libcxxwrap-julia.

The first step is to build [libcxxwrap-julia](https://github.com/JuliaInterop/libcxxwrap-julia) from source. The link explains how to do that. You must also setup the override in `/.julia/artifacts/Overrides.toml` as explained at the link.

Once that's done you start a Julia terminal and Just start the REPL. Hit `]` and then type `add CxxWrap`.

```bash
$ julia
...
julia> ]
pkg> add CxxWrap
```

This should install CxxWrap. At this step you should also check whether your libcxxwrap-julia override was set correctly or not. You can do this by checking the value of `CxxWrap.CxxWrapCore.prefix_path()` The output should show the build directory of libcxxwrap-julia

```bash
julia> using CxxWrap

julia> CxxWrap.CxxWrapCore.prefix_path()
"$HOME/src/libcxxwrap-julia-build"
```



You're now ready to build the Julia bindings. Just add the `-DWITH_JULIA=ON` to your cmake configure command and Julia bindings will be built. For example:

`cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -DWITH_JULIA=ON ../opencv`

This command assumes that the parent directory has `opencv` and `opencv_contrib` folders containing the repositories. If cmake complains about being unable to find the Julia executable add `Julia_EXECUTABLE` variable like:

`cmake -DWITH_JULIA=ON -DJulia_EXECUTABLE=$HOME/julia-1.4.1/bin/julia ...`

By default, the installed package will stay in the same directory as your build directory. You can change this using the cmake variable `JULIA_PKG_INSTALL_PATH`

Finally, type `sudo make install` to have the binding registered with the Julia package manager.

Sample Usage
-----------------------

In order to use the bindings, simply type

```
$ julia
...
julia> using OpenCV
```

Note that this works only if you called `make install`. To run the wrapper package without making the installation target you must first set the environment variable `JULIA_LOAD_PATH` to the directory containing the OpenCV package. For example if in the build directory

```
$ export JULIA_LOAD_PATH=$PWD/OpenCV
$ julia
...
julia> using OpenCV
```

The Julia package does not export any symbols so all functions/structs/constants must be prefixed with OpenCV

```
using OpenCV
const cv = OpenCV
img = cv.imread('cameraman.tif');

cv.imshow("window name", img)

cv.waitKey(Int32(0))
```

Finally, because Julia does not support OOP paradigm some changes had to be made. To access functions like `obj.function(ARGS)` you should instead use `function(obj, ARGS)`. The below example of reading frames from a VideoCapture should make it more clear.

```
cap = OpenCV.VideoCapture(Int32(0))
ret, img = OpenCV.read(cap)
```

Instead of calling `cap.read()`, we called `OpenCV.read(cap)`

Another change is that all integer and float constants might need to prefixed with appropriate type constructor. This is needed because OpenCV functions accept 32-bit integers/floats but integer and float constants in Julia are sized based on the whether Julia is running in 64bit or 32bit mode.

Running The Included Sample
-----------------------


Let's try running one of the included samples now. In this tutorial we will see how to run the `face_detect_dnn.jl` sample. This samples uses a deep neural network to detect faces in the video stream by webcam. The screenshot is from a slightly edited version that reads an image instead. First navigate to `opencv_contrib/modules/julia/samples/`. Next, you need two files "opencv_face_detector.pbtxt" and "opencv_face_detector_uint8.pb" from [link](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn);simply download and place them in the same directory as `face_detect_dnn.jl`. Now you're ready to run. Start a terminal and simply type:

```
> julia face_detect_dnn.jl
```

![image](images/julia_facedetect_sample.jpg)

You should now see a working example of face detection using deep neural networks.

Note: The sample might take some time to load.


Contributors
------------

Below is the list of contributors of OpenCV.jl bindings and tutorials.

-  Archit Rungta  (Author of the initial version and GSoC student, Indian Institute of Technology, Kharagpur)
-  Sayan Sinha  (GSoC mentor, Indian Institute of Technology, Kharagpur)
-  Mosè Giordano  (GSoC Phase 2 mentor)
-  Vadim Pisarevsky  (GSoC mentor)