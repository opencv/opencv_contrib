## Repository for OpenCV's extra modules

This repository is intended for development of so-called "extra" modules,
contributed functionality. New modules quite often do not have stable API,
and they are not well-tested. Thus, they shouldn't be released as a part of
official OpenCV distribution, since the library maintains binary compatibility,
and tries to provide decent performance and stability.

So, all the new modules should be developed separately, and published in the
`opencv_contrib` repository at first. Later, when the module matures and gains
popularity, it is moved to the central OpenCV repository, and the development team
provides production quality support for this module.

### How to build OpenCV with extra modules

You can build OpenCV, so it will include the modules from this repository.
Here is the CMake command for you:

```
$ cd <opencv_build_directory>
$ cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory>
$ make -j5
```

As the result, OpenCV will be built in the `<opencv_build_directory>` with all
modules from `opencv_contrib` repository. If you don't want all of the modules,
use CMake's `BUILD_opencv_*` options. Like in this example:

```
$ cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -DBUILD_opencv_legacy=OFF <opencv_source_directory>
```
