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

You can build OpenCV, so it will include the modules from this repository. Contrib modules are under constant development and it is recommended to use them alongside the master branch or latest releases of OpenCV.

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

If you prefer using the gui version of cmake (cmake-gui), then, you can add `opencv_contrib` modules within `opencv` core by doing the following:

1. start cmake-gui

2. select the opencv source code folder and the folder where binaries will be built (the 2 upper forms of the interface)

3. press the `configure` button. you will see all the opencv build parameters in the central interface

4. browse the parameters and look for the form called `OPENCV_EXTRA_MODULES_PATH` (use the search form to focus rapidly on it)

5. complete this `OPENCV_EXTRA_MODULES_PATH` by the proper pathname to the `<opencv_contrib>/modules` value using its browse button.

6. press the `configure` button followed by the `generate` button (the first time, you will be asked which makefile style to use)

7. build the `opencv` core with the method you chose (make and make install if you chose Unix makfile at step 6) 

8. to run, linker flags to contrib modules will need to be added to use them in your code/IDE. For example to use the aruco module, "-lopencv_aruco" flag will be added.

### Update the repository documentation

In order to keep a clean overview containing all contributed modules the following files need to be created/adapted.

1. Update the README.md file under the modules folder. Here you add your model with a single line description.

2. Add a README.md inside your own module folder. This README explains which functionality (seperate functions) is available, links to the corresponding samples and explains in somewhat more detail what the module is expected to do. If any extra requirements are needed to build the module without problems, add them here also.
