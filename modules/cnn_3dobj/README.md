CNN for 3D object recognition and pose estimation including a completed Sphere View of 3D objects from .ply files, when the windows shows the coordinate, press 'q' to go on image generation.
============================================
Building Process:
$ cd <opencv_source_directory>
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules ..
$ make -j4
$ sudo make install
$ cd <opencv_contrib>/modules/cnn_3dobj/samples
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./sphereview_test 350 2 ../ape.ply
==============================================
Then press 'q' to run the demo for images generation, parameter '350' represent the radius of the sphere and '2' stand for the depth of the sphere building iteration and '../ape.ply' represent the .ply model
==============================================
