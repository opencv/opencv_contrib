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
$ ./sphereview_test -radius=350 -ite_depth=1 -plymodel=../ape.ply -imagedir=../data/images_ape/
==============================================
demo :$ ./sphereview_test -radius=350 -ite_depth=1 -plymodel=../ape.ply -imagedir=../data/images_ape/
Then press 'q' to run the demo for images generation when you see the gray background and a coordinate.
==============================================
