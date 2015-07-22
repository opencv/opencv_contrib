##CNN for 3D object recognition and pose estimation including a completed Sphere View of 3D objects from .ply files, when the windows shows the coordinate, press 'q' to go on image generation.
============================================

#Building Process:
###Prerequisite for this module: protobuf, leveldb, glog, gflags and caffe, for the libcaffe installation, you can install it on standard system path for being able to be linked by this OpenCV module when compiling. Just using: -D CMAKE_INSTALL_PREFIX=/usr/local, so the building process on Caffe on system could be like this:
```
$ cd <caffe_source_directory>
$ mkdir biuld
$ cd build
$ cmake　-D CMAKE_INSTALL_PREFIX=/usr/local ..
$ make all
$ make install
```
###After all these steps, the headers and libs of caffe will be set on /usr/local/ path, and when you compiling opencv with opencv_contrib modules as below, the protobif, leveldb, glog, gflags and caffe will be recognized as already installed while building.

#Compiling OpenCV
```
$ cd <opencv_source_directory>
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=OFF -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules ..
$ make -j4
$ sudo make install
```

================================================
#Building samples
```
$ cd <opencv_contrib>/modules/cnn_3dobj/samples
$ mkdir build
$ cd build
$ cmake ..
$ make
```

=============
#Demo1:
###Imagas generation from different pose, 4 models are used, there will be 276 images in all which each class contains 69 iamges
```
$ ./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/ape.ply -imagedir=../data/images_ape/ -labeldir=../data/label_ape.txt -num_class=4 -label_class=0
```
###press q to start
```
$ ./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/ant.ply -imagedir=../data/images_ant/ -labeldir=../data/label_ant.txt -num_class=4 -label_class=1
```
###press q to start
```
$ ./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/cow.ply -imagedir=../data/images_cow/ -labeldir=../data/label_cow.txt -num_class=4 -label_class=2
```
###press q to start
```
$ ./sphereview_test -ite_depth=2 -plymodel=../3Dmodel/plane.ply -imagedir=../data/images_plane/ -labeldir=../data/label_plane.txt -num_class=4 -label_class=3
```
###press q to start, when all images are created in each class folder, you should copy all images from ../data/images_ape, ../data/images_ant, ../data/images_cow and ../data/images_plane into ../data/images_all folder as a collection of images for network tranining and feature extraction, when all images are copyed correctlly, proceed on.
###After this demo, the binary files of images and labels will be stored as 'binary_image' and 'binary_label' in current path, you should copy them into the leveldb folder in Caffe triplet training, for example: copy these 2 files in <caffe_source_directory>/data/linemod and rename them as 'binary_image_train', 'binary_image_test' and 'binary_label_train', 'binary_label_train'.
###We could start triplet tranining using Caffe
```
$ cd
$ cd <caffe_source_directory>
$ ./examples/triplet/create_3d_triplet.sh
$ ./examples/triplet/train_3d_triplet.sh
```
###After doing this, you will get .caffemodel files as the trained net work. I have already provide the net definition .prototxt files and the trained .caffemodel in <opencv_contrib>/modules/cnn_3dobj/samples/build folder, you could just use them without training in caffe. If you are not interested on feature analysis with the help of binary files provided in Demo2, just skip to Demo3 for feature extraction or Demo4 for classifier.

==============
```
$ cd
$ cd <opencv_contrib>/modules/cnn_3dobj/samples/build
```
#Demo2:
###Convert data into leveldb format from folder ../data/images_all for feature extraction afterwards. The leveldb files including all data will be stored in ../data/dbfile. If you will use the OpenCV defined feature extraction process, you could also skip Demo2 for data converting, just run Demo3 after Demo1 for feature extraction because Demo3 also includes the db file converting process before feature extraction.
```
$ ./images2db_test -images2db_demo=../data/images_all -src_dst=../data/dbfile -attach_dir=../data/dbfile -channel=1 -width=64 -height=64
```

==============
#Demo3:
###feature extraction, this demo will convert a set of images in a particular path into leveldb database for feature extraction using Caffe.
```
$ ./feature_extract_test
```
###This will extract feature from a set of images in a folder as vector<cv::Mat> for further classification and a binary file with containing all feature vectors of each sample. Pay attention: if it's warning you that 'Check failed: leveldb::DB::Open(options, outputdb, &db).ok()', the reason is that there is alreay leveldb files in ../data/dbfile as previous running of Demo2 or Demo3, just delete all files in ../data/dbfile and run Demo3 again.
###After running this, you will get a binary file storing features in ../data/feature folder, I can provide a Matlab script reading this file if someone need it. If you don't need the binary file, the feature could also be stored in vector<cv::Mat> for directly classification using the softmax layer as shown in Demo4.

==============
#Demo4:
###Classifier
==============================================
