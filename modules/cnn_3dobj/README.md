#Convolutional Neural Network for 3D object classification and pose estimation.
===========================================================
#Module Description on cnn_3dobj:
####This learning structure construction and feature extraction concept is based on Convolutional Neural Network, the main reference paper could be found at:
<https://cvarlab.icg.tugraz.at/pubs/wohlhart_cvpr15.pdf>.
####The author provided Codes on Theano on:
<https://cvarlab.icg.tugraz.at/projects/3d_object_detection/>.
####I implemented the training and feature extraction codes mainly based on CAFFE project(<http://caffe.berkeleyvision.org/>) which will be compiled as libcaffe for the cnn_3dobj OpenCV module, codes are mainly concentrating on triplet and pair-wise jointed loss layer, the training data arrangement is also important which basic training information.
####Codes about my triplet version of caffe are released on Github:
<https://github.com/Wangyida/caffe/tree/cnn_triplet>.
####You can git it through:
```
$ git clone https://github.com/Wangyida/caffe/tree/cnn_triplet.
```
===========================================================
#Module Building Process:
####Prerequisite for this module: protobuf and caffe, for the libcaffe installation, you can install it on standard system path for the aim of being able to be linked by this OpenCV module when compiling and function using. Using: -D CMAKE_INSTALL_PREFIX=/usr/local as an building option when you cmake, the building process on Caffe on system could be like this:
```
$ cd <caffe_source_directory>
$ mkdir biuld
$ cd build
$ cmake -D CMAKE_INSTALL_PREFIX=/usr/local ..
$ make all -j4
$ sudo make install
```
####After all these steps, the headers and libs of CAFFE will be set on /usr/local/ path, and when you compiling opencv with opencv_contrib modules as below, the protobuf and caffe will be recognized as already installed while building. Protobuf is needed.

#Compiling OpenCV
```
$ cd <opencv_source_directory>
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=OFF -D WITH_OPENGL=ON -D WITH_VTK=ON -D INSTALL_TESTS=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
$ make -j4
$ sudo make install
```
##Tips on compiling problems:
####If you encouter the no declaration errors when you 'make', it might becaused that you have installed the older version of cnn_3dobj module and the header file changed in a newly released version of codes. This problem is the cmake and make can't detect the header should be updated and it keeps the older header remains in /usr/local/include/opencv2 whithout updating. This error could be solved by remove the installed older version of cnn_3dobj module by:
```
$ cd /
$ cd usr/local/include/opencv2/
$ sudo rm -rf cnn_3dobj.hpp
```
####And redo the compiling steps above again.
===========================================================
#Building samples
```
$ cd <opencv_contrib>/modules/cnn_3dobj/samples
$ mkdir build
$ cd build
$ cmake ..
$ make
```
===========================================================
#Demos
##Demo1: training data generation
####Imagas generation from different pose, by default there are 4 models used, there will be 276 images in all which each class contains 69 iamges, if you want to use additional .ply models, it is necessary to change the class number parameter to the new class number and also give it a new class label. If you will train net work and extract feature from RGB images set the parameter rgb_use as 1.
```
$ ./sphereview_test -plymodel=../data/3Dmodel/ape.ply -label_class=0 -cam_head_x=0 -cam_head_y=0 -cam_head_z=1
```
####press 'Q' to start 2D image genaration
```
$ ./sphereview_test -plymodel=../data/3Dmodel/ant.ply -label_class=1 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
```
```
$ ./sphereview_test -plymodel=../data/3Dmodel/cow.ply -label_class=2 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
```
```
$ ./sphereview_test -plymodel=../data/3Dmodel/plane.ply -label_class=3 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
```
```
$ ./sphereview_test -plymodel=../data/3Dmodel/bunny.ply -label_class=4 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
```
```
$ ./sphereview_test -plymodel=../data/3Dmodel/horse.ply -label_class=5 -cam_head_x=0 -cam_head_y=0 -cam_head_z=-1
```
####When all images are created in images_all folder as a collection of training images for network tranining and as a gallery of reference images for the classification part, then proceed on.
####After this demo, the binary files of images and labels will be stored as 'binary_image' and 'binary_label' in current path, you should copy them into the leveldb folder in Caffe triplet training, for example: copy these 2 files in <caffe_source_directory>/data/linemod and rename them as 'binary_image_train', 'binary_image_test' and 'binary_label_train', 'binary_label_train'. Here I use the same as trianing and testing data, you can use different data for training and testing the performance in the CAFFE training process. It's important to observe the loss of testing data to check whether training data is suitable for the your aim. Loss should be obseved as keep decreasing and remain on a much smaller number than the initial loss.
####You could start triplet tranining using Caffe like this:
```
$ cd
$ cd <caffe_source_directory>
$ ./examples/triplet/create_3d_triplet.sh
$ ./examples/triplet/train_3d_triplet.sh
```
####After doing this, you will get .caffemodel files as the trained parameter of net work. I have already provide the net definition .prototxt files and the pretrained .caffemodel in <opencv_contrib>/modules/cnn_3dobj/testdata/cv folder, you could just use them without training in caffe.
===========================================================
##Demo2: feature extraction and classification
```
$ cd
$ cd <opencv_contrib>/modules/cnn_3dobj/samples/build
```
####Classifier, this will extracting the feature of a single image and compare it with features of gallery samples for prediction. This demo uses a set of images for feature extraction in a given path, these features will be a reference for prediction on target image. The caffe model and network prototxt file is attached in <opencv_contrib>/modules/cnn_3dobj/testdata/cv. Just run:
```
$ ./classify_test
```
####if the classification and pose estimation issue need to extract mean got from all training images, you can run this:
```
$ ./classify_test -mean_file=../data/images_mean/triplet_mean.binaryproto
```
===========================================================
##Demo3: model performance test
####This demo will have a test on the performance of trained CNN model on several images. If the the model fail on telling different samples from seperate classes or confused on samples with similar pose but from different classes, it will give some information on the model analysis.
```
$ ./model_test
```
===========================================================
#Test
####If you want to have a test on cnn_3dobj module, the path of test data must be set in advance:
```
$ export OPENCV_TEST_DATA_PATH=<opencv_contrib>/modules/cnn_3dobj/testdata
```
