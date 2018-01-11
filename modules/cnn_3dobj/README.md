#Convolutional Neural Networks for 3D object classification and pose estimation.
===========================================================
#Module Description on cnn_3dobj:
####This module uses Convolutional Neural Networks (Caffe) to train and recognize 3D poses of objects using triplet networks. The main reference paper can be found at:
<https://cvarlab.icg.tugraz.at/pubs/wohlhart_cvpr15.pdf>.
####The author provided Theano code for this on:
<https://cvarlab.icg.tugraz.at/projects/3d_object_detection/>.
####This implements training and feature extraction code mainly using CAFFE (<http://caffe.berkeleyvision.org/>) which will be compiled as libcaffe for the cnn_3dobj OpenCV module. The code mainly concentrats on triplet networks using pair-wise jointed loss function layers. The training data arrangement is also important and there is basic information about that.
####Code for the triplet version of Caffe are on my (Yilda Wang's) Github:
<https://github.com/Wangyida/caffe/tree/cnn_triplet>.
####You can get it through:
```
$ git clone https://github.com/Wangyida/caffe/tree/cnn_triplet.
```
===========================================================
#Module Building Process:
####Prerequisites for this module are protobuf and Caffe. For libcaffe installation, you can install it on the standard system path so that it can be linked by this OpenCV module when compiling and linking using: -D CMAKE_INSTALL_PREFIX=/usr/local as an build option when you cmake. The building process on Caffe on system could be like this:
```
$ cd <caffe_source_directory>
$ mkdir biuld
$ cd build
$ cmake -D CMAKE_INSTALL_PREFIX=/usr/local ..
$ make all -j4
$ sudo make install
```
####After the above steps, the headers and libs of CAFFE will be set on /usr/local/ path so that when you compile opencv with opencv_contrib modules as below, the protobuf and caffe libs will be recognized as already installed while building. Obviously protobuf is also needed.

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
####If you encouter the no declaration errors when you 'make', it might becaused that you have installed the older version of the cnn_3dobj module and the header file changed in a newly released version of the code. This problem is cmake and make can't detect that the header should be updated and it keeps the older header in /usr/local/include/opencv2 whithout updating. This error could be solved by removing the installed older version of the cnn_3dobj module using:
```
$ cd /
$ cd usr/local/include/opencv2/
$ sudo rm -rf cnn_3dobj.hpp
```
####And then redo the compiling steps above again.
#Demos
##Demo1: Training set data generation
####Image generation for different poses: by default, there are 4 models used and there will be 276 images in all in which each class contains 69 iamges. If you want to use additional .ply models, it is necessary to change the class number parameter to the new class number and also give it a new class label. If you want to train the network and extract feature from RGB images, set the parameter rgb_use as 1.
```
$ ./example_cnn_3dobj_sphereview_data -plymodel=../data/3Dmodel/ape.ply -label_class=0 -cam_head_x=0 -cam_head_y=0 -cam_head_z=1
$ ./example_cnn_3dobj_sphereview_data -plymodel=../data/3Dmodel/ant.ply -label_class=1 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
$ ./example_cnn_3dobj_sphereview_data -plymodel=../data/3Dmodel/cow.ply -label_class=2 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
$ ./example_cnn_3dobj_sphereview_data -plymodel=../data/3Dmodel/plane.ply -label_class=3 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
$ ./example_cnn_3dobj_sphereview_data -plymodel=../data/3Dmodel/bunny.ply -label_class=4 -cam_head_x=0 -cam_head_y=-1 -cam_head_z=0
$ ./example_cnn_3dobj_sphereview_data -plymodel=../data/3Dmodel/horse.ply -label_class=5 -cam_head_x=0 -cam_head_y=0 -cam_head_z=-1
```
####When all images are created in images_all folder as a collection of training images for network tranining and as a gallery of reference images for classification, then proceed onward.
####After this demo, the binary files of images and labels will be stored as 'binary_image' and 'binary_label' in current path. You should copy them into the leveldb folder for Caffe triplet training. For example: copy these 2 files in <caffe_source_directory>/data/linemod and rename them as 'binary_image_train', 'binary_image_test' and 'binary_label_train', 'binary_label_train'. Here I use the same as trianing and testing data but you can use different data for training and testing. It's important to observe the error on the testing data to check whether the training data is suitable for the your aim. Error should be obseved to keep decreasing and remain much smaller than the initial error.
####Start triplet tranining using Caffe like this:
```
$ cd
$ cd <caffe_source_directory>
$ ./examples/triplet/create_3d_triplet.sh
$ ./examples/triplet/train_3d_triplet.sh
```
####After doing this, you will get .caffemodel files as the trained parameters of the network. I have already provide the network definition .prototxt files and the pretrained .caffemodel in <opencv_contrib>/modules/cnn_3dobj/testdata/cv folder, so you could just use them instead of training with Caffe.
===========================================================
##Demo2: Feature extraction and classification
```
$ cd
$ cd <opencv_contrib>/modules/cnn_3dobj/samples/build
```
####Classification: This will extract features of a single image and compare it with features of a gallery of samples for prediction. This demo uses a set of images for feature extraction in a given path, these features will be a reference for prediction on the target image. The Caffe model and the network prototxt file are in <opencv_contrib>/modules/cnn_3dobj/testdata/cv. Just run:
```
$ ./example_cnn_3dobj_classify
```
####if you want to extract mean classification and pose estimation performance from all the training images, you can run this:
```
$ ./example_cnn_3dobj_classify -mean_file=../data/images_mean/triplet_mean.binaryproto
```
===========================================================
##Demo3: Model performance test
####This demo will run a performance test of a trained CNN model on several images. If the the model fails on telling different samples from seperate classes apart, or is confused on samples with similar pose but from different classes, this will give some information for model analysis.
```
$ ./example_cnn_3dobj_model_analysis
```
===========================================================
#Test
####If you want to run a test on the cnn_3dobj module, the path of test data must be set in advance:
```
$ export OPENCV_TEST_DATA_PATH=<opencv_contrib>/modules/cnn_3dobj/testdata
```
