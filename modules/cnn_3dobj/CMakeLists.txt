set(the_description "CNN for 3D object recognition and pose estimation including a completed Sphere View on 3D objects")
ocv_define_module(cnn_3dobj opencv_core opencv_imgproc opencv_viz opencv_highgui caffe protobuf glog OPTIONAL WRAP python)
target_link_libraries(opencv_cnn_3dobj caffe protobuf glog)
