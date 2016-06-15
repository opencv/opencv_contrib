LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#opencv
OPENCVROOT:= /Users/Sarthak/Dropbox/OpenCV_GSoC/opencv/build_android_arm/install
OPENCV_CAMERA_MODULES:=on
OPENCV_LIB_TYPE:=STATIC
OPENCV_INSTALL_MODULES:=on
include ${OPENCVROOT}/sdk/native/jni/OpenCV.mk

LOCAL_SRC_FILES := org_opencv_sample_app_NativeClass.cpp
LOCAL_LDLIBS += -llog
LOCAL_MODULE := opencv_mylib

include $(BUILD_SHARED_LIBRARY)