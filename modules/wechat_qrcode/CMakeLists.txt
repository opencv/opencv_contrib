include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
set(the_description "WeChat QR code Detector")
ocv_define_module(wechat_qrcode opencv_core opencv_imgproc opencv_dnn WRAP java objc python js)
