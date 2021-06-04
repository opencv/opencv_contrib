set(the_description "WeChat QR code Detector")
ocv_define_module(wechat_qrcode opencv_core opencv_imgproc opencv_dnn WRAP java objc python js)

# iconv support isn't automatic on some systems
if(CMAKE_VERSION VERSION_GREATER "3.11")
  find_package(Iconv QUIET)
  if(Iconv_FOUND)
    ocv_target_link_libraries(${the_module} Iconv::Iconv)
  else()
    ocv_target_compile_definitions(${the_module} PRIVATE "NO_ICONV=1")
  endif()
endif()

# need to change
set(wechat_qrcode_commit_hash "a8b69ccc738421293254aec5ddb38bd523503252")
set(hash_detect_caffemodel "238e2b2d6f3c18d6c3a30de0c31e23cf")
set(hash_detect_prototxt "6fb4976b32695f9f5c6305c19f12537d")
set(hash_sr_caffemodel "cbfcd60361a73beb8c583eea7e8e6664")
set(hash_sr_prototxt "69db99927a70df953b471daaba03fbef")

set(model_types caffemodel prototxt)
set(model_names detect sr)

foreach(model_name ${model_names})
    foreach(model_type ${model_types})
        ocv_download(FILENAME ${model_name}.${model_type}
                        HASH ${hash_${model_name}_${model_type}}
                        URL
                        "${OPENCV_WECHAT_QRCODE_URL}"
                        "$ENV{OPENCV_WECHAT_QRCODE_URL}"
                        "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/${wechat_qrcode_commit_hash}/"
                        DESTINATION_DIR "${CMAKE_BINARY_DIR}/downloads/wechat_qrcode"
                        ID "wechat_qrcode"
                        RELATIVE_URL
                        STATUS res)
        if(NOT res)
            message(WARNING "WeChatQRCode: Can't get ${model_name} ${model_type} file for wechat qrcode.")
        endif()
    endforeach()
endforeach()
