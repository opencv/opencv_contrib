set(the_description "Face recognition etc")
ocv_define_module(face opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_video     # estimateRigidTransform() (trainFacemark)
    opencv_photo     # seamlessClone() (face_swap sample)
    WRAP python java
)
# NOTE: objdetect module is needed for one of the samples

set(__commit_hash "8afa57abc8229d611c4937165d20e2a2d9fc5a12")
set(__file_hash "7505c44ca4eb54b4ab1e4777cb96ac05")
ocv_download(
    FILENAME face_landmark_model.dat
    HASH ${__file_hash}
    URL
      "${OPENCV_FACE_ALIGNMENT_URL}"
      "$ENV{OPENCV_FACE_ALIGNMENT_URL}"
      "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${__commit_hash}/"
    DESTINATION_DIR "${CMAKE_BINARY_DIR}/${OPENCV_TEST_DATA_INSTALL_PATH}/cv/face/"
    ID "data"
    RELATIVE_URL
    STATUS res
)
if(NOT res)
  message(WARNING "Face: Can't get model file for face alignment.")
endif()
