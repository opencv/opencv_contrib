
set(OPENCV_3RDPARTY_COMMIT "fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d")
set(FILE_HASH_VGG_48 "e8d0dcd54d1bcfdc29203d011a797179")
set(FILE_HASH_VGG_64 "7126a5d9a8884ebca5aea5d63d677225")
set(FILE_HASH_VGG_80 "7cd47228edec52b6d82f46511af325c5")
set(FILE_HASH_VGG_120 "151805e03568c9f490a5e3a872777b75")


set(VGG_DOWNLOAD_URL ${OPENCV_CONTRIB_VGG_URL};$ENV{OPENCV_CONTRIB_VGG_URL};https://raw.githubusercontent.com/opencv/opencv_3rdparty/${OPENCV_3RDPARTY_COMMIT}/)

function(vgg_download file id)
  message(STATUS "Check contents of ${file} ...")
  ocv_download(PACKAGE ${file}
               HASH ${FILE_HASH_${id}}
               URL ${VGG_DOWNLOAD_URL}
               DESTINATION_DIR ${CMAKE_CURRENT_LIST_DIR}/../src
               DOWNLOAD_DIR ${CMAKE_CURRENT_LIST_DIR}/.download)
endfunction()

vgg_download(vgg_generated_48.i VGG_48)
vgg_download(vgg_generated_64.i VGG_64)
vgg_download(vgg_generated_80.i VGG_80)
vgg_download(vgg_generated_120.i VGG_120)
