function(download_vgg_descriptors dst_dir status_var)
  set(OPENCV_3RDPARTY_COMMIT "fccf7cd6a4b12079f73bbfb21745f9babcd4eb1d")

  set(ids VGG_48 VGG_64 VGG_80 VGG_120)
  set(name_VGG_48 "vgg_generated_48.i")
  set(name_VGG_64 "vgg_generated_64.i")
  set(name_VGG_80 "vgg_generated_80.i")
  set(name_VGG_120 "vgg_generated_120.i")
  set(hash_VGG_48 "e8d0dcd54d1bcfdc29203d011a797179")
  set(hash_VGG_64 "7126a5d9a8884ebca5aea5d63d677225")
  set(hash_VGG_80 "7cd47228edec52b6d82f46511af325c5")
  set(hash_VGG_120 "151805e03568c9f490a5e3a872777b75")

  set(${status_var} TRUE PARENT_SCOPE)
  foreach(id ${ids})
    ocv_download(FILENAME ${name_${id}}
                 HASH ${hash_${id}}
                 URL
                   "${OPENCV_VGGDESC_URL}"
                   "$ENV{OPENCV_VGGDESC_URL}"
                   "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${OPENCV_3RDPARTY_COMMIT}/"
                 DESTINATION_DIR "${dst_dir}"
                 ID "xfeatures2d/vgg"
                 RELATIVE_URL
                 STATUS res)
    if(NOT res)
      set(${status_var} FALSE PARENT_SCOPE)
    endif()
  endforeach()
endfunction()
