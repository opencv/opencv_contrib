function(download_boost_descriptors dst_dir status_var)
  set(OPENCV_3RDPARTY_COMMIT "34e4206aef44d50e6bbcd0ab06354b52e7466d26")

  set(ids BGM BGM_BI BGM_HD BINBOOST_064 BINBOOST_128 BINBOOST_256 LBGM)
  set(name_BGM boostdesc_bgm.i)
  set(name_BGM_BI boostdesc_bgm_bi.i)
  set(name_BGM_HD boostdesc_bgm_hd.i)
  set(name_BINBOOST_064 boostdesc_binboost_064.i)
  set(name_BINBOOST_128 boostdesc_binboost_128.i)
  set(name_BINBOOST_256 boostdesc_binboost_256.i)
  set(name_LBGM boostdesc_lbgm.i)
  set(hash_BGM "0ea90e7a8f3f7876d450e4149c97c74f")
  set(hash_BGM_BI "232c966b13651bd0e46a1497b0852191")
  set(hash_BGM_HD "324426a24fa56ad9c5b8e3e0b3e5303e")
  set(hash_BINBOOST_064 "202e1b3e9fec871b04da31f7f016679f")
  set(hash_BINBOOST_128 "98ea99d399965c03d555cef3ea502a0b")
  set(hash_BINBOOST_256 "e6dcfa9f647779eb1ce446a8d759b6ea")
  set(hash_LBGM "0ae0675534aa318d9668f2a179c2a052")

  set(${status_var} TRUE PARENT_SCOPE)
  foreach(id ${ids})
    ocv_download(FILENAME ${name_${id}}
                 HASH ${hash_${id}}
                 URL
                   "${OPENCV_BOOSTDESC_URL}"
                   "$ENV{OPENCV_BOOSTDESC_URL}"
                   "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${OPENCV_3RDPARTY_COMMIT}/"
                 DESTINATION_DIR ${dst_dir}
                 ID "xfeatures2d/boostdesc"
                 RELATIVE_URL
                 STATUS res)
    if(NOT res)
      set(${status_var} FALSE PARENT_SCOPE)
    endif()
  endforeach()
endfunction()
