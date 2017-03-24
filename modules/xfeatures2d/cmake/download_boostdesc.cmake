
set(OPENCV_3RDPARTY_COMMIT "34e4206aef44d50e6bbcd0ab06354b52e7466d26")
set(FILE_HASH_BOOSTDESC_BGM "0ea90e7a8f3f7876d450e4149c97c74f")
set(FILE_HASH_BOOSTDESC_BGM_BI "232c966b13651bd0e46a1497b0852191")
set(FILE_HASH_BOOSTDESC_BGM_HD "324426a24fa56ad9c5b8e3e0b3e5303e")
set(FILE_HASH_BOOSTDESC_BINBOOST_064 "202e1b3e9fec871b04da31f7f016679f")
set(FILE_HASH_BOOSTDESC_BINBOOST_128 "98ea99d399965c03d555cef3ea502a0b")
set(FILE_HASH_BOOSTDESC_BINBOOST_256 "e6dcfa9f647779eb1ce446a8d759b6ea")
set(FILE_HASH_BOOSTDESC_LBGM "0ae0675534aa318d9668f2a179c2a052")



set(BOOSTDESC_DOWNLOAD_URL ${OPENCV_CONTRIB_BOOSTDESC_URL};$ENV{OPENCV_CONTRIB_BOOSTDESC_URL};https://raw.githubusercontent.com/opencv/opencv_3rdparty/${OPENCV_3RDPARTY_COMMIT}/)

function(boostdesc_download file id)
  message(STATUS "Check contents of ${file} ...")
  ocv_download(PACKAGE ${file}
               HASH ${FILE_HASH_${id}}
               URL ${BOOSTDESC_DOWNLOAD_URL}
               DESTINATION_DIR ${CMAKE_CURRENT_LIST_DIR}/../src
               DOWNLOAD_DIR ${CMAKE_CURRENT_LIST_DIR}/.download)
endfunction()

boostdesc_download(boostdesc_bgm.i BOOSTDESC_BGM)
boostdesc_download(boostdesc_bgm_bi.i BOOSTDESC_BGM_BI)
boostdesc_download(boostdesc_bgm_hd.i BOOSTDESC_BGM_HD)
boostdesc_download(boostdesc_binboost_064.i BOOSTDESC_BINBOOST_064)
boostdesc_download(boostdesc_binboost_128.i BOOSTDESC_BINBOOST_128)
boostdesc_download(boostdesc_binboost_256.i BOOSTDESC_BINBOOST_256)
boostdesc_download(boostdesc_lbgm.i BOOSTDESC_LBGM)
