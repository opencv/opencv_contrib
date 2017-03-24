set(PROTOBUF_CPP_NAME "libprotobuf")
set(PROTOBUF_CPP_DOWNLOAD_HASH "bd5e3eed635a8d32e2b99658633815ef")
set(PROTOBUF_CPP_PATH "${CMAKE_CURRENT_BINARY_DIR}/3rdparty/protobuf/sources") # /protobuf-3.1.0 subdirectory

set(OPENCV_PROTOBUF_CPP_DOWNLOAD_URL ${OPENCV_PROTOBUF_URL};$ENV{OPENCV_PROTOBUF_URL};https://github.com/google/protobuf/releases/download/)

function(ocv_protobuf_download file ID)
  if(DEFINED ${ID}_DOWNLOADED_HASH
       AND ${ID}_DOWNLOADED_HASH STREQUAL ${ID}_DOWNLOAD_HASH
       AND EXISTS ${${ID}_PATH})
    # Files have been downloaded and checked by the previous CMake run
    return()
  else()
    if(EXISTS ${${ID}_PATH})
      message(STATUS "${${ID}_NAME}: Removing previous unpacked files: ${${ID}_PATH}")
      file(REMOVE_RECURSE ${${ID}_PATH})
    endif()
  endif()
  unset(${ID}_DOWNLOADED_HASH CACHE)

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.download)
  file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/.download/.gitignore" "*\n")
  ocv_download(PACKAGE ${file}
               HASH ${${ID}_DOWNLOAD_HASH}
               URL ${OPENCV_${ID}_DOWNLOAD_URL}
               DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.download)
  set(${ID}_ARCHIVE "${DOWNLOAD_PACKAGE_LOCATION}")

  ocv_assert(EXISTS "${${ID}_ARCHIVE}")
  ocv_assert(NOT EXISTS "${${ID}_PATH}")
  file(MAKE_DIRECTORY ${${ID}_PATH})
  ocv_assert(EXISTS "${${ID}_PATH}")
  file(WRITE "${${ID}_PATH}/.gitignore" "*\n")

  message(STATUS "${${ID}_NAME}: Unpacking ${file} to ${${ID}_PATH}...")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xz "${${ID}_ARCHIVE}"
                  WORKING_DIRECTORY "${${ID}_PATH}"
                  RESULT_VARIABLE __result)

  if(NOT __result EQUAL 0)
    message(FATAL_ERROR "${${ID}_NAME}: Failed to unpack ${ID} archive from ${${ID}_ARCHIVE} to ${${ID}_PATH} with error ${__result}")
  endif()

  ocv_assert(EXISTS "${${ID}_PATH}")

  set(${ID}_DOWNLOADED_HASH "${${ID}_DOWNLOAD_HASH}" CACHE INTERNAL "${ID} hash")

  #message(STATUS "${${ID}_NAME}: Successfully downloaded")
endfunction()

ocv_protobuf_download(v3.1.0/protobuf-cpp-3.1.0.tar.gz PROTOBUF_CPP)
