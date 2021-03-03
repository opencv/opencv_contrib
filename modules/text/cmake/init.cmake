OCV_OPTION(WITH_TESSERACT "Include Tesseract OCR library support" (NOT CMAKE_CROSSCOMPILING)
  VERIFY HAVE_TESSERACT)

if(NOT HAVE_TESSERACT
    AND (WITH_TESSERACT OR OPENCV_FIND_TESSERACT)
)
  if(NOT Tesseract_FOUND)
    find_package(Tesseract QUIET)  # Prefer CMake's standard locations (including Tesseract_DIR)
  endif()
  if(NOT Tesseract_FOUND)
    include("${CMAKE_CURRENT_LIST_DIR}/FindTesseract.cmake")  # OpenCV's fallback
  endif()
  if(Tesseract_FOUND)
    if(Tesseract_VERSION)
      message(STATUS "Tesseract:   YES (ver ${Tesseract_VERSION})")
    else()
      message(STATUS "Tesseract:   YES (ver unknown)")
    endif()
    if(NOT ENABLE_CXX11 AND NOT OPENCV_SKIP_TESSERACT_BUILD_CHECK)
      try_compile(__VALID_TESSERACT
        "${OpenCV_BINARY_DIR}/cmake_check/tesseract"
        "${CMAKE_CURRENT_LIST_DIR}/checks/tesseract_test.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${Tesseract_INCLUDE_DIRS}"
        LINK_LIBRARIES ${Tesseract_LIBRARIES}
        OUTPUT_VARIABLE TRY_OUT
        )
      if(NOT __VALID_TESSERACT)
        if(OPENCV_DEBUG_TESSERACT_BUILD)
          message(STATUS "${TRY_OUT}")
        endif()
        message(STATUS "Can't use Tesseract (details: https://github.com/opencv/opencv_contrib/pull/2220)")
        return()
      endif()
    endif()
    set(HAVE_TESSERACT 1)
    set(HAVE_TESSERACT 1)

    # TODO use ocv_add_external_target
    set(name "tesseract")
    set(inc "${Tesseract_INCLUDE_DIRS}")
    set(link "${Tesseract_LIBRARIES}")
    set(def "")
    if(BUILD_SHARED_LIBS)
      set(imp IMPORTED)
    endif()
    add_library(ocv.3rdparty.${name} INTERFACE ${imp})
    set_target_properties(ocv.3rdparty.${name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${inc}"
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${inc}"
      INTERFACE_LINK_LIBRARIES "${link}"
      INTERFACE_COMPILE_DEFINITIONS "${def}")
    if(NOT BUILD_SHARED_LIBS)
      install(TARGETS ocv.3rdparty.${name} EXPORT OpenCVModules)
    endif()

  else()
    message(STATUS "Tesseract:   NO")
  endif()
endif()
