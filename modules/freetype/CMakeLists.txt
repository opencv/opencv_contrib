set(the_description "FreeType module. It enables to draw strings with outlines and mono-bitmaps/gray-bitmaps.")

if(PKG_CONFIG_FOUND)
  pkg_search_module(FREETYPE freetype2)
  pkg_search_module(HARFBUZZ harfbuzz)
endif()

if(NOT FREETYPE_FOUND)
  message(STATUS "freetype2:   NO")
else()
  message(STATUS "freetype2:   YES")
endif()

if(NOT HARFBUZZ_FOUND)
  message(STATUS "harfbuzz:    NO")
else()
  message(STATUS "harfbuzz:    YES")
endif()


if( FREETYPE_FOUND AND HARFBUZZ_FOUND )
  ocv_define_module(freetype opencv_core opencv_imgproc PRIVATE_REQUIRED ${freetype2_LIBRARIES} ${harfbuzz_LIBRARIES} WRAP python)
  ocv_include_directories( ${FREETYPE_INCLUDE_DIRS} ${HARFBUZZ_INCLUDE_DIRS} )
else()
  ocv_module_disable(freetype)
endif()
